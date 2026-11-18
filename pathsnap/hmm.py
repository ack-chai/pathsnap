"""Kinematic Hidden Markov Model (HMM) Map Matching & Viterbi Route Decoder.

Architecture:
1. Emission Modeling:
   - Supervised classification posterior log P_clf(a | z_t)
   - Blended with Gaussian GPS orthogonal distance likelihood:
     P_spatial(z_t | a) ~ N(0, sigma_gps^2)
   - Total Emission: E_t(a) = power * log(p_clf) + spatial_weight * (-0.5 * (d / sigma)^2)

2. Kinematic Transition Dynamics:
   - Driving distance vs. straight-line displacement penalty (Newson & Krumm, 2009):
     T_dist = -| route_dist(a -> b) - gc_dist(z_{t-1}, z_t) | / beta
   - Angular Heading Deflection Penalty:
     Smooth cosine penalty gamma * (1 - cos(theta_b - theta_a)) to penalize sharp,
     physically implausible turning reversals at road intersections.
   - Kinematic Speed Bounding:
     Soft penalty when implied segment speed (route_dist / dt) exceeds physical maximum.

3. Cycle-Suppressed Continuous Route Reconstruction (build_route):
   - Jitter suppression: eliminates single-fix transient ping-pong oscillations (A -> B -> A).
   - Topological shortest-path closure for disconnected observation intervals.
   - Turnaround loop suppression: prunes spurious back-and-forth micro-cycles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _split_points(t: np.ndarray, gap_seconds: float) -> list[np.ndarray]:
    """Split observation index sequence at time gaps exceeding threshold."""
    cut = np.flatnonzero(np.diff(t) > gap_seconds) + 1
    return np.split(np.arange(len(t)), cut)


def _split_candidate_runs(points: np.ndarray, starts: np.ndarray) -> list[np.ndarray]:
    """Split a time segment into contiguous runs that have candidate states."""
    runs = []
    run = []
    for point in points:
        if starts[point] < starts[point + 1]:
            run.append(point)
        elif run:
            runs.append(np.asarray(run, dtype=np.int64))
            run = []
    if run:
        runs.append(np.asarray(run, dtype=np.int64))
    return runs


class KinematicHMM:
    """Object-oriented Kinematic HMM Decoder for high-frequency GPS trajectories."""

    def __init__(self, net, params: dict | None = None):
        self.net = net
        self.params = {**config.HMM_DEFAULTS, **(params or {})}
        self.beta = float(self.params["beta"])
        self.power = float(self.params["emission_power"])
        self.top_m = int(self.params.get("top_m", 8))
        self.unreach = float(self.params["unreachable_logp"])
        self.vmax = float(self.params["max_speed_mps"])
        self.turn_weight = float(self.params.get("turn_weight", 0.08))
        self.spatial_weight = float(self.params.get("spatial_weight", 0.02))
        self.sigma_gps = float(self.params.get("sigma_gps", 15.0))
        self.has_bearing = getattr(self.net, "bearing", None) is not None

    def decode(self, xy: np.ndarray, t: np.ndarray, frame: pd.DataFrame, score_col: str = "p") -> np.ndarray:
        """Execute Viterbi trellis decoding over candidates."""
        xy = np.asarray(xy)
        t = np.asarray(t)
        if len(xy) != len(t):
            raise ValueError("xy and t must contain the same number of observations")

        pred = np.full(len(xy), -1, dtype=np.int64)
        if frame.empty:
            return pred

        point_index = frame["pidx"].to_numpy()
        if not np.issubdtype(point_index.dtype, np.integer):
            if not np.all(np.isfinite(point_index)) or not np.all(point_index == np.floor(point_index)):
                raise ValueError("pidx values must be integer observation indices")
            point_index = point_index.astype(np.int64)
        if np.any((point_index < 0) | (point_index >= len(xy))):
            raise ValueError("pidx values must be within the observation range")

        # Prepare candidates sorted by pidx and probability
        f = frame.assign(pidx=point_index).sort_values(["pidx", score_col], ascending=[True, False])
        f = f.groupby("pidx", sort=True).head(self.top_m)
        starts = np.searchsorted(f["pidx"].to_numpy(), np.arange(len(xy) + 1))
        arcs_all = f["arc"].to_numpy()
        tal_all = f["t_along"].to_numpy()

        # Hybrid emission: classifier posterior + Gaussian spatial prior
        base_em = self.power * np.log(np.clip(f[score_col].to_numpy(), 1e-6, 1.0))
        if self.spatial_weight > 0.0 and "dist" in f.columns:
            dists = f["dist"].to_numpy(dtype=float)
            spatial_logp = -0.5 * (dists / max(self.sigma_gps, 1.0)) ** 2
            em_all = base_em + self.spatial_weight * spatial_logp
        else:
            em_all = base_em

        # Dynamic programming Viterbi search
        for time_segment in _split_points(t, self.params["gap_seconds"]):
            for seg in _split_candidate_runs(time_segment, starts):
                V = None
                back = []
                prev = None
                for i in seg:
                    s, e = starts[i], starts[i + 1]
                    arcs, tal, em = arcs_all[s:e], tal_all[s:e], em_all[s:e]
                    if V is None:
                        V = em.copy()
                        back.append(None)
                        prev = (i, arcs, tal)
                        continue

                    pi, parcs, ptal = prev
                    gc = float(np.linalg.norm(xy[i] - xy[pi]))
                    dt = max(float(t[i] - t[pi]), 1.0)
                    limit = min(self.vmax * dt, 3.0 * gc + 200.0) + 50.0
                    T = np.full((len(parcs), len(arcs)), self.unreach)

                    for r, (a, ta) in enumerate(zip(parcs, ptal)):
                        rest_a = (1.0 - ta) * self.net.length[a]
                        reach = None
                        for c, (b, tb) in enumerate(zip(arcs, tal)):
                            if a == b:
                                rd = abs(tb - ta) * self.net.length[a]
                                turn_pen = 0.0
                            else:
                                head_b = tb * self.net.length[b]
                                budget = limit - rest_a - head_b
                                if budget < 0:
                                    continue
                                if reach is None:
                                    reach = self.net.dists_from(int(self.net.dst[a]), limit)
                                d = reach.get(int(self.net.src[b]))
                                if d is None or d > budget:
                                    continue
                                rd = rest_a + d + head_b

                                # Kinematic angular deflection penalty
                                if self.turn_weight > 0.0 and self.has_bearing:
                                    dth = abs(self.net.bearing[b] - self.net.bearing[a]) % 360.0
                                    if dth > 180.0:
                                        dth = 360.0 - dth
                                    turn_pen = self.turn_weight * (1.0 - np.cos(np.radians(dth)))
                                else:
                                    turn_pen = 0.0

                            T[r, c] = -abs(rd - gc) / self.beta - turn_pen

                    scores = V[:, None] + T
                    bp = np.argmax(scores, axis=0)
                    V = scores[bp, np.arange(len(arcs))] + em
                    back.append(bp)
                    prev = (i, arcs, tal)

                # Backtrack optimal state sequence
                state = int(np.argmax(V))
                for k in range(len(seg) - 1, -1, -1):
                    i = seg[k]
                    pred[i] = arcs_all[starts[i] + state]
                    if back[k] is not None:
                        state = int(back[k][state])

        return pred


def viterbi(net, xy, t, frame: pd.DataFrame, params: dict | None = None, score_col="p") -> np.ndarray:
    """Decode one trip using Kinematic HMM Viterbi solver."""
    decoder = KinematicHMM(net, params)
    return decoder.decode(xy, t, frame, score_col=score_col)


def classifier_only(frame: pd.DataFrame, n_points: int, score_col="p") -> np.ndarray:
    """Observation-level greedy baseline picking max-probability candidate arc."""
    top = frame.loc[frame.groupby("pidx")[score_col].idxmax()]
    pred = np.full(n_points, -1, dtype=np.int64)
    pred[top["pidx"].to_numpy()] = top["arc"].to_numpy()
    return pred


def build_route(net, point_arcs: np.ndarray, fill_limit: float = 3000.0, filter_cycles: bool = True) -> list[int]:
    """Turn per-point arcs into a continuous, topologically connected road arc sequence:
    1. Filter unmatched observations (-1).
    2. Suppress single-step ping-pong flickers (e.g. A -> B -> A) caused by GPS jitter.
    3. Deduplicate consecutive identical arcs.
    4. Fill topological gaps between non-adjacent arcs with shortest path.
    5. Eliminate immediate turnaround sub-cycles (U-turns within 2 steps) to ensure physical continuity.
    """
    raw_seq = [int(a) for a in point_arcs if a >= 0]
    if not raw_seq:
        return []

    # Step 1: Suppress 1-step transient ping-pong flickers
    if filter_cycles and len(raw_seq) >= 3:
        cleaned = list(raw_seq)
        for i in range(1, len(cleaned) - 1):
            if cleaned[i - 1] == cleaned[i + 1] and cleaned[i] != cleaned[i - 1]:
                cleaned[i] = cleaned[i - 1]
        raw_seq = cleaned

    # Step 2: Deduplicate consecutive identical arcs
    seq = [a for k, a in enumerate(raw_seq) if k == 0 or a != raw_seq[k - 1]]

    # Step 3: Connect consecutive arcs along road network
    route = []
    for a in seq:
        if route and route[-1] != a and not net.connected(route[-1], a):
            fill = net.shortest_arc_path(route[-1], a, limit=fill_limit)
            route.extend(fill)
        if not route or route[-1] != a:
            route.append(a)

    # Step 4: Prune immediate turnaround cycles (e.g. [..., A, B, A, ...])
    if filter_cycles and len(route) >= 3:
        pruned = []
        for arc in route:
            if len(pruned) >= 2 and pruned[-2] == arc:
                pruned.pop()
            else:
                pruned.append(arc)
        route = pruned

    return route


def route_metrics(net, pred_route, ref_route) -> dict:
    """Length-weighted overlap between predicted and reference arc sets, and the
    route mismatch fraction (Newson & Krumm): (erroneously added + missed) / ref."""
    P, R = set(map(int, pred_route)), set(map(int, ref_route))
    L = net.length
    inter = sum(L[a] for a in P & R)
    lp, lr = sum(L[a] for a in P), sum(L[a] for a in R)
    prec = inter / lp if lp else 0.0
    rec = inter / lr if lr else 0.0
    return {
        "route_precision": prec,
        "route_recall": rec,
        "route_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "route_mismatch_fraction": ((lp - inter) + (lr - inter)) / lr if lr else np.nan,
        "pred_route_km": lp / 1000,
        "ref_route_km": lr / 1000,
    }


def point_continuity(net, point_arcs) -> float:
    """Fraction of consecutive matched point pairs that are adjacent on the road network."""
    a = point_arcs[point_arcs >= 0]
    if len(a) < 2:
        return 1.0
    ok = (a[1:] == a[:-1]) | (net.src[a[1:]] == net.dst[a[:-1]])
    return float(ok.mean())
