"""Candidate generation: for each GPS obs, find nearby arcs."""
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

def build_arc_index(nodes_xy, arcs):
    geoms, arc_ids = [], []
    for aid, (u, v) in enumerate(arcs):
        p, q = nodes_xy[u], nodes_xy[v]
        geoms.append(LineString([p, q]))
        arc_ids.append(aid)
    return geoms, np.array(arc_ids)

def generate_candidates(obs_xy, geoms, arc_ids, radius_m=75, max_cands=12):
    tree = STRtree(geoms)
    rows = []
    for oi, p in enumerate(obs_xy):
        pt = Point(p)
        idx = tree.query(pt.buffer(radius_m))
        cands = []
        for i in idx:
            g = geoms[i]
            d = pt.distance(g)
            frac = g.project(pt) / max(g.length, 1e-9)
            cands.append((int(arc_ids[i]), float(d), float(frac)))
        cands.sort(key=lambda c: c[1])
        rows.append(cands[:max_cands])
    return rows
