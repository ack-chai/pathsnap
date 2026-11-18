"""Parsers for .track / .nodes / .arcs / .route files."""
import numpy as np
from pathlib import Path

def load_track(path):
    return np.loadtxt(path)

def load_nodes(path):
    return np.loadtxt(path)

def load_arcs(path):
    return np.loadtxt(path, dtype=int)

def load_route(path):
    return np.loadtxt(path, dtype=int)

def load_trip(trip_dir):
    d = Path(trip_dir)
    trip = {"track": load_track(d / "gps.track"), "nodes": load_nodes(d / "road_network.nodes"),
            "arcs": load_arcs(d / "road_network.arcs")}
    r = d / "route.route"
    trip["route"] = load_route(r) if r.exists() else None
    return trip
