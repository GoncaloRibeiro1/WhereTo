"""
Web server for the WhereTo map app.

    python server.py --graph porto.graphml --landmarks porto_landmarks.geojson

Then open http://127.0.0.1:8000/docs to try the API by hand (the map page
will be served at http://127.0.0.1:8000/).

The app talks to three endpoints, all with coordinates as [lat, lon]:

  POST /api/snap      a click on the map -> the road under it, in each
                      direction it can be driven (where to put the arrow)
  POST /api/next      a route -> the choices at its last junction: one per
                      road the car may take, ending at the next junction
                      where the driver has a choice again (pulsing circles)
  POST /api/predict   a route + settings -> heatmap squares, top places

A route is a list of edges [u, v, key] (see whereto.route_nodes):
the road the arrow was placed on, then the edges of each option clicked.
"""

import argparse
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyproj
import shapely
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from shapely.ops import substring

import whereto as wt

SNAP_MAX_M = 80      # a click further than this from any road is ignored
CHAIN_MAX = 100      # safety limit when following roads without a choice
TOP_PLACES = 20

Edge = tuple[int, int, int]


# ----------------------------------------------------------------------------
# Request / response shapes (also documented at /docs)
# ----------------------------------------------------------------------------
class SnapRequest(BaseModel):
    lat: float
    lon: float


class RouteRequest(BaseModel):
    route: list[Edge] = Field(..., min_length=1,
                              description="edges [u, v, key], in driving order")


class Settings(BaseModel):
    beta: float = Field(wt.BETA, gt=0, description="driver rationality per second "
                        "of detour (0.06: a 60 s detour is ~36x less likely)")
    trip_scale: float | None = Field(wt.TRIP_SCALE, ge=0, description="favour "
                                     "destinations within about this many seconds "
                                     "of free-flow driving (0 or null = off)")
    landmarks: bool = Field(True, description="landmark prior (if the server "
                            "has one) instead of a uniform one")
    time: str | None = Field(None, description="'HH:MM' or 'now'; weights "
                             "landmarks by how busy they are then")
    day: str | None = Field(None, description="'weekday' or monday..sunday")
    landmark_floor: float = Field(0.2, ge=0, le=1)
    landmark_sigma: float = Field(100, gt=0)
    cell: float = Field(300, ge=50, le=5000, description="square size, metres")


class PredictRequest(RouteRequest):
    settings: Settings = Settings()


# ----------------------------------------------------------------------------
# The loaded city
# ----------------------------------------------------------------------------
class City:
    """The road graph, the model and the lookup structures, loaded once."""

    def __init__(self, graph_path, landmarks_path=None):
        print(f"Loading {graph_path} ...")
        self.G = G = wt.load_graph_file(graph_path)
        self.predictor = wt.Predictor(G)
        self.landmarks_path = landmarks_path
        self._landmarks = OrderedDict()        # settings -> LandmarkPrior
        self.lock = threading.Lock()           # Predictor's cache isn't thread-safe

        crs = G.graph["crs"]
        self._to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        self._to_xy = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        # Every directed edge as a line in metres, in a spatial index for snapping
        self.edges = list(G.edges(keys=True))
        self.lines = [shapely.LineString(wt._edge_coords(G, u, v, G.edges[u, v, k]))
                      for u, v, k in self.edges]
        self.tree = shapely.STRtree(self.lines)
        self.edge_index = {e: i for i, e in enumerate(self.edges)}

        lat = [d["lat"] for _, d in G.nodes(data=True)]
        lon = [d["lon"] for _, d in G.nodes(data=True)]
        self.bounds = [[min(lat), min(lon)], [max(lat), max(lon)]]
        print(f"Ready: {G.number_of_nodes()} junctions, {len(self.edges)} roads"
              + (f", landmarks from {landmarks_path}" if landmarks_path else ""))

    # -- coordinates ---------------------------------------------------------
    def latlon(self, coords):
        """Projected (x, y) pairs -> [[lat, lon], ...]."""
        xy = np.asarray(coords, float).reshape(-1, 2)
        lon, lat = self._to_ll.transform(xy[:, 0], xy[:, 1])
        return [[round(a, 6), round(o, 6)] for a, o in zip(lat, lon)]

    def node_latlon(self, n):
        d = self.G.nodes[n]
        return [round(d["lat"], 6), round(d["lon"], 6)]

    def edge_line(self, e):
        return self.lines[self.edge_index[e]]

    # -- landmarks -----------------------------------------------------------
    def landmarks(self, s: Settings):
        """The LandmarkPrior for these settings (cached), or None."""
        if not (s.landmarks and self.landmarks_path):
            return None
        when = wt.parse_when(s.time, s.day)
        key = (when, s.landmark_floor, s.landmark_sigma)
        lm = self._landmarks.get(key)
        if lm is None:
            lm = wt.LandmarkPrior(self.G, self.landmarks_path, s.landmark_sigma,
                                  s.landmark_floor, when)
            self._landmarks[key] = lm
            if len(self._landmarks) > 8:
                self._landmarks.popitem(last=False)
        return lm


def _edge_bearing_at(line, s):
    """Compass heading along a projected line at distance s from its start."""
    a = line.interpolate(max(0.0, min(s, line.length - 1.0)))
    b = line.interpolate(min(line.length, max(s, 0.0) + 1.0))
    if a.equals(b):                               # line shorter than 1 m
        (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    else:
        x0, y0, x1, y1 = a.x, a.y, b.x, b.y
    return round(wt._bearing(x0, y0, x1, y1), 1)


def _all_exits(exits):
    return [e for lst in exits.values() for e in lst]


def junction_options(G, edge):
    """
    The choices of a car arriving at a junction along `edge` = (u, X, key).
    Each choice is a road leaving X, followed through any junctions where
    there is only one way to go, up to the next real choice (or dead end).
    Returns (kind, [(turn, Exit, [edges])]).
    """
    u, X, k = edge
    in_bearing = wt.arrival_bearing(G, u, X, G.edges[u, X, k])
    exits, kind = wt.exits_by_turn(G, X, u, in_bearing)
    options = []
    for cls in ("left", "straight", "right"):
        for e in sorted(exits.get(cls, []), key=lambda e: e.delta):
            chain, seen = [(X, e.v, e.key)], {X}
            while len(chain) < CHAIN_MAX:
                a, b, kk = chain[-1]
                if b in seen:
                    break                         # went round a loop
                seen.add(b)
                nxt = _all_exits(wt.exits_by_turn(
                    G, b, a, wt.arrival_bearing(G, a, b, G.edges[a, b, kk]))[0])
                if len(nxt) != 1:
                    break                         # a real choice, or a dead end
                chain.append((b, nxt[0].v, nxt[0].key))
            options.append((cls, e, chain))
    return kind, options


# ----------------------------------------------------------------------------
# The web app
# ----------------------------------------------------------------------------
def create_app(city: City) -> FastAPI:
    app = FastAPI(title="WhereTo",
                  description="Where is the car going? See server.py.")
    G = city.G

    def check_route(route):
        """Validate a route, turning model errors into HTTP 400."""
        route = [tuple(e) for e in route]
        for e in route:
            if not G.has_edge(*e):
                raise HTTPException(400, f"Unknown road {list(e)}")
        try:
            wt.crossings_along(G, route)
        except ValueError as err:
            raise HTTPException(400, str(err))
        return route

    @app.get("/api/info")
    def info():
        """Map extent and what this server can do."""
        return {"bounds": city.bounds,
                "landmarks": city.landmarks_path is not None,
                "defaults": Settings().model_dump(),
                "days": ["weekday"] + wt.DAYS}

    @app.post("/api/snap")
    def snap(req: SnapRequest):
        """The road under a click, once per direction it can be driven.
        The first option is the direction of the edge OSM stores first; the
        app can offer a 'flip' to the second."""
        (s, w), (n, e) = city.bounds
        if not (s <= req.lat <= n and w <= req.lon <= e):
            raise HTTPException(404, f"({req.lat}, {req.lon}) is outside the map "
                                f"(lat {s:.3f}..{n:.3f}, lon {w:.3f}..{e:.3f}); "
                                f"west longitudes are negative")
        x, y = city._to_xy.transform(req.lon, req.lat)
        pt = shapely.Point(x, y)
        near = city.tree.query_nearest(pt, max_distance=SNAP_MAX_M)
        if len(near) == 0:
            raise HTTPException(404, f"No road within {SNAP_MAX_M} m")
        u0, v0, _ = city.edges[near[0]]
        dist = pt.distance(city.lines[near[0]])
        # The same street in both directions (and parallel roads between the
        # same two junctions) are separate edges with the same line.
        same = [i for i in city.tree.query(pt, predicate="dwithin", distance=dist + 1)
                if {city.edges[i][0], city.edges[i][1]} == {u0, v0}]
        options = []
        for i in sorted(same, key=lambda i: (city.edges[i][0] != u0, city.edges[i])):
            u, v, k = e = city.edges[i]
            line = city.lines[i]
            s = line.project(pt)
            p = line.interpolate(s)
            ahead = substring(line, s, line.length)
            options.append({
                "edge": list(e),
                "name": wt.road_name(G.edges[e]),
                "position": city.latlon([(p.x, p.y)])[0],
                "bearing": _edge_bearing_at(line, s),
                "ahead": city.latlon(ahead.coords if ahead.length > 0
                                     else [(p.x, p.y), line.coords[-1]]),
                "junction": city.node_latlon(v),
            })
        return {"distance_m": round(dist, 1), "options": options}

    @app.post("/api/next")
    def next_choices(req: RouteRequest):
        """Where the car can go from the end of the route."""
        route = check_route(req.route)
        last = route[-1]
        kind, options = junction_options(G, last)
        out = []
        for cls, e, chain in options:
            coords = []
            for ce in chain:
                pts = city.latlon(city.edge_line(ce).coords)
                coords.extend(pts if not coords else pts[1:])
            a, b, kk = chain[-1]
            out.append({
                "turn": cls,
                "delta": round(e.delta, 1),
                "name": wt.road_name(e.data),
                "edges": [list(ce) for ce in chain],
                "coords": coords,
                "target": city.node_latlon(b),
                "arrival_bearing": round(wt.arrival_bearing(G, a, b,
                                                            G.edges[a, b, kk]), 1),
            })
        return {"junction": {"node": last[1], "position": city.node_latlon(last[1]),
                             "kind": kind},
                "dead_end": not out,
                "options": out}

    @app.post("/api/predict")
    def predict(req: PredictRequest):
        """Probability of every destination given the route so far."""
        route = check_route(req.route)
        s = req.settings
        try:
            with city.lock:
                landmarks = city.landmarks(s)
                res = city.predictor.predict(route, beta=s.beta,
                                             trip_scale=s.trip_scale,
                                             landmarks=landmarks, observed="exit",
                                             cell_m=s.cell)
        except ValueError as err:
            raise HTTPException(400, str(err))

        # Squares: projected corners -> lat/lon (slightly rotated quads)
        cells = sorted(res.cells.items(), key=lambda kv: -kv[1]["p"])
        corners = [((cx + dx) * s.cell, (cy + dy) * s.cell)
                   for (cx, cy), _ in cells
                   for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))]
        corners = city.latlon(corners) if corners else []
        cells_out = [{"p": float(c["p"]), "n": c["n"],
                      "corners": corners[4 * i: 4 * i + 4]}
                     for i, (_, c) in enumerate(cells)]

        places = []
        if landmarks is not None:
            top = res.places[:TOP_PLACES]
            lonlat = wt._to_lonlat(landmarks, [i for i, _ in top])
            places = [{"p": p, "name": landmarks.names[i],
                       "category": landmarks.categories[i],
                       "position": [round(lat, 6), round(lon, 6)]}
                      for (i, p), (lon, lat) in zip(top, lonlat)]

        # Plain JSON types already: skip FastAPI's slow generic encoder
        # (~50 ms on the few thousand squares).
        return JSONResponse({
            "crossings": [{"node": c.node, "position": city.node_latlon(c.node),
                           "kind": c.kind, "turn": c.turn,
                           "onto": wt.road_name(c.taken.data),
                           "options": len(_all_exits(c.exits))}
                          for c in res.crossings],
            "cells": cells_out,
            "places": places,
            "elsewhere": res.elsewhere,
        })

    web = Path(__file__).with_name("web")
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


def main():
    ap = argparse.ArgumentParser(description="Run the WhereTo map app.")
    ap.add_argument("--graph", default="porto.graphml",
                    help="road network from whereto.py --build-graph")
    ap.add_argument("--landmarks", default="porto_landmarks.geojson",
                    help="landmarks from --build-landmarks ('' = none)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.landmarks and not Path(args.landmarks).exists():
        print(f"{args.landmarks} not found; running without landmarks")
        args.landmarks = None

    import uvicorn
    app = create_app(City(args.graph, args.landmarks or None))
    print(f"Open http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
