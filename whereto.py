"""
Destination probabilities from a sequence of observed turns.

Given a crossing, the direction the car arrived from, and the turns it made
at this and the following crossings (left / right / straight), this script
computes, for every place on the road network, the probability that it is
the car's destination, then groups those places into square regions and
draws a heatmap.

Model (Bayes' rule):

    P(dest = d | turns)  ∝  P(turn_1 | d) · P(turn_2 | d) · ...  ·  P(dest = d)
        posterior                  likelihood                       prior

  prior       P(d)        how likely d is as a destination before seeing any
                          turn. Default: every intersection equally likely.
                          With --landmarks: most of it on intersections near
                          hospitals, universities, malls, stations, ... (see
                          LANDMARKS for the weights), plus a uniform floor.
                          Times a preference for nearby destinations
                          (--trip-scale, default TRIP_SCALE = 600 s).
  likelihood  P(turn | d) if the driver were going to d, how likely is this
                          turn at this crossing? A "noisy rational driver":
                          for each exit road compute the fastest time to d if
                          the car takes that exit, then
                              P(exit | d) = exp(-beta*T_exit) / sum exp(-beta*T)
                          The fastest exit gets the most probability; an exit
                          that costs 60 s extra is exp(-beta*60) times as likely.
                          Each turn is scored at the crossing where it happened,
                          and the scores are multiplied together.
  posterior   P(d | turns) multiply everything, then divide by the total so all
                          the probabilities add up to 1.

Turn sequence
-------------
--turn takes one word per crossing, in order. The car makes the first turn
at the starting crossing, drives to the next crossing, makes the second turn
there, and so on. If the car drove through a crossing without turning, say
"straight" for it:

    --turn right left              right here, left at the next crossing
    --turn right straight left     right, straight through one, then left

At a fork (the road splits in two, both branches roughly straight ahead) say
"left" for keep left and "right" for keep right.

Usage
-----
Real map (needs internet access to OpenStreetMap):
    python whereto.py --lat 38.7223 --lon -9.1393 --heading 0 \
        --turn right left --radius 3000

  --lat/--lon   point at or near the first crossing (nearest intersection)
  --heading     compass direction the car was driving when it arrived at the
                first crossing (0 = north, 90 = east, 180 = south, 270 = west)
  --turn        one or more of: left | right | straight

Cached city graph (download once, then reuse offline):
    python whereto.py --build-graph porto.graphml
    python whereto.py --graph porto.graphml --lat 41.1622 \
        --lon -8.5989 --heading 0 --turn right right

  --build-graph downloads the roads of --places (default: Porto, Matosinhos,
  Maia) and saves them; --graph loads that file instead of downloading, and
  --radius is then ignored.

Landmark prior (download once, then reuse):
    python whereto.py --build-landmarks porto_landmarks.geojson
    python whereto.py --graph porto.graphml \
        --landmarks porto_landmarks.geojson --lat 41.1622 --lon -8.5989 \
        --heading 0 --turn right right

  Also prints the most likely named places and writes them to *_places.csv.
  --landmark-floor (default 0.2) is the share of the prior kept uniform;
  --landmark-sigma (default 100 m) is how far a landmark reaches.

  --time HH:MM (or now) and --day (weekday, monday ... sunday) weight each
  kind of landmark by how busy it is then (PROFILES): universities, offices
  and schools in the morning, malls and homes in the evening, few offices
  at the weekend.

Offline demo (synthetic city grid, no internet needed):
    python whereto.py --demo --turn right left

Outputs: a PNG heatmap and a CSV of regions ranked by probability (and, with
landmarks, a CSV of places ranked by probability).

Requires: pip install osmnx networkx matplotlib numpy
"""

import argparse
import ast
import csv
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Turn classes by the change in heading (degrees, + = clockwise = to the right)
STRAIGHT_MAX = 30     # |change| < 30°           -> straight
TURN_MAX = 150        # 30° <= |change| <= 150°  -> left / right
                      # beyond 150°              -> U-turn (not allowed)
STRAIGHT_LOOSE = 60   # at a 4-way crossroads the middle road is 'straight'
                      # if within 60° (see crossroads_classes)

# Ideal heading change for each class, used to pick one road when a class
# has several (e.g. two roads that both count as "right").
IDEAL_DELTA = {"straight": 0, "right": 90, "left": -90}

# How strongly a turn counts against destinations it leads away from:
# P(exit | d) ∝ exp(-BETA * T_exit(d)). 0.06 per second: a route 60 s slower
# is ~36x less likely. With 0.03 (6x) a car could drive several minutes away
# from a big attractor (NorteShopping) and it stayed the top place, because
# each junction only charges part of the detour. Much higher would punish the
# free-flow travel times' own errors (traffic lights, congestion: easily 30 s).
BETA = 0.06

# Most car trips are short, so the prior favours nearby destinations:
# P(d) is multiplied by exp(-T / TRIP_SCALE), T = fastest (free-flow) driving
# time from the first junction to d. 600 s of free-flow driving (~15 min in
# real traffic): a place 10 min away counts 37% as much as one next door,
# 20 min away 14%. --trip-scale changes it, 0 turns it off.
TRIP_SCALE = 600


# ----------------------------------------------------------------------------
# Loading the road network
# ----------------------------------------------------------------------------
# Municipalities loaded by --build-graph by default: Porto plus its neighbours,
# since many trips cross the municipal border.
PORTO_PLACES = [
    "Porto, Portugal",
    "Matosinhos, Portugal",
    "Maia, Portugal",
]


def _prepare_osm_graph(G):
    """Add speeds and travel times, remember each node's lon/lat, and project
    to metres. (osmnx's project_graph overwrites x/y and does not keep the
    original coordinates, so they are copied to 'lon'/'lat' first.)"""
    import osmnx as ox
    G = ox.add_edge_speeds(G)          # fills missing speed limits by road type
    G = ox.add_edge_travel_times(G)    # travel_time = length / speed
    for _, nd in G.nodes(data=True):
        nd["lon"], nd["lat"] = nd["x"], nd["y"]
    return ox.project_graph(G)         # UTM metres in x/y


def nearest_node(G, lat, lon):
    """The node closest to (lat, lon). Uses the 'lon'/'lat' node attributes
    (local flat-earth approximation, fine at city scale)."""
    nodes = list(G.nodes)
    la = np.array([G.nodes[n]["lat"] for n in nodes])
    lo = np.array([G.nodes[n]["lon"] for n in nodes])
    dx = (lo - lon) * math.cos(math.radians(lat))
    dy = la - lat
    i = int(np.argmin(dx * dx + dy * dy))
    dist_m = math.hypot(dx[i], dy[i]) * 111_320
    if dist_m > 200:
        print(f"Warning: nearest intersection is {dist_m:.0f} m from "
              f"({lat}, {lon}); is the point inside the map?")
    return nodes[i]


def load_osm_graph(lat, lon, radius_m):
    """Download drivable roads around (lat, lon). Returns the graph projected
    to metres (x, y), with 'travel_time' (seconds) on every edge."""
    import osmnx as ox
    G = ox.graph_from_point((lat, lon), dist=radius_m, network_type="drive")
    G = _prepare_osm_graph(G)
    return G, nearest_node(G, lat, lon)


def build_place_graph(places, path):
    """Download drivable roads for the named places (one merged graph),
    prepare it like load_osm_graph and save it as GraphML for --graph."""
    import osmnx as ox
    print(f"Downloading drivable roads for: {'; '.join(places)} ...")
    G = ox.graph_from_place(places, network_type="drive")
    G = _prepare_osm_graph(G)
    ox.save_graphml(G, path)
    print(f"Saved {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"to {path}")
    return G


def load_graph_file(path):
    """Load a graph saved by build_place_graph (already projected, with
    travel_time on edges and lon/lat on nodes)."""
    import osmnx as ox
    G = ox.load_graphml(path, node_dtypes={"lon": float, "lat": float})
    missing = [a for a in ("lon", "lat") if a not in next(iter(G.nodes.values()))]
    if missing or not all("travel_time" in d for _, _, d in G.edges(data=True)):
        raise ValueError(f"{path} was not made by --build-graph "
                         f"(needs lon/lat on nodes and travel_time on edges)")
    return G


def make_demo_graph(n=20, spacing=150.0):
    """A synthetic grid city: 2-way streets at 40 km/h, plus one fast avenue
    (row 14) and one fast boulevard (column 6) at 70 km/h. Origin of the car
    is the centre intersection."""
    G = nx.MultiDiGraph()
    for i in range(n):
        for j in range(n):
            G.add_node((i, j), x=j * spacing, y=i * spacing)
    for i in range(n):
        for j in range(n):
            for di, dj in ((0, 1), (1, 0)):
                a, b = (i, j), (i + di, j + dj)
                if b[0] >= n or b[1] >= n:
                    continue
                fast = (di == 0 and i == 14) or (dj == 0 and j == 6)
                speed = 70 if fast else 40                 # km/h
                tt = spacing / (speed / 3.6)
                G.add_edge(a, b, length=spacing, travel_time=tt)
                G.add_edge(b, a, length=spacing, travel_time=tt)
    G.graph["demo"] = True
    return G, (n // 2, n // 2)


# ----------------------------------------------------------------------------
# Landmarks: the destination prior P(d)
# ----------------------------------------------------------------------------
# OSM tag -> (weight, typical area in m²). The weight is roughly how many car
# trips a day end at one typical place of that kind, in hundreds: a mall of
# ~20k m² in OSM ~6k car arrivals a day (NorteShopping, 47k m²: ~35k visits
# a day, ~70% by car, ~1.8 people per car -> ~14k); a hospital ~3k (staff
# shifts, appointments, visits; a central one like São João ~7k); a
# university faculty ~2k (many students come by metro or bus).
# Rough estimates, not measured data: tune freely.
# A polygon bigger or smaller than typical has its weight scaled by its size
# (SIZE_SCALING); typical area None = no size scaling.
# `True` matches any value of that key (every office=*).
LANDMARKS = {
    ("aeroway", "terminal"):        (100, 40_000),
    ("shop", "mall"):               (60, 20_000),
    ("amenity", "hospital"):        (30, 30_000),
    ("amenity", "university"):      (20, 20_000),
    ("leisure", "stadium"):         (20, None),
    ("railway", "station"):         (10, None),
    ("amenity", "college"):         (10, 5_000),
    ("shop", "department_store"):   (10, None),
    ("amenity", "marketplace"):     (8, None),
    ("shop", "supermarket"):        (8, 1_500),
    ("amenity", "school"):          (6, 5_000),
    ("office", True):               (2, None),
    ("amenity", "parking"):         (1, 1_000),
}
# Size factor = (area / typical area) ** exponent, clipped to [low, high];
# `unknown` for a place mapped as a point (no area): big places are nearly
# always mapped with their outline, small ones often as a point.
#   category -> (exponent, low, high, unknown)
SIZE_SCALING = {
    # sqrt: OSM outlines include grounds and car parks (a hospital's gardens),
    # so area overstates traffic
    "default":   (0.5, 0.25, 3, 0.5),
    # shop floor, and so visits, grows with the building; the many small
    # galleries in old Porto buildings get few car trips
    "shop=mall": (1.0, 0.05, 3, 0.1),
}


def _category(key, value):
    return key if value is True else f"{key}={value}"


CATEGORY_WEIGHTS = {_category(k, v): wt for (k, v), wt in LANDMARKS.items()}

# When do car trips arrive at each kind of place? (--time / --day)
# name -> (24 values, one per hour 0h..23h on a weekday, relative shape only:
#          they are rescaled to average 1 over the day;
#          Saturday and Sunday daily volume relative to a weekday)
# Hand-made shapes from common sense, not measured data: tune freely.
PROFILES = {
    #             0    1    2    3    4    5    6    7    8    9   10   11
    #            12   13   14   15   16   17   18   19   20   21   22   23
    "work":     ([.05, .05, .05, .05, .05, .05, .4,  2.5, 5,   4,   1.5, .8,
                  .6,  1.8, 1.5, .6,  .4,  .3,  .2,  .1,  .05, .05, .05, .05],
                 0.15, 0.05),
    "study":    ([.02, .02, .02, .02, .02, .02, .1,  1.5, 4,   3.5, 2.5, 2,
                  1,   2,   2.5, 1.2, .8,  .6,  .5,  .3,  .1,  .02, .02, .02],
                 0.1, 0.03),
    # drop-off in the morning, pick-up in the afternoon
    "school":   ([.01, .01, .01, .01, .01, .01, .05, 2,   8,   1,   .3,  .3,
                  .5,  1,   .5,  .5,  1.5, 2,   1,   .2,  .05, .01, .01, .01],
                 0.02, 0.01),
    # staff shifts and appointments in the morning, visits later, ER at night
    "hospital": ([.3,  .3,  .3,  .3,  .3,  .3,  1,   3,   3.5, 3,   2.2, 1.6,
                  1,   1.2, 1.8, 1.5, 1.2, 1,   1,   1,   .8,  .4,  .4,  .4],
                 0.5, 0.4),
    "shopping": ([.02, .02, .02, .02, .02, .02, .02, .02, .02, .3,  .8,  1.2,
                  1.6, 1.5, 1.3, 1.4, 1.6, 2,   2.2, 2.2, 1.8, 1,   .4,  .05],
                 1.4, 1.1),
    "groceries": ([.02, .02, .02, .02, .02, .02, .02, .3,  .8,  1.2, 1.4, 1.5,
                   1.4, 1.1, 1,   1.1, 1.4, 1.9, 2.2, 2,   1.4, .6,  .1,  .02],
                  1.3, 0.8),
    "market":   ([.02, .02, .02, .02, .02, .02, .5,  1.5, 2.5, 3,   3,   2.5,
                  1.5, .8,  .5,  .5,  .5,  .5,  .3,  .1,  .02, .02, .02, .02],
                 1.2, 0.2),
    # drop-offs and park & ride: mostly the morning commute
    "station":  ([.05, .05, .05, .05, .05, .3,  1,   3,   3.5, 1.5, .8,  .8,
                  .8,  .8,  .8,  .8,  1.2, 2,   2,   1.2, .6,  .3,  .3,  .3],
                 0.6, 0.5),
    "airport":  ([.2,  .2,  .2,  .2,  1,   2,   2,   1.2, 1.2, 1.2, 1.2, 1.2,
                  1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1,   .8,  .5],
                 1.0, 1.0),
    "events":   ([.05, .05, .05, .05, .05, .05, .05, .05, .05, .5,  .5,  .5,
                  .5,  .5,  .5,  .5,  .5,  1,   2,   3,   3,   1.5, .5,  .1],
                 2.5, 2.5),
    # car parks serve everything: a blend of commute and shopping
    "general":  ([.1,  .1,  .1,  .1,  .1,  .1,  .5,  1.5, 2.2, 1.8, 1.2, 1.1,
                  1.1, 1.3, 1.3, 1.1, 1.2, 1.4, 1.5, 1.3, .9,  .6,  .4,  .2],
                 0.8, 0.5),
    # homes: people coming back for lunch and in the evening
    "home":     ([.6,  .3,  .15, .15, .15, .15, .2,  .3,  .4,  .4,  .5,  .6,
                  1.2, 1,   .7,  .8,  1.2, 1.8, 2.5, 2.8, 2.2, 1.5, 1.2, .9],
                 1.0, 1.1),
}
CATEGORY_PROFILE = {
    "aeroway=terminal": "airport", "amenity=hospital": "hospital",
    "amenity=university": "study", "amenity=college": "study",
    "shop=mall": "shopping", "shop=department_store": "shopping",
    "leisure=stadium": "events", "railway=station": "station",
    "amenity=marketplace": "market", "shop=supermarket": "groceries",
    "amenity=school": "school", "office": "work", "amenity=parking": "general",
}
# The uniform floor stands for everything that is not a landmark: homes, but
# also most workplaces and small shops (only a fraction are in OSM), so it
# follows a blend of the home and general-traffic profiles.
FLOOR_PROFILE = {"home": 0.5, "general": 0.5}
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday"]


def activity(profile, day, hour):
    """How busy `profile` is at `hour` (0-24, fractional) on `day` (a name in
    DAYS, or None = any day), relative to an average hour of an average
    weekday. hour=None gives the whole-day factor only."""
    values, sat, sun = PROFILES[profile]
    day_factor = {"saturday": sat, "sunday": sun}.get(day, 1.0)
    if hour is None:
        return day_factor
    v = np.asarray(values, float)
    v /= v.mean()
    # Each value is for the hour starting at h, i.e. centred on h + 0.5;
    # interpolate between centres, wrapping around midnight.
    centres = np.arange(-0.5, 25.5)
    return day_factor * float(np.interp(hour, centres,
                                        np.concatenate([v[-1:], v, v[:1]])))


def parse_when(time_str, day):
    """--time / --day -> (day name or None, hour as float or None)."""
    if time_str == "now":
        import datetime
        now = datetime.datetime.now()
        return day or DAYS[now.weekday()], now.hour + now.minute / 60
    hour = None
    if time_str:
        h, _, m = time_str.partition(":")
        hour = int(h) + int(m or 0) / 60
        if not 0 <= hour < 24:
            raise ValueError(f"--time {time_str}: expected HH:MM")
    return day, hour


def build_landmarks(places, path):
    """Download the LANDMARKS of `places` from OSM and save them (name,
    category, area, outline in lon/lat) as GeoJSON for --landmarks."""
    import geopandas as gpd
    import osmnx as ox
    print(f"Downloading landmarks for: {'; '.join(places)} ...")
    # The exact municipal outline makes the Overpass query too large (HTTP 413),
    # so query its convex hull; landmarks far from any road are dropped later.
    area = ox.geocode_to_gdf(places).union_all().convex_hull
    tags = defaultdict(list)
    for key, value in LANDMARKS:
        if value is True:
            tags[key] = True
        else:
            tags[key].append(value)
    out = landmarks_table(ox.features_from_polygon(area, dict(tags)))
    out.to_file(path, driver="GeoJSON")
    print(f"Saved {len(out)} landmarks to {path}:")
    for c, n in out["category"].value_counts().items():
        print(f"    {n:5d}  {c}")
    return out


def landmarks_table(gdf):
    """OSM features (from osmnx) -> GeoDataFrame of name, category, area_m2."""
    import geopandas as gpd
    gdf = gdf[gdf.geometry.geom_type.isin(["Point", "Polygon", "MultiPolygon"])]

    def best_category(row):
        """A feature can carry several tags; keep the heaviest category."""
        cats = [_category(k, v) for k, v in LANDMARKS
                if k in row and isinstance(row[k], str)
                and (v is True or row[k] == v)]
        return max(cats, key=lambda c: CATEGORY_WEIGHTS[c][0]) if cats else None

    cat = gdf.apply(best_category, axis=1)
    gdf = gdf[cat.notna()]
    out = gpd.GeoDataFrame(
        {"name": gdf["name"].fillna("").values if "name" in gdf else "",
         "category": cat[cat.notna()].values,
         "area_m2": gdf.geometry.to_crs(gdf.estimate_utm_crs()).area.round().values},
        geometry=gdf.geometry.values, crs=gdf.crs)
    return _drop_nested(out)


def _drop_nested(lm):
    """
    Drop a polygon that is mostly (> 50%) covered by other polygons of the
    same category, so a place is not counted twice: a university building
    inside a university campus, or an umbrella like "Universidade do Porto",
    one multipolygon made of all the faculties that are also mapped one by one.

    Largest first, counting only polygons not dropped yet as cover: the
    umbrella is dropped (its faculties cover it) and the faculties then stay;
    a campus stays and the building inside it is dropped; of two identical
    outlines one stays.
    """
    import shapely
    utm = lm.geometry.to_crs(lm.estimate_utm_crs()).values
    drop = np.zeros(len(lm), bool)
    for cat in lm["category"].unique():
        idx = np.flatnonzero((lm["category"] == cat).values
                             & (lm["area_m2"] > 0).values)
        if len(idx) < 2:
            continue
        geoms = utm[idx]
        tree = shapely.STRtree(geoms)
        for i in np.argsort(-shapely.area(geoms)):
            near = [j for j in tree.query(geoms[i], predicate="intersects")
                    if j != i and not drop[idx[j]]]
            if not near:
                continue
            covered = shapely.area(shapely.intersection(
                geoms[i], shapely.union_all(geoms[near])))
            if covered > 0.5 * shapely.area(geoms[i]):
                drop[idx[i]] = True
    if drop.any():
        print(f"Dropped {drop.sum()} landmarks nested in others of the same "
              f"kind, e.g. {', '.join(sorted(set(lm['name'][drop]) - {''})[:4])}")
    return lm[~drop].reset_index(drop=True)


class LandmarkPrior:
    """
    P(d) from landmarks. Each landmark's weight is spread over the
    intersections around it with a Gaussian kernel (sigma metres, distance
    measured from the landmark's outline, so a big campus feeds every road
    around it), then mixed with a uniform `floor` so places without a
    landmark (homes, small shops) stay possible:

        P(d) = (1 - floor) * landmark_mass(d) + floor / N

    With `when` = (day, hour) (see parse_when), each landmark's weight is
    multiplied by how busy its kind of place is then (PROFILES), and the
    floor by FLOOR_PROFILE, before normalising: at 8:30 on a weekday
    universities and offices gain, homes and malls lose; at 19:00 the reverse.
    """

    def __init__(self, G, path, sigma=100.0, floor=0.2, when=(None, None)):
        import geopandas as gpd
        import shapely
        self.crs = G.graph["crs"]
        lm = gpd.read_file(path).to_crs(self.crs)
        unknown = set(lm["category"]) - set(CATEGORY_WEIGHTS)
        if unknown:
            raise ValueError(f"{path}: unknown categories {sorted(unknown)}; "
                             f"rebuild it with --build-landmarks")
        self.names = lm["name"].fillna("").tolist()
        self.categories = lm["category"].tolist()
        self.points = shapely.get_coordinates(
            lm.geometry.representative_point())          # for the map labels
        self.weights = np.array([self._weight(c, a) for c, a
                                 in zip(lm["category"], lm["area_m2"])])
        self.floor = floor

        self.nodes = list(G.nodes)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        pts = shapely.points([G.nodes[n]["x"] for n in self.nodes],
                             [G.nodes[n]["y"] for n in self.nodes])
        tree = shapely.STRtree(pts)
        geoms = lm.geometry.values
        # (landmark, node) pairs within 3 sigma, weighted by the kernel
        li, ni = tree.query(geoms, predicate="dwithin", distance=3 * sigma)
        k = np.exp(-0.5 * (shapely.distance(geoms[li], pts[ni]) / sigma) ** 2)
        # Landmarks with no intersection that close: nearest one within 500 m
        lonely = np.setdiff1d(np.arange(len(geoms)), li)
        mi, mn = tree.query_nearest(geoms[lonely], max_distance=500)
        li = np.concatenate([li, lonely[mi]])
        ni = np.concatenate([ni, mn])
        k = np.concatenate([k, np.ones(len(mi))])
        # Each landmark's weight is spread in full, however many nodes it reaches
        k_sum = np.bincount(li, weights=k, minlength=len(geoms))
        self.pair_l, self.pair_n = li, ni
        self.pair_share = k / k_sum[li]
        self.weights[k_sum == 0] = 0                     # outside the map

        # Time of day: scale each landmark, and the floor, by its activity
        day, hour = when
        factor = {p: activity(p, day, hour) for p in PROFILES}
        w_now = self.weights * np.array([factor[CATEGORY_PROFILE[c]]
                                         for c in self.categories])
        lm_share = (1 - floor) * w_now.sum() / self.weights.sum()
        floor_share = floor * sum(s * factor[p] for p, s in FLOOR_PROFILE.items())
        total = lm_share + floor_share
        self.floor_share = floor_share / total     # share of P(d) off-landmark

        self.pair_mass = lm_share / total * w_now[li] / w_now.sum() * self.pair_share
        self.prior = np.full(len(self.nodes), self.floor_share / len(self.nodes))
        np.add.at(self.prior, ni, self.pair_mass)
        self.when, self.factor = when, factor

    def describe(self):
        """A few lines summarising the prior, for printing."""
        lines = [f"Landmark prior: {int((self.weights > 0).sum())} landmarks, "
                 f"{1 - self.floor_share:.0%} of P(d) on landmarks, "
                 f"{self.floor_share:.0%} uniform"]
        day, hour = self.when
        if day is not None or hour is not None:
            when_s = day or "weekday"
            if hour is not None:
                when_s += " %02d:%02d" % divmod(round(hour * 60), 60)
            busy = sorted(self.factor.items(), key=lambda kv: -kv[1])
            lines.append(f"  {when_s}: " +
                         ", ".join(f"{p} x{f:.2f}" for p, f in busy))
        return lines

    @staticmethod
    def _weight(category, area):
        w, typical = CATEGORY_WEIGHTS[category]
        if typical:
            exponent, low, high, unknown = SIZE_SCALING.get(
                category, SIZE_SCALING["default"])
            if area and area > 0:
                w *= float(np.clip((area / typical) ** exponent, low, high))
            else:
                w *= unknown
        return w

    def label(self, i):
        name = self.names[i] or "unnamed"
        return f"{name} ({self.categories[i]})"

    def rank(self, post):
        """
        P(the car is going to landmark i | turns). Each node's posterior is
        shared among the landmarks in proportion to how much of that node's
        prior each one contributed; the uniform floor's share is returned as
        'elsewhere'. Returns ([(i, p), ...] best first, p_elsewhere).
        """
        p_node = np.zeros(len(self.nodes))
        for n, p in post.items():
            p_node[self.index[n]] = p
        ratio = p_node / self.prior                       # posterior / prior
        p_lm = np.bincount(self.pair_l, weights=self.pair_mass * ratio[self.pair_n],
                           minlength=len(self.weights))
        elsewhere = float((self.floor_share / len(self.nodes) * ratio).sum())
        order = np.argsort(-p_lm)
        return [(int(i), float(p_lm[i])) for i in order if p_lm[i] > 0], elsewhere


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def _bearing(x0, y0, x1, y1):
    """Compass bearing from point 0 to point 1 (0 = north, clockwise)."""
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360


def _angle_diff(a, b):
    """Signed difference b - a in (-180, 180]; positive = clockwise."""
    d = (b - a + 180) % 360 - 180
    return 180.0 if d == -180 else d


def _edge_coords(G, u, v, data):
    """Coordinates along an edge (uses the curved shape if OSM provides it)."""
    if "geometry" in data:
        return list(data["geometry"].coords)
    return [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]


def arrival_bearing(G, u, v, data):
    """Heading of a car as it reaches v along edge u->v (last road segment)."""
    pts = _edge_coords(G, u, v, data)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return _bearing(x0, y0, x1, y1)


def departure_bearing(G, u, v, data):
    """Heading of a car as it leaves u along edge u->v (first road segment)."""
    pts = _edge_coords(G, u, v, data)
    (x0, y0), (x1, y1) = pts[0], pts[1]
    return _bearing(x0, y0, x1, y1)


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
class Exit(NamedTuple):
    """One road leaving a junction."""
    v: object            # the node it leads to
    key: int             # which edge, when several roads join the same two nodes
    travel_time: float   # seconds along this road
    delta: float         # heading change, degrees (+ = to the right)
    data: dict           # the OSM edge attributes (name, geometry, ...)


@dataclass
class Crossing:
    """A junction on the route: what the driver could do there, and what
    they did."""
    node: object         # the junction
    came_from: object    # node the car arrived from
    in_bearing: float    # heading on arrival
    kind: str            # 'crossroads', 'fork' or 'angles' (see exits_by_turn)
    exits: dict          # class -> [Exit, ...]
    taken: Exit          # the road the car left on
    turn: str            # class of `taken`: 'left' / 'straight' / 'right'


# A route is a list of consecutive edges (u, v, key): the road the car
# arrived on, then the road it took at each junction. The junctions are the
# nodes where one edge ends and the next begins.
def route_nodes(route):
    """[first node, every junction, last node] of a route."""
    return [route[0][0]] + [v for _, v, _ in route]


def find_approach(G, X, heading):
    """Pick the incoming road whose arrival heading best matches `heading`.
    Returns (node it comes from, edge key, arrival bearing)."""
    best = None
    for u, _, k, data in G.in_edges(X, keys=True, data=True):
        b = arrival_bearing(G, u, X, data)
        err = abs(_angle_diff(heading, b))
        if best is None or err < best[0]:
            best = (err, u, k, b)
    if best is None:
        raise ValueError("No road enters this intersection.")
    err, u, k, b = best
    if err > 45:
        print(f"Warning: closest incoming road has heading {b:.0f}°, "
              f"{err:.0f}° away from the {heading:.0f}° you gave.")
    return u, k, b


def road_name(data):
    """Street name of an edge ('unnamed road' if OSM has none). Merged edges
    can carry several names, stored as a list or, after GraphML, as its repr."""
    name = data.get("name")
    if isinstance(name, str) and name.startswith("["):
        try:
            name = ast.literal_eval(name)
        except (ValueError, SyntaxError):
            pass
    if isinstance(name, (list, tuple)):
        name = " / ".join(map(str, name))
    return name or "unnamed road"


def _arms(G, X):
    """Every road touching X, including one-way roads that only lead in:
    {neighbour: bearing pointing away from X}."""
    arms = {}
    for _, v, data in G.out_edges(X, data=True):
        if v != X:
            arms.setdefault(v, departure_bearing(G, X, v, data))
    for u, _, data in G.in_edges(X, data=True):
        if u != X:
            arms.setdefault(u, (arrival_bearing(G, u, X, data) + 180) % 360)
    return arms


def crossroads_classes(G, X, came_from, in_bearing):
    """
    At a four-way crossroads, people name turns by position, not by angle:
    of the three other roads, the leftmost is 'left', the middle one is
    'straight' (across the junction), the rightmost is 'right', even when the
    crossing is skewed (e.g. the middle road is 47° off). Returns
    {neighbour: class}, or None if X is not a clear crossroads, in which case
    the angle thresholds are used instead.
    """
    arms = _arms(G, X)
    others = sorted((_angle_diff(in_bearing, b), v)
                    for v, b in arms.items() if v != came_from)
    if len(arms) != 4 or len(others) != 3:
        return None
    (dl, vl), (dm, vm), (dr, vr) = others
    if not (dl < 0 < dr and abs(dm) <= STRAIGHT_LOOSE):
        return None
    return {vl: "left", vm: "straight", vr: "right"}


def fork_classes(exits):
    """
    A fork (Y-junction): exactly two roads leave X and both are nearly
    straight ahead, e.g. Antero de Quental splitting into Vale Formoso (-9°)
    and its own one-way branch (+9°). The angle rule calls both 'straight',
    which would make the observation useless; drivers say 'keep left' /
    'keep right', so the left branch is 'left' and the right one 'right'.
    Returns the regrouped exits, or None if X is not such a fork.
    """
    if set(exits) != {"straight"} or len(exits["straight"]) != 2:
        return None
    left, right = sorted(exits["straight"], key=lambda e: e.delta)
    return defaultdict(list, left=[left], right=[right])


def exits_by_turn(G, X, came_from, in_bearing):
    """Group the roads leaving X into 'left' / 'right' / 'straight'.
    Returns (exits, kind), kind = 'crossroads', 'fork' or 'angles'."""
    by_position = crossroads_classes(G, X, came_from, in_bearing)
    exits = defaultdict(list)       # class -> [Exit, ...]
    for _, v, k, data in G.out_edges(X, keys=True, data=True):
        if v == came_from:
            continue                # going back where it came from = U-turn
        delta = _angle_diff(in_bearing, departure_bearing(G, X, v, data))
        if abs(delta) > TURN_MAX:
            continue                # sharp U-turn, not allowed
        if by_position is not None:
            cls = by_position[v]
        elif abs(delta) < STRAIGHT_MAX:
            cls = "straight"
        else:
            cls = "right" if delta > 0 else "left"
        exits[cls].append(Exit(v, k, data["travel_time"], delta, data))
    if by_position is not None:
        return exits, "crossroads"
    fork = fork_classes(exits)
    if fork is not None:
        return fork, "fork"
    return exits, "angles"


def crossings_along(G, route):
    """
    For each junction the route passes through: the roads the driver could
    take there (exits_by_turn) and the one they took. Raises ValueError if
    the edges don't join up or the route makes a move that isn't allowed
    (U-turn, wrong way up a one-way street).
    """
    crossings = []
    for (u, X, k), (X2, v, k2) in zip(route, route[1:]):
        if X2 != X:
            raise ValueError(f"Route is not connected: {u}->{X} is followed "
                             f"by {X2}->{v}")
        in_bearing = arrival_bearing(G, u, X, G.edges[u, X, k])
        exits, kind = exits_by_turn(G, X, u, in_bearing)
        taken = next(((cls, e) for cls, lst in exits.items() for e in lst
                      if (e.v, e.key) == (v, k2)), None)
        if taken is None:
            raise ValueError(f"At junction {X}, {X}->{v} is not an allowed "
                             f"exit when arriving from {u}")
        crossings.append(Crossing(X, u, in_bearing, kind, exits,
                                  taken[1], taken[0]))
    return crossings


def pick_exit(exits, turn):
    """The road the car took for `turn`. If several roads fall in that class,
    take the one closest to an ideal straight / 90° turn."""
    lst = exits[turn]
    choice = min(lst, key=lambda e: abs(e.delta - IDEAL_DELTA[turn]))
    if len(lst) > 1:
        print(f"    note: {len(lst)} roads count as '{turn}' here; following "
              f"{road_name(choice.data)} ({choice.delta:+.0f}°, closest to "
              f"{IDEAL_DELTA[turn]:+d}°)")
    return choice


def route_from_turns(G, X, heading, turns):
    """
    The command-line way to describe a route: start at junction X, arriving
    with compass `heading`, then one turn word per junction. Prints each
    junction's exits so the classification can be checked. Returns the
    route as a list of edges (see route_nodes).
    """
    came_from, key, in_bearing = find_approach(G, X, heading)
    route = [(came_from, X, key)]
    for k, turn in enumerate(turns, 1):
        exits, kind = exits_by_turn(G, X, came_from, in_bearing)
        print(f"Crossing {k} {X} ({kind}): arriving at heading "
              f"{in_bearing:.0f}° on {road_name(G.edges[came_from, X, key])}")
        for cls in ("left", "straight", "right"):
            for e in sorted(exits.get(cls, []), key=lambda e: e.delta):
                print(f"    {cls:<8} {road_name(e.data)} ({e.delta:+.0f}°)")
        print(f"  -> turned {turn}")
        if turn not in exits:
            hint = (" (this is a fork: use 'left' / 'right' for keep left / "
                    "keep right)" if kind == "fork" else "")
            raise ValueError(f"No '{turn}' exit at crossing {k} {X}; "
                             f"available: {sorted(exits)}{hint}")
        # Drive to the next junction along the road that was taken
        e = pick_exit(exits, turn)
        route.append((X, e.v, e.key))
        came_from, key, in_bearing = X, e.key, arrival_bearing(G, X, e.v, e.data)
        X = e.v
    return route


@dataclass
class Result:
    """What predict() returns."""
    route: list          # the edges (u, v, key) the car drove along
    crossings: list      # a Crossing for every junction it passed
    post: dict           # {node: P(destination = node | route)}, non-zero only
    cells: dict          # post grouped into squares (see aggregate_regions)
    places: list = None  # [(landmark index, p), ...] best first (with landmarks)
    elsewhere: float = None  # P(destination is not near any landmark)

    @property
    def nodes(self):
        return route_nodes(self.route)

    @property
    def turns(self):
        return [c.turn for c in self.crossings]


class Predictor:
    """
    The model for one road network, ready to answer many questions.

    It keeps the graph and remembers the shortest-path searches it has run
    (the slow part), so asking again about a route that grew by one junction,
    or with a different beta / prior, only searches from the new roads.
    """

    def __init__(self, G, cache_size=512):
        self.G = G
        self.nodes = list(G.nodes)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self._times = OrderedDict()            # node -> array, least recent first
        self.cache_size = cache_size

    def times_from(self, v):
        """Fastest travel time (s) from v to every node, in self.nodes order
        (inf where unreachable)."""
        t = self._times.get(v)
        if t is not None:
            self._times.move_to_end(v)
            return t
        dist = nx.single_source_dijkstra_path_length(self.G, v, weight="travel_time")
        t = np.full(len(self.nodes), np.inf)
        t[[self.index[n] for n in dist]] = list(dist.values())
        self._times[v] = t
        if len(self._times) > self.cache_size:
            self._times.popitem(last=False)
        return t

    def likelihood(self, c, beta, observed="exit"):
        """
        P(what was seen at crossing c | dest = d) for every node d.

        For every exit road e = X->v, T_e(d) = time(X->v) + fastest time v->d,
        and P(exit e | d) = softmax(-beta * T_e(d)) over all exits.
        observed='exit': the exact road taken was seen (clicking on a map);
        observed='turn': only its class, so P(turn | d) = sum of P(exit | d)
        over the exits in that class (the --turn words).
        Also returns the fastest time from X to each d.
        """
        options = [(cls, e) for cls, lst in c.exits.items() for e in lst]
        T = np.vstack([e.travel_time + self.times_from(e.v) for _, e in options])
        best = T.min(axis=0)
        reachable = np.isfinite(best)
        # Softmax over exits (shifted by the best time for numerical stability)
        w = np.zeros_like(T)
        w[:, reachable] = np.exp(-beta * (T[:, reachable] - best[reachable]))
        total = w.sum(axis=0)
        p_exit = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
        if observed == "exit":
            obs = [(e.v, e.key) == (c.taken.v, c.taken.key) for _, e in options]
        elif observed == "turn":
            obs = [cls == c.turn for cls, _ in options]
        else:
            raise ValueError(f"observed must be 'exit' or 'turn', not {observed!r}")
        return p_exit[np.array(obs)].sum(axis=0), best

    def predict(self, route, *, beta=BETA, trip_scale=TRIP_SCALE, landmarks=None,
                observed="exit", cell_m=300):
        """
        P(destination | route) for every node.

        route       edges (u, v, key): the road the car arrived on, then the
                    road it took at each junction (at least two edges)
        beta        driver rationality per second of detour
        trip_scale  favour destinations within about this many seconds of
                    driving (see TRIP_SCALE; None or 0 = off)
        landmarks   a LandmarkPrior for P(d); None = uniform
        observed    'exit' or 'turn', see likelihood()
        cell_m      size of the squares in Result.cells
        """
        crossings = crossings_along(self.G, route)
        if not crossings:
            raise ValueError("The route must pass through at least one junction.")

        likelihood = np.ones(len(self.nodes))
        first_best = None
        for c in crossings:
            lk, best = self.likelihood(c, beta, observed)
            likelihood *= lk
            if first_best is None:
                first_best = best

        # Prior P(d): uniform or from landmarks, optionally times a preference
        # for trips of about `trip_scale` seconds (from the first junction).
        reachable = np.isfinite(first_best)
        prior = np.where(reachable, 1.0, 0.0)
        if landmarks is not None:
            if landmarks.nodes != self.nodes:
                raise ValueError("landmarks were built for a different graph")
            prior *= landmarks.prior
        if trip_scale:
            prior[reachable] *= np.exp(-first_best[reachable] / trip_scale)
        # A car that drove through a junction wasn't going to that junction.
        prior[[self.index[c.node] for c in crossings]] = 0

        post = likelihood * prior
        if post.sum() == 0:
            raise ValueError("No destination is consistent with these turns.")
        post /= post.sum()
        post = {self.nodes[i]: post[i] for i in np.flatnonzero(post)}

        res = Result(route, crossings, post, aggregate_regions(self.G, post, cell_m))
        if landmarks is not None:
            res.places, res.elsewhere = landmarks.rank(post)
        return res


def aggregate_regions(G, post, cell_m):
    """Sum node probabilities into square cells of `cell_m` metres."""
    cells = defaultdict(lambda: {"p": 0.0, "n": 0, "lat": [], "lon": []})
    for d, p in post.items():
        nd = G.nodes[d]
        key = (math.floor(nd["x"] / cell_m), math.floor(nd["y"] / cell_m))
        c = cells[key]
        c["p"] += p
        c["n"] += 1
        if "lat" in nd:
            c["lat"].append(nd["lat"])
            c["lon"].append(nd["lon"])
    return cells


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_csv(cells, cell_m, path):
    rows = sorted(cells.items(), key=lambda kv: -kv[1]["p"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "probability", "cumulative", "centre_x_m",
                    "centre_y_m", "centre_lat", "centre_lon", "intersections"])
        cum = 0.0
        for rank, ((cx, cy), c) in enumerate(rows, 1):
            cum += c["p"]
            lat = np.mean(c["lat"]) if c["lat"] else ""
            lon = np.mean(c["lon"]) if c["lon"] else ""
            w.writerow([rank, f"{c['p']:.5f}", f"{cum:.4f}",
                        round((cx + .5) * cell_m), round((cy + .5) * cell_m),
                        lat if lat == "" else f"{lat:.6f}",
                        lon if lon == "" else f"{lon:.6f}", c["n"]])
    return rows


def write_places_csv(landmarks, ranked, elsewhere, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "probability", "cumulative", "name", "category",
                    "lat", "lon"])
        cum = 0.0
        lonlat = _to_lonlat(landmarks, [i for i, _ in ranked])
        for rank, ((i, p), (lon, lat)) in enumerate(zip(ranked, lonlat), 1):
            cum += p
            w.writerow([rank, f"{p:.6f}", f"{cum:.4f}", landmarks.names[i],
                        landmarks.categories[i], f"{lat:.6f}", f"{lon:.6f}"])
        w.writerow(["", f"{elsewhere:.6f}", "", "(elsewhere: no landmark)",
                    "", "", ""])


def _to_lonlat(landmarks, idx):
    """Landmark label points (projected) back to lon/lat."""
    import pyproj
    if not idx:
        return []
    tr = pyproj.Transformer.from_crs(landmarks.crs, "EPSG:4326", always_xy=True)
    xy = landmarks.points[idx]
    return list(zip(*tr.transform(xy[:, 0], xy[:, 1])))


def plot(G, path_nodes, post, cells, cell_m, turns, path, places=()):
    """`places` = [(x, y, label), ...]: top landmarks to mark on the map."""
    fig, ax = plt.subplots(figsize=(9, 9))

    # Region heatmap
    keys = list(cells)
    xs = [k[0] for k in keys]
    ys = [k[1] for k in keys]
    nx_, ny_ = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    grid = np.full((ny_, nx_), np.nan)
    for (cx, cy), c in cells.items():
        grid[cy - min(ys), cx - min(xs)] = c["p"]
    extent = [min(xs) * cell_m, (max(xs) + 1) * cell_m,
              min(ys) * cell_m, (max(ys) + 1) * cell_m]
    im = ax.imshow(grid, origin="lower", extent=extent, cmap="magma_r",
                   alpha=0.75, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("P(destination in region | turns)")

    # Roads (one LineCollection: much faster than one plot() per edge on a
    # city-sized graph)
    from matplotlib.collections import LineCollection
    segs = [_edge_coords(G, u, v, data) for u, v, data in G.edges(data=True)]
    ax.add_collection(LineCollection(segs, colors="0.55", linewidths=0.5,
                                     zorder=2))

    # The car's observed route: arrows from node to node, dots at crossings
    for a, b in zip(path_nodes, path_nodes[1:]):
        xa, ya = G.nodes[a]["x"], G.nodes[a]["y"]
        xb, yb = G.nodes[b]["x"], G.nodes[b]["y"]
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2.5),
                    zorder=4)
    for i, n in enumerate(path_nodes[1:-1], 1):
        ax.plot(G.nodes[n]["x"], G.nodes[n]["y"], "o", color="tab:blue",
                ms=9, zorder=5)
        ax.annotate(str(i), (G.nodes[n]["x"], G.nodes[n]["y"]),
                    textcoords="offset points", xytext=(7, 7),
                    color="tab:blue", fontweight="bold", zorder=6)

    # Everything interesting, to zoom the map to it
    xs = [G.nodes[n]["x"] for n in path_nodes]
    ys = [G.nodes[n]["y"] for n in path_nodes]

    # Outline the smallest set of regions holding 50% of the probability
    cum = 0.0
    for (cx, cy), c in sorted(cells.items(), key=lambda kv: -kv[1]["p"]):
        ax.add_patch(plt.Rectangle((cx * cell_m, cy * cell_m), cell_m, cell_m,
                                   fill=False, ec="tab:green", lw=1.5, zorder=3))
        xs += [cx * cell_m, (cx + 1) * cell_m]
        ys += [cy * cell_m, (cy + 1) * cell_m]
        cum += c["p"]
        if cum >= 0.5:
            break

    # Most likely landmarks: numbered stars, names in a legend
    for rank, (x, y, label) in enumerate(places, 1):
        ax.plot(x, y, "*", color="tab:red", ms=16, mec="white", zorder=7,
                label=f"{rank}. {label}")
        ax.annotate(str(rank), (x, y), textcoords="offset points",
                    xytext=(6, 6), color="tab:red", fontweight="bold", zorder=8)
        xs.append(x)
        ys.append(y)
    if places:
        ax.legend(loc="upper left", fontsize=7, title="Most likely places",
                  title_fontsize=8, framealpha=0.85)

    pad = 800
    ax.set_xlim(max(min(xs) - pad, extent[0]), min(max(xs) + pad, extent[1]))
    ax.set_ylim(max(min(ys) - pad, extent[2]), min(max(ys) + pad, extent[3]))
    route = (f"turned {' then '.join(turns)}" if len(turns) <= 6
             else f"{len(turns)} observed turns")
    ax.set_title(f"Where is the car going?\n"
                 f"(blue route, {route} at numbered crossings)\n"
                 f"green boxes = most likely regions, together holding 50%")
    ax.set_aspect("equal")
    ax.set_xlabel("metres")
    ax.set_ylabel("metres")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--heading", type=float, default=0,
                    help="compass direction of travel when arriving at the "
                         "first crossing (0=N, 90=E)")
    ap.add_argument("--turn", nargs="+", choices=["left", "right", "straight"],
                    help="turns at successive crossings, e.g. --turn right left")
    ap.add_argument("--radius", type=float, default=3000,
                    help="map radius, metres (ignored with --graph)")
    ap.add_argument("--graph", metavar="FILE",
                    help="use a saved road network (.graphml from --build-graph) "
                         "instead of downloading one")
    ap.add_argument("--build-graph", metavar="FILE",
                    help="download the roads of --places, save them to FILE and "
                         "exit (or continue with that graph if --turn is given)")
    ap.add_argument("--places", nargs="+", default=PORTO_PLACES,
                    help="place names for --build-graph / --build-landmarks "
                         "(default: Porto, Matosinhos, Maia)")
    ap.add_argument("--landmarks", metavar="FILE",
                    help="use landmarks (.geojson from --build-landmarks) as "
                         "the destination prior instead of a uniform one")
    ap.add_argument("--build-landmarks", metavar="FILE",
                    help="download the landmarks of --places, save them to FILE "
                         "and exit (or continue with them if --turn is given)")
    ap.add_argument("--landmark-floor", type=float, default=0.2,
                    help="share of the prior spread uniformly over all "
                         "intersections (default 0.2; the rest goes to landmarks)")
    ap.add_argument("--time", metavar="HH:MM",
                    help="time the car was seen (or 'now'); with --landmarks, "
                         "favours places busy at that hour, e.g. universities "
                         "in the morning (see PROFILES)")
    ap.add_argument("--day", choices=["weekday"] + DAYS,
                    help="day the car was seen (default: weekday, or today "
                         "with --time now)")
    ap.add_argument("--landmark-sigma", type=float, default=100,
                    help="how far a landmark's weight spreads to nearby "
                         "intersections, metres (default 100)")
    ap.add_argument("--beta", type=float, default=BETA,
                    help=f"driver rationality per second of detour (default {BETA}: "
                         "a 60 s detour is ~36x less likely)")
    ap.add_argument("--trip-scale", type=float, default=TRIP_SCALE,
                    help="favour destinations within about this many seconds "
                         f"of free-flow driving (default {TRIP_SCALE}; 0 = off)")
    ap.add_argument("--cell", type=float, default=300, help="region size, metres")
    ap.add_argument("--demo", action="store_true", help="synthetic grid, offline")
    ap.add_argument("--out", default="destination")
    args = ap.parse_args()

    if args.build_graph and args.graph:
        ap.error("use either --build-graph or --graph, not both")
    if args.build_landmarks and args.landmarks:
        ap.error("use either --build-landmarks or --landmarks, not both")
    if args.demo and (args.landmarks or args.build_landmarks):
        ap.error("landmarks need a real map, not --demo")
    if (args.time or args.day) and not (args.landmarks or args.build_landmarks):
        ap.error("--time / --day weight the landmarks, so they need --landmarks")
    try:
        when = parse_when(args.time, args.day)
    except ValueError as e:
        ap.error(str(e))
    G = None
    if args.build_graph:
        G = build_place_graph(args.places, args.build_graph)
    if args.build_landmarks:
        build_landmarks(args.places, args.build_landmarks)
        args.landmarks = args.build_landmarks
    if not args.turn:
        if args.build_graph or args.build_landmarks:
            return
        ap.error("--turn is required")

    if args.demo:
        G, X = make_demo_graph()
    else:
        if args.lat is None or args.lon is None:
            ap.error("--lat and --lon are required unless --demo is used")
        if args.graph:
            G = load_graph_file(args.graph)
        if G is not None:
            X = nearest_node(G, args.lat, args.lon)
        else:
            G, X = load_osm_graph(args.lat, args.lon, args.radius)

    landmarks = None
    if args.landmarks:
        landmarks = LandmarkPrior(G, args.landmarks, args.landmark_sigma,
                                  args.landmark_floor, when)
        print("\n".join(landmarks.describe()))

    route = route_from_turns(G, X, args.heading, args.turn)
    res = Predictor(G).predict(route, beta=args.beta, trip_scale=args.trip_scale,
                               landmarks=landmarks, observed="turn",
                               cell_m=args.cell)

    rows = write_csv(res.cells, args.cell, f"{args.out}_regions.csv")
    places = []
    if landmarks:
        write_places_csv(landmarks, res.places, res.elsewhere,
                         f"{args.out}_places.csv")
        places = [(*landmarks.points[i], landmarks.names[i] or landmarks.categories[i])
                  for i, _ in res.places[:5]]
    plot(G, res.nodes, res.post, res.cells, args.cell, res.turns,
         f"{args.out}_map.png", places)

    if landmarks:
        print("\nTop places:")
        cum = 0.0
        for rank, (i, p) in enumerate(res.places[:10], 1):
            cum += p
            print(f"  {rank:2d}. {p * 100:5.2f}%  (cum {cum * 100:5.1f}%)  "
                  f"{landmarks.label(i)}")
        print(f"      {res.elsewhere * 100:5.2f}%  elsewhere (no landmark)")

    print(f"\nTop regions ({args.cell:.0f} m squares):")
    cum = 0.0
    for rank, ((cx, cy), c) in enumerate(rows[:10], 1):
        cum += c["p"]
        where = (f"lat {np.mean(c['lat']):.5f}, lon {np.mean(c['lon']):.5f}"
                 if c["lat"] else
                 f"x={(cx + .5) * args.cell:.0f} m, y={(cy + .5) * args.cell:.0f} m")
        print(f"  {rank:2d}. {c['p'] * 100:5.2f}%  (cum {cum * 100:5.1f}%)  {where}")
    print(f"\nWrote {args.out}_map.png and {args.out}_regions.csv"
          + (f" and {args.out}_places.csv" if landmarks else ""))


if __name__ == "__main__":
    main()
