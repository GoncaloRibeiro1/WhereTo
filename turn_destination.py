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
                          turn. Default: every intersection equally likely;
                          optionally favour shorter trips (--trip-scale).
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

Usage
-----
Real map (needs internet access to OpenStreetMap):
    python turn_destination.py --lat 38.7223 --lon -9.1393 --heading 0 \
        --turn right left --radius 3000

  --lat/--lon   point at or near the first crossing (nearest intersection)
  --heading     compass direction the car was driving when it arrived at the
                first crossing (0 = north, 90 = east, 180 = south, 270 = west)
  --turn        one or more of: left | right | straight

Cached city graph (download once, then reuse offline):
    python turn_destination.py --build-graph porto.graphml
    python turn_destination.py --graph porto.graphml --lat 41.1622 \
        --lon -8.5989 --heading 0 --turn right right

  --build-graph downloads the roads of --places (default: Porto, Matosinhos,
  Maia, Vila Nova de Gaia, Gondomar) and saves them; --graph loads that file
  instead of downloading, and --radius is then ignored.

Offline demo (synthetic city grid, no internet needed):
    python turn_destination.py --demo --turn right left

Outputs: a PNG heatmap and a CSV of regions ranked by probability.

Requires: pip install osmnx networkx matplotlib numpy
"""

import argparse
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Turn classes by the change in heading (degrees, + = clockwise = to the right)
STRAIGHT_MAX = 30     # |change| < 30°           -> straight
TURN_MAX = 150        # 30° <= |change| <= 150°  -> left / right
                      # beyond 150°              -> U-turn (not allowed)

# Ideal heading change for each class, used to pick one road when a class
# has several (e.g. two roads that both count as "right").
IDEAL_DELTA = {"straight": 0, "right": 90, "left": -90}


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
def find_approach(G, X, heading):
    """Pick the incoming road whose arrival heading best matches `heading`."""
    best = None
    for u, _, k, data in G.in_edges(X, keys=True, data=True):
        b = arrival_bearing(G, u, X, data)
        err = abs(_angle_diff(heading, b))
        if best is None or err < best[0]:
            best = (err, u, b)
    if best is None:
        raise ValueError("No road enters this intersection.")
    err, u, b = best
    if err > 45:
        print(f"Warning: closest incoming road has heading {b:.0f}°, "
              f"{err:.0f}° away from the {heading:.0f}° you gave.")
    return u, b


def exits_by_turn(G, X, came_from, in_bearing):
    """Group the roads leaving X into 'left' / 'right' / 'straight'."""
    exits = defaultdict(list)       # class -> list of (v, travel_time, delta, data)
    for _, v, k, data in G.out_edges(X, keys=True, data=True):
        if v == came_from:
            continue                # going back where it came from = U-turn
        delta = _angle_diff(in_bearing, departure_bearing(G, X, v, data))
        if abs(delta) < STRAIGHT_MAX:
            cls = "straight"
        elif abs(delta) <= TURN_MAX:
            cls = "right" if delta > 0 else "left"
        else:
            continue                # sharp U-turn, not allowed
        exits[cls].append((v, data["travel_time"], delta, data))
    return exits


def turn_likelihood(G, exits, observed_turn, beta, nodes):
    """
    P(observed_turn | d) at one crossing, for every d in `nodes`.

    For every exit road e = X->v, T_e(d) = time(X->v) + fastest time v->d.
    P(exit e | d) = softmax(-beta * T_e(d)) over all exits.
    P(turn class | d) = sum of P(exit | d) over exits in that class.
    Also returns the fastest time from this crossing to each d.
    """
    # One Dijkstra per exit road gives the time to every destination at once.
    options = []                                      # (class, {d: T_e(d)})
    for cls, lst in exits.items():
        for v, t_edge, _, _ in lst:
            dist = nx.single_source_dijkstra_path_length(G, v, weight="travel_time")
            options.append((cls, {d: t_edge + t for d, t in dist.items()}))

    T = np.full((len(options), len(nodes)), np.inf)
    for i, (_, dist) in enumerate(options):
        for j, d in enumerate(nodes):
            if d in dist:
                T[i, j] = dist[d]

    best = T.min(axis=0)
    reachable = np.isfinite(best)
    # Softmax over exits (shifted by the best time for numerical stability)
    w = np.zeros_like(T)
    w[:, reachable] = np.exp(-beta * (T[:, reachable] - best[reachable]))
    total = w.sum(axis=0)
    p_exit = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    obs = np.array([cls == observed_turn for cls, _ in options])
    return p_exit[obs].sum(axis=0), best


def pick_exit(exits, turn):
    """The road the car took for `turn`. If several roads fall in that class,
    take the one closest to an ideal straight / 90° turn."""
    lst = exits[turn]
    if len(lst) > 1:
        print(f"    note: {len(lst)} roads count as '{turn}' here; "
              f"using the one closest to {IDEAL_DELTA[turn]:+d}°")
    return min(lst, key=lambda e: abs(e[2] - IDEAL_DELTA[turn]))


def destination_posterior(G, X, heading, turns, beta, trip_scale=None):
    """
    Follow the car through the observed `turns`, starting at crossing X.

    Returns ({node: P(dest = node | turns)}, path), where path is the list
    of nodes the car passed through, starting with the road it arrived on.
    """
    came_from, in_bearing = find_approach(G, X, heading)
    path = [came_from, X]
    steps = []                                  # (crossing, exits, turn)

    for k, turn in enumerate(turns, 1):
        exits = exits_by_turn(G, X, came_from, in_bearing)
        print(f"Crossing {k} {X}: arriving at heading {in_bearing:.0f}°, exits: "
              + ", ".join(f"{c} ({len(v)})" for c, v in exits.items())
              + f"  ->  turned {turn}")
        if turn not in exits:
            raise ValueError(f"No '{turn}' exit at crossing {k} {X}; "
                             f"available: {sorted(exits)}")
        steps.append((X, exits, turn))

        # Drive to the next crossing along the road that was taken
        if k < len(turns):
            v, _, _, data = pick_exit(exits, turn)
            came_from, in_bearing = X, arrival_bearing(G, X, v, data)
            X = v
            path.append(X)
        else:
            path.append(pick_exit(exits, turn)[0])   # for the map arrow only

    # A car that drove through a crossing wasn't going to that crossing.
    passed = {X0 for X0, _, _ in steps}
    nodes = [d for d in G.nodes if d not in passed]

    likelihood = np.ones(len(nodes))
    first_best = None
    for X0, exits, turn in steps:
        lk, best = turn_likelihood(G, exits, turn, beta, nodes)
        likelihood *= lk
        if first_best is None:
            first_best = best

    # Prior P(d): uniform, or favour trips of about `trip_scale` seconds
    # (trip time measured from the first crossing).
    reachable = np.isfinite(first_best)
    prior = np.where(reachable, 1.0, 0.0)
    if trip_scale:
        prior[reachable] = np.exp(-first_best[reachable] / trip_scale)

    post = likelihood * prior
    if post.sum() == 0:
        raise ValueError("No destination is consistent with these turns.")
    post /= post.sum()
    keep = post > 0
    return dict(zip([d for d, k in zip(nodes, keep) if k], post[keep])), path


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


def plot(G, path_nodes, post, cells, cell_m, turns, path):
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
    cb.set_label(f"P(destination in region | turned {', '.join(turns)})")

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

    # Outline the smallest set of regions holding 50% of the probability
    cum = 0.0
    for (cx, cy), c in sorted(cells.items(), key=lambda kv: -kv[1]["p"]):
        ax.add_patch(plt.Rectangle((cx * cell_m, cy * cell_m), cell_m, cell_m,
                                   fill=False, ec="tab:green", lw=1.5, zorder=3))
        cum += c["p"]
        if cum >= 0.5:
            break

    ax.set_title(f"Where is the car going?\n"
                 f"(blue route, turned {' then '.join(turns)} at numbered crossings)\n"
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
                    help="place names for --build-graph (default: Porto, "
                         "Matosinhos, Maia, Vila Nova de Gaia, Gondomar)")
    ap.add_argument("--beta", type=float, default=0.03,
                    help="driver rationality per second of detour (default 0.03: "
                         "a 60 s detour is ~6x less likely)")
    ap.add_argument("--trip-scale", type=float, default=None,
                    help="favour trips of about this many seconds (default: off)")
    ap.add_argument("--cell", type=float, default=300, help="region size, metres")
    ap.add_argument("--demo", action="store_true", help="synthetic grid, offline")
    ap.add_argument("--out", default="destination")
    args = ap.parse_args()

    if args.build_graph and args.graph:
        ap.error("use either --build-graph or --graph, not both")
    G = None
    if args.build_graph:
        G = build_place_graph(args.places, args.build_graph)
        if not args.turn:
            return
    if not args.turn:
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

    post, path_nodes = destination_posterior(G, X, args.heading, args.turn,
                                             args.beta, args.trip_scale)
    cells = aggregate_regions(G, post, args.cell)
    rows = write_csv(cells, args.cell, f"{args.out}_regions.csv")
    plot(G, path_nodes, post, cells, args.cell, args.turn, f"{args.out}_map.png")

    print(f"\nTop regions ({args.cell:.0f} m squares):")
    cum = 0.0
    for rank, ((cx, cy), c) in enumerate(rows[:10], 1):
        cum += c["p"]
        where = (f"lat {np.mean(c['lat']):.5f}, lon {np.mean(c['lon']):.5f}"
                 if c["lat"] else
                 f"x={(cx + .5) * args.cell:.0f} m, y={(cy + .5) * args.cell:.0f} m")
        print(f"  {rank:2d}. {c['p'] * 100:5.2f}%  (cum {cum * 100:5.1f}%)  {where}")
    print(f"\nWrote {args.out}_map.png and {args.out}_regions.csv")


if __name__ == "__main__":
    main()
