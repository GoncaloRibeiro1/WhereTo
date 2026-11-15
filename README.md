# WhereTo?

## How this started

I was driving home one night, stuck in traffic about 50 metres from a red light, when the car in front of me turned on its right indicator. It was far too early, and I found it funny.

Then I started thinking about it. The blinker was telling me what he'd do at this turn, but could an algorithm use it to work out where he was going in the end? If it could, and he was signalling this early, someone could get to his destination before he did. My mind went straight to police chases, where knowing where a car is likely to end up could be really useful.

So I built this. **WhereTo?** takes a car seen turning left, right or straight at a junction and estimates where it's most likely heading. It checks every possible destination on a real road network, asks how sensible that turn would be for a driver going there, and returns a probability heatmap of the likely places.

## Running the map app

These steps take you from a fresh clone to the app in your browser. They were tested on Ubuntu with Python 3.12.

### 1. First-time setup (once)

Create the virtual environment and install the dependencies:

```bash
python3 -m venv .venv            # on Ubuntu you may first need: sudo apt install python3-venv
source .venv/bin/activate
pip install osmnx networkx matplotlib numpy fastapi uvicorn
```

`osmnx` also installs `geopandas`, `shapely` and `pyproj`, which the server needs.

Then download the Porto data from OpenStreetMap. The road network and the landmarks are not in the repository because they are large, and this step needs internet access:

```bash
python whereto.py --build-graph porto.graphml                    # roads, takes about 2 min
python whereto.py --build-landmarks porto_landmarks.geojson      # hospitals, malls, universities, ...
```

These commands create `porto.graphml` and `porto_landmarks.geojson` in the project folder. The landmarks file is optional: without it the app still runs, but it treats every intersection as an equally likely destination.

### 2. Start the server (every time)

```bash
source .venv/bin/activate        # once per new terminal
python server.py
```

It loads `porto.graphml` and `porto_landmarks.geojson` by default and takes a few seconds to start. Leave this terminal open. Press Ctrl+C to stop the server.

### 3. Open it in the browser

Go to **http://127.0.0.1:8000/**.

- Click a road to place the car. The arrow shows which way it is driving; on a two-way street, click the flip button to reverse it.
- Click one of the pulsing circles to drive to the next junction. The heatmap and the top-10 list update after each move.
- Backspace or Ctrl+Z undoes a move, and Esc resets.
- Use the settings panel to change the time, the day, the landmarks, β, the trip scale and the cell size.

The server log prints `Open http://127.0.0.1:8000/docs`. That page is the API documentation. The map itself is at `/`.

Useful options:

```bash
python server.py --port 8080                  # use a different port
python server.py --host 0.0.0.0               # reach it from other devices on your network
python server.py --landmarks ''               # run without landmarks
python server.py --graph other.graphml        # use another city built with --build-graph
```

### 4. Stop and leave the venv

Press Ctrl+C in the server terminal to stop the server, then leave the virtual environment:

```bash
deactivate
```

The `(.venv)` prefix disappears from your prompt. Closing the terminal also leaves the venv.

## Command-line use (no browser)

`whereto.py` also works on its own and writes a heatmap PNG and CSV files:

```bash
python whereto.py --demo --heading 0 --turn right left          # synthetic grid, works offline
python whereto.py --graph porto.graphml --landmarks porto_landmarks.geojson \
    --lat 41.1622 --lon -8.5989 --heading 0 --turn right right --time 08:30
```

`--heading` is the compass direction the car was driving when it reached the first junction (0 = north, 90 = east). Run `python whereto.py --help` for every option.
