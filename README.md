# WhereTo

## How this started

I was driving home one night, stuck in traffic about 50 metres from a red light, when the car in front of me turned on its right indicator. It was far too early, and I found it funny.

Then I started thinking about it. The blinker was telling me what he'd do at this turn, but could an algorithm use it to work out where he was going in the end? If it could, and he was signalling this early, someone could get to his destination before he did. My mind went straight to police chases, where knowing where a car is likely to end up could be really useful.

So I built this. **WhereTo** takes a car seen turning left, right or straight at a junction and estimates where it's most likely heading. It checks every possible destination on a real road network, asks how sensible that turn would be for a driver going there, and returns a probability heatmap of the likely places.
