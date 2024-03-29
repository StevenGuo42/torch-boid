# PyTorch Boids Simulation

This is a simple boids simulation using PyTorch. It is based on the [Craig Reynolds' boids model](https://en.wikipedia.org/wiki/Boids).

Algorithm is parallelized using PyTorch tensors and CUDA.

> **Note:** This is a work in progress. The code contains bugs and some core features are not fully implemented.
> Demo of the current state of the simulation:
> ![boids](./img/boids.gif)

## Requirements

- Python 3.7+
- PyTorch 2.1+
- pygame 2.5+
- CUDA 12.1

## TODO

- ### Features

  - [x] Add separation, alignment and cohesion to boids
  - [x] Add simulation using pygame
  - [ ] Add bias to boids
  - [ ] Add a GUI to control the simulation
  - [ ] Add predators
  - [ ] Add better velocity indicators
  - [ ] Add a better way to visualize the boids
  - [ ] Add obstacles

## References

- [Boids Wikipedia page](https://en.wikipedia.org/wiki/Boids)
- [Craig Reynold's Boids page](https://www.red3d.com/cwr/boids/)
- [Boids algorithm from V. Hunter Adams](https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html)