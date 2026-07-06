## Current state of af3-neutrons

Correctly recreate original af3 output even with diffusion overhead and no experimental data passed
![image](beta_lac.png)

Uses [hydride-jax](github.com/vivek-booshan/hydride-jax) to continuosly refine hydrogens.

Next up:
* recreate af3 output with exp data passed and placeholder loss
* custom loss to handle exp data hydrogens
