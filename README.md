# ParticleInCellCodeGolf.jl

This is a silly repo for tiny inscrutable PIC codes!

So far they are all a collisionless electrostatic 1D1V (one spatial dimension 
and one velocity dimension) particle-in-cell codes.

**Warning: There will be bugs!**

## 7 Lines

I wrote this one for fun for the Julia discourse thread [Seven Lines of Julia (examples sought)](https://discourse.julialang.org/t/seven-lines-of-julia-examples-sought/50416/113?u=jcook).

```julia
# src/7LinePIC.jl
```

It uses Nearest Grid Point (NGP) particle deposition to acculumate charge on a
grid and a fast Fourier transform (FFT) to calculate the electric field. It uses
an explicit leapfrog (Verlet) time stepper where the CFL condition hard coded
but is set by the fastest particle in the simulation.

The code is set up to display the two-stream instability:

![](https://github.com/jwscook/ParticleInCellCodeGolf.jl/blob/main/gifs/NGPFourierWithDiagnostics.gif)

Points represent the particles and The traces drawn across the plot represent
the total energy, particle energy, wave energy, and total momentum, from top to
bottom respectively.

## Gausian shapes particles

NGP particles are so noisey so I created a PIC code where the particles all have
Gaussian shapes. Charge is deposited in each cell based on the integral of the
shape function across each cell.

Now that particles' charge smoothly transition between cells and being inspired
by implicit PIC methods, I thought I'd add
some fixed point iteration to calculate the mid-point electric field from
updates to the particles' positions and velocities. This conserves momentum
perfectly and does a pretty great job at keeping energy bounded too.
Furthermore, the CFL condition is lifted! (at the expense of energy conservation
being not quite so good.)

```julia
# src/GaussianFixedPoint.jl
```

![](https://github.com/jwscook/ParticleInCellCodeGolf.jl/blob/main/gifs/GaussianFixedPoint.gif)
