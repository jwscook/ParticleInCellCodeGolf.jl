# ParticleInCellCodeGolf.jl

This is a silly repo for tiny inscrutable PIC codes!

So far they are all a collisionless electrostatic 1D1V (one spatial dimension 
and one velocity dimension) particle-in-cell codes.

**Warning: There will be bugs!**

## 7 Lines

I wrote this one for fun for the Julia discourse thread [Seven Lines of Julia (examples sought)](https://discourse.julialang.org/t/seven-lines-of-julia-examples-sought/50416/113?u=jcook).

```julia
using FFTW,Plots;N=128;P=64N;dt=1/4N;NT=1024;W=200;w=W/P*N;
x=rand(P);v=collect(1:P.>P/2).*2 .-1;u()=(x.=mod.(x.+v/2*dt,1));n=zeros(N);
k=im.*2π*vcat(1,1:N/2,-N/2+1:-1);f(x)=Int(mod1(round(x*N),N));Σ=sum;
@gif for t in 1:NT;
  u();E=real.(ifft((E=fft((n.*=0;for j∈x;n[f(j)]+=w;end;n))./k;E[1]*=0;E)));
  u();v+=E[f.(x)]*dt;scatter(x,v,ylims=(-3,3),legend=false);
end every 8
```

It uses Nearest Grid Point (NGP) particle deposition to acculumate charge on a
grid and a fast Fourier transform (FFT) to calculate the electric field. It uses
an explicit leapfrog (Verlet) time stepper where the CFL condition hard coded
but is set by the fastest particle in the simulation.

The code is set up to display the two-stream instability:

![](https://github.com/jwscook/ParticleInCellCodeGolf.jl/blob/main/gifs/NGPFourierWithDiagnostics.gif)

Points represent the particles and The traces drawn across the plot represent
the total energy, particle energy, wave energy, and total momentum, from top to
bottom respectively (gif generated with src/NGPFourierWithDiagnostics.jl).

## Gaussian shapes particles

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
using FFTW,LinearAlgebra,Plots,SpecialFunctions;N=128;P=32N;dt=1/6N;T=1024;
W=400;w=W/P*N;ξ=im*zeros(N);x=rand(P);v=rand([-1.0,1.0],P);r=zeros(N);
E=zeros(N);X=copy(x);V=copy(v);ik=2π*im*vcat(1,1:N/2,-N/2+1:-1);D=zeros(T,4);
f(g,c)=erf((g-c)*N)/2;ff(i,c)=f((i+0.5)/N,c)-f((i-0.5)/N,c);l=1e-8;
d(c)=((mod1(i,N),ff(i,c)) for i∈(-6:6).+Int(round(c*N)));F=copy(E);
ρ(x,y)=(r.*=0;for j∈d.((x.+y)./2);for k∈j;r[k[1]]+=k[2]*w;end;end;r);
@gif for t ∈1:T;X.=x;V.=v;F.*=NaN;for _∈0:9;isapprox(F,E,rtol=l)&&break;F.=E;
  x.=X.+(v.+V)/2*dt;E.=real.(ifft((ξ.=fft(ρ(x,X))./ik;ξ[1]*=0;ξ)));
  for j∈1:P;v[j]=V[j]+sum(k->E[k[1]]*k[2],d((x[j]+X[j])/2))*dt;end;end;x.=mod.(x,1);
  D[t,1:2].=(sum(E.^2)/N,sum(v.^2)*W/P)./2;D[t,3:4].=sum.((D[t,1:2],v/P));
  scatter(x,v,ylims=(-3,3));D[t,1:3].*=2/W;plot!((1:t)./T,D[1:t,:],legend=0==1)
end every 8
```

![](https://github.com/jwscook/ParticleInCellCodeGolf.jl/blob/main/gifs/GaussianFixedPoint.gif)

Note that the trajectories of the particles are smoother, this is because the particle
shape functions are smoother. It's almost possible to see the step changes in velocity
in the gif from the NGP version of the code.

## Quiet start

The bit reversal algorithm can be used as a low discrepancy number generator to
initialise a "quiet start". This reduces noise to a level whereby the electric field
energy grows over the course of ~30 decades! As standard, twice the predicted growth rate is overplotted on the change in integrated electric field energy trace with outstanding agreement.

```julia
using FFTW,LinearAlgebra,Plots,SpecialFunctions;pic=()->(N=64;P=32N;dt=1/6N;
T=2^13;W=32π^2/3;w=W/P*N;ξ=im*zeros(N);x=(bitreverse.(0:P-1).+2.0^63)/2.0^64;
v=collect(1:P.>P/2).*2 .-1.;r=zeros(N);
E=zeros(N);X=copy(x);V=copy(v);ik=2π*im*vcat(1,1:N/2,-N/2+1:-1);D=zeros(T,4);
f(g,c)=erf((g-c)*N)/2;ff(i,c)=f((i+0.5)/N,c)-f((i-0.5)/N,c);l=4eps();
d(c)=((mod1(i,N),ff(i,c)) for i∈(-7:7).+Int(round(c*N)));F=copy(E);
ρ(x,y=x)=(r.*=0;for l∈d.((x.+y)./2);for k∈l;r[k[1]]+=k[2]*w;end;end;r);
@time @gif for t ∈1:T;X.=x;V.=v;F.*=NaN;for _∈0:9;≈(F,E,rtol=l,atol=0)&&break;F.=E;
 x.=X.+(v.+V)/2*dt;E.=real.(ifft((ξ.=fft(ρ(x,X))./ik;ξ[1]*=0;ξ)));
 for j∈1:P;v[j]=V[j]+sum(k->E[k[1]]*k[2],d((x[j]+X[j])/2))*dt;end;end;x.=mod.(x,1);
 D[t,1:2].=(sum(E.^2)/N,sum(v.^2)*W/P)./2;D[t,3:4].=sum.((D[t,1:2],v/P));
 scatter(x[1:P÷2],v[1:P÷2],c=1);scatter!(x[P÷2+1:P],v[P÷2+1:P],ylims=(-3,3),c=2);
 D[t,1:3].*=2/W;plot!((1:t)./T,D[1:t,:],legend=0==1,xlims=(0,1)); @show t/T
 plot!((0.5:N)/N,ρ(x[1:P÷2])./W,c=1);plot!((0.5:N)/N,ρ(x[P÷2+1:P])./W,c=2)
end every 8;return (D,N,P,dt,T,W,w););(D,N,P,dt,T,W,w)=pic()
la(x)=log10(abs(x));t=(1:T).*dt;plot(t,la.(D[:,1]),label="electric field energy")
plot!(t,la.(1 .-D[:,3]),label="total energy change");yticks!(-35:1)
plot!(t,la.(D[:,4]),label="total momentum",legend=:bottomright)
γ(x)=imag(sqrt(Complex(x^2+1-sqrt(4*x^2+1))))*sqrt(W/2)/log(10);ylims!(-35,1)
plot!(t,2*γ(2π/sqrt(W/2)).*(t.-T÷8*dt).+la.(D[T÷8,1]),label="predicted")
xlabel!("Time");ylabel!("log 10");savefig("GaussianFixedPointQuiet.jpg")
```

<img src="https://github.com/jwscook/ParticleInCellCodeGolf.jl/blob/main/figs/GaussianFixedPointQuiet.jpg" width="600" height="400">

![](https://github.com/jwscook/ParticleInCellCodeGolf.jl/blob/main/gifs/GaussianFixedPointQuiet.gif)

