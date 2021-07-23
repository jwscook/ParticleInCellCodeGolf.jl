using FFTW, Plots, SpecialFunctions;
NX=128; NP=64NX; dt = 1/10NX; NT=1024; W=1600; w=W/NP; dx=1/NX;
x=rand(NP); v=2collect(1:NP.>NP/2) .-1.; u()=(x.=mod.(x.+v/2*dt,1))
ik=2π*im*vcat(1,1:NX/2,-NX/2+1:-1); K=zeros(NT, 2); ρ=zeros(NX);
f(g, c)=erf((g-c)/dx)/2; ff(i, c) = f((i+0.5)*dx,c)-f((i-0.5)*dx,c)
d(c)=((mod1(i,NX),ff(i, c)) for i∈(-6:6).+Int(round(c*NX)))
rho()=(fill!(ρ, 0);for j∈d.(x);for k∈j;ρ[k[1]]+=k[2]*w/dx;end;end;ρ)
@gif for t in 1:NT; 
  u(); E=real.(ifft((E=fft(rho())./ik;E[1]*=0;E))); u();
  for j∈eachindex(v);v[j]+=sum(k->E[k[1]]*k[2],d(x[j]))*dt;end;
  K[t,:].=(sum(E.^2)*dx,sum(v.^2)*w)./2; scatter(x,v,ylims=(-3,3))
end every 4
