using FFTW,LinearAlgebra,Plots,SpecialFunctions,Roots, StaticArrays, ProgressMeter
using LaTeXStrings
igr(d) = 1 / Roots.find_zero(x->x^(d+1) - x - 1, 1.5)
R(α, N, s0=0) = [rem(s0 + n * α, 1) for n in 1:N]
function boris(vx, vy, E, B, dt)
  v⁻ = @SArray [vx + E * dt/2, vy, 0.0]
  t = @SArray [0, 0, B * dt/2]
  v⁺ = v⁻ + 2 * cross(v⁻ + cross(v⁻, t), t) / (1 + dot(t, t))
  return (v⁺[1] + E * dt/2, v⁺[2])
end
function halton(i, base, seed=0.0)
  result, f = 0.0, 1.0
  while i > 0
    f = f / base;
    result += f * mod(i, base)
    i ÷= base;
  end
  return mod(result + seed, 1)
end

function pic()
  N=128;P=15N;T=2^13;TO=T÷16;
  n0=4*pi^2;vth=sqrt(n0)/N;dt=1/N/6vth;B0=sqrt(n0)/16;w=n0/P;
  @show N, P, T, TO, n0, vth, B0, dt
  @show 2pi^2 * (vth / B0)^2
  ξ=im*zeros(N);Es=zeros(N,TO);
  rx = rand(P);#halton.(0:P-1, 2, 1/sqrt(2));(bitreverse.(0:P-1).+2.0^63)/2.0^64;
  rvx = rand(P);#halton.(0:P-1, 3, 1/sqrt(2));R(igr(1),P)
  rvy = rand(P);#halton.(0:P-1, 5, 1/sqrt(2));R(igr(2),P)
  x = rx;
  vx = vth * erfinv.(rvx);
  vy = vth * erfinv.(rvy);

  r=zeros(N);E=zeros(N);X=copy(x);Vx=copy(vx);Vy=copy(vy);
  ik=2π*im*vcat(1,1:N/2,-N/2+1:-1);D=zeros(TO,5);
  f(g,c)=erf((g-c)*N)/2;ff(i,c)=f((i+0.5)/N,c)-f((i-0.5)/N,c);l=4eps();
  d(c)=((mod1(i,N),ff(i,c)) for i∈(-7:7).+Int(round(c*N)));F=copy(E);
  ρ(x,y=x)=(r.*=0;for l∈d.((x.+y)./2);for k∈l;r[k[1]]+=k[2]*w;end;end;r);
  @time @showprogress for t in 1:T;
   E.=real.(ifft((ξ.=fft(ρ(x))./ik;ξ[1]*=0;ξ)));
   Base.Threads.@threads for j∈1:P;
     Ej = sum(k->E[k[1]]*k[2], d(x[j]));
     vx[j], vy[j] = boris(vx[j], vy[j], Ej, B0, dt);
     x[j] += vx[j] * dt;
   end
   #X.=x;Vx.=vx;Vy.=vy;F.*=NaN;
   #for _∈0:0;
   #  ≈(F,E,rtol=l,atol=0)&&break;F.=E;
   #  x.=X.+(vx.+Vx)/2*dt;E.=real.(ifft((ξ.=fft(ρ(x,X))./ik;ξ[1]*=0;ξ)));
   #  Base.Threads.@threads for j∈1:P;
   #    Ej = sum(k->E[k[1]]*k[2], d((x[j]+X[j])/2));
   #    vx[j], vy[j] = boris(Vx[j], Vy[j], Ej, B0, dt);
   #  end;
   #end;
   x.=mod.(x,1);
   ti = cld(t, T÷TO)
   Es[:,ti] .+= E;
   if mod(t, T÷TO) == 0
     D[ti,1:2].+=(sum(abs2, E)/N,sum(vy[i]^2 + vx[i]^2 for i in 1:P)*n0/P)./2;
     D[ti,3:5].+=sum.(((@view D[ti,1:2]),vx/P,vy/P));
     D[ti,1:3].*=2/n0;
   end
  end# every 8;
  D ./= T÷TO; Es ./= T÷TO
  return (Es,D,N,P,dt,T,n0,w,B0,vth);
end
(Es,D,N,P,dt,T,n0,w,B0,vth)=pic()
@show T, dt, B0, vth, n0
@show T * dt / (2pi / B0)
plot(D)
savefig("Bernstein_D.png")
x = (1:N) ./ N ./ (vth / B0)
t = (1:size(Es, 2)) .* ((T * dt / size(Es, 2)) / (2pi/B0))
heatmap(x, t, Es')
xlabel!("Space [vth/Omega_c] ");ylabel!("Time [tau_c]")
savefig("Bernstein_TX.png")
filter = sin.(((1:size(Es,2)) .- 0.5) ./ size(Es,2) .* pi)'
ws = 2π/(T * dt) .* (1:size(Es,2)÷2) ./ (B0);
ks = 2π .* (1:N÷2-1) ./ (B0/vth);
Z = log10.(abs.(fft((Es .* filter)')))[1:end÷2, 2:end÷2];
heatmap(ks, ws, Z)
xlabel!(L"Wavenumber $[\Omega_c / v_{th}]$");
ylabel!(L"Frequency $[\Omega_c]$")
savefig("Bernstein_WK.png")
