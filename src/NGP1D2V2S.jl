using FFTW,LinearAlgebra,Plots,SpecialFunctions,Roots, StaticArrays, ProgressMeter
using LoopVectorization, Base.Threads
igr(d) = 1 / Roots.find_zero(x->x^(d+1) - x - 1, 1.5)
R(α, N, s0=0) = [rem(s0 + n * α, 1) for n in 1:N]
function boris(vx, vy, E, B, dt, q_m)
  dt2q_m = dt/2 * q_m
  v⁻ = @SArray [vx + E * dt2q_m, vy, 0.0]
  t = @SArray [0, 0, B * dt2q_m]
  v⁺ = v⁻ + 2 * cross(v⁻ + cross(v⁻, t), t) / (1 + dot(t, t))
  return (v⁺[1] + E * dt2q_m, v⁺[2])
end
function pic()
  N=256;P=8N;T=2^15;TO=T÷32;M=8;
  n0=4*pi^2;vth=sqrt(n0)/N;dt=1/N/8vth;B0=sqrt(n0)/8;w=n0/2P;
  @show N, P, T, TO, n0, vth, B0, dt
  @show 2pi^2 * (vth / B0)^2
  ξ=im*zeros(N);Es=zeros(N,TO); 
  x1=mod.((bitreverse.(0:P-1).+2.0^63)/2.0^64, 1);
  x2=mod.((bitreverse.(0:P-1).+2.0^63)/2.0^64 .+ pi, 1);
  vx1=vth * erfinv.(R(igr(1),P));vy1=vth * erfinv.(R(igr(2),P));
  vx2=deepcopy(vx1) ./ sqrt(M);vy2=deepcopy(vy1) ./ sqrt(M);
  r=zeros(N);E=zeros(N);
  ik=2π*im*vcat(1,1:N/2,-N/2+1:-1);D=zeros(TO,5);
  f(g,c)=erf((g-c)*N)/2;ff(i,c)=f((i+0.5)/N,c)-f((i-0.5)/N,c);l=4eps();
  d(c)=((mod1(i,N),ff(i,c)) for i∈(-7:7).+Int(round(c*N)));F=copy(E);
  ρ(x,q)=(for l in d.(x);for k in l;r[k[1]]+=q*k[2]*w;end;end);
  ρ() = (r.*=0;ρ(x1, -1);ρ(x2, 1);r)

  chunks = collect(Iterators.partition(1:P, ceil(Int, P/nthreads())))
  @showprogress for t in 1:T;
   E.=real.(ifft((ξ.=fft(ρ())./ik;ξ[1]*=0;ξ)));
   hv1 = @spawn @threads for i in eachindex(chunks); for j in chunks[i];
     Ej = sum(k->E[k[1]]*k[2], d(x1[j]));
     vx1[j], vy1[j] = boris(vx1[j], vy1[j], Ej, B0, dt, -1);
     x1[j]+=vx1[j]*dt;
   end; end
   hv2 = @spawn @threads for i in eachindex(chunks); for j in chunks[i];
     Ej = sum(k->E[k[1]]*k[2], d(x2[j]));
     vx2[j], vy2[j] = boris(vx2[j], vy2[j], Ej, B0, dt, 1/M);
     x2[j]+=vx2[j]*dt;
   end; end
   wait(hv1)
   wait(hv2)
   hx1 = @spawn @turbo x1.=mod.(x1,1);
   hx2 = @spawn @turbo x2.=mod.(x2,1);
   ti = cld(t, T÷TO)
   Es[:,ti] .+= E;
   wait(hx1)
   wait(hx2)
   if mod(t, T÷TO) == 0
     D[ti,1:2].+=(sum(abs2, E)/N,sum(vy1[i]^2 + vx1[i]^2 + M*(vy2[i]^2 + vx2[i]^2) for i in 1:P)*n0/P)./2;
     D[ti,3:5].+=sum.(((@view D[ti,1:2]),vx1/P + M*vx2/P,vy1/P+M*vy2/P));
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
xlabel!("Wavenumber");ylabel!("Frequency")
savefig("Bernstein_WK.png")
