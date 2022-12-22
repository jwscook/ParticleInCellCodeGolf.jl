using FFTW,Plots, SpecialFunctions, StaticArrays, LinearAlgebra
using LoopVectorization, Base.Threads, ThreadsX, Base.Iterators, Statistics
using ProgressMeter, LaTeXStrings
FFTW.set_num_threads(Threads.nthreads())
Plots.gr()
function halton(i, base, seed=0.0)
  result, f = 0.0, 1.0
  while i > 0
    f = f / base;
    result += f * mod(i, base)
    i ÷= base;
  end
  return mod(result + seed, 1)
end

function foo()
sq = 2 * 3 * 5 * 7 * 11
NX=NY=2^8;
NP1 = ceil(Int, NX * NY/(2*3*5*7*11)) * (2*3*5*7*11)
P=NX*NY*2^7;T=2^13;NS=16;TO=T÷NS;NG=sqrt(NX^2 + NY^2)
n0=4*pi^2;vth=sqrt(n0)/NG;dt=1/NG/10vth;B0=sqrt(n0)/4;w=n0/P;
Δ=1/NG;Δx=1/NX;Δy=1/NY
@show NX, NY, P, T, TO, NS, n0, vth, B0, dt
@show vth * dt / Δ, vth / B0 / Δy
@show 2pi/sqrt(n0) / (NS * dt), 2pi/B0 / (NS * dt)
@show T * dt / (2pi/sqrt(n0)), T * dt / (2pi/B0)
@show 2 * pi^2 * (vth/B0)^2

function boris(vx, vy, vz, Ex, Ey, B, dt)
  v⁻ = @SArray [vx + Ex * dt/2, vy + Ey * dt/2, vz]
  t = @SArray [B * dt/2, 0, 0]
  v⁺ = v⁻ + 2 * cross(v⁻ + cross(v⁻, t), t) / (1 + dot(t, t))
  return (v⁺[1] + Ex * dt/2, v⁺[2] + Ey * dt/2, v⁺[3])
end

K=zeros(TO,5); Exs=zeros(NX,NY,TO);
Eys=zeros(NX,NY,TO); phis=zeros(NX,NY,TO);
x=rand(P);#halton.(0:P-1, 2, 1/sqrt(2));#rand(P);
y=rand(P);#halton.(0:P-1, 3, 1/sqrt(2));#rand(P);
vx=vth * rand(P);#erfinv.(halton.(0:P-1, 5, 1/sqrt(2)));#rand(P));
vy=vth * rand(P);#erfinv.(halton.(0:P-1, 7, 1/sqrt(2)));#rand(P));
vz=vth * rand(P);#erfinv.(halton.(0:P-1, 11, 1/sqrt(2)));#rand(P));
#th=2pi.*rand(P÷10)
#vx[1:P÷10] .= 0
#vy[1:P÷10] = 4*vth * sin.(th)
#vz[1:P÷10] = 4*vth * cos.(th)
phi=zeros(ComplexF64, NX, NY);
Ex=zeros(ComplexF64,NX, NY);
Ey=zeros(ComplexF64, NX, NY);
ns=zeros(NX, NY, nthreads());
kx=im.*2π*vcat(0:NX/2,-NX/2+1:-1);
ky=im.*2π*vcat(0:NY/2,-NY/2+1:-1);

mymod1(x, n) = 1 <= x <= n ? x : x < 1 ? x + n : x - n
mymod(x, n) = 0 < x <= n ? x : x < 0 ? x + n : x - n #mymod(x + n, n) : mymod(x - n, n)
@inline function g(z, NZ)
  zNZ = z * NZ
  i = mymod1(ceil(Int, zNZ), NZ)
  r = i - zNZ;
  @assert 0 <= r <= 1
  return ((i, 1-r), (mymod1(i+1, NZ), r))# ((i, 0.5), (i, 0.5)) #
end

@inline function eval(F1, F2, xi, yi)
  F1o = zero(eltype(F1))
  F2o = zero(eltype(F2))
  for (j, wy) in g(yi, NY), (i, wx) in g(xi, NX)
    F1o += real(F1[i,j]) * wx * wy
    F2o += real(F2[i,j]) * wx * wy
  end
  return (F1o, F2o)
end

function deposit!(F, x, y, w)
  for (j, wy) in g(y, NY), (i, wx) in g(x, NX)
    F[i,j] += wx * wy * w
  end
end

pfft = plan_fft!(phi; flags=FFTW.ESTIMATE, timelimit=Inf)
pifft = plan_ifft!(phi; flags=FFTW.ESTIMATE, timelimit=Inf)

minvkk = -1 ./ (kx.^2 .+ (ky').^2)
minvkk[1, 1] = 0
@assert all(isfinite, minvkk)

chunks = collect(Iterators.partition(1:P, ceil(Int, P/nthreads())))
@assert maximum(maximum.(chunks)) == P
@assert length(chunks) == size(ns, 3)

@showprogress 1 for t in 1:T;
  hphi0 = @spawn @inbounds @threads for i in eachindex(phi); phi[i] = 0;end
  hparticles = @spawn @threads for j in axes(ns, 3)
    n = @view ns[:, :, j]
    for i in chunks[j]
      Exi, Eyi = eval(Ex, Ey, x[i], y[i])
      vx[i], vy[i], vz[i] = boris(vx[i], vy[i], vz[i], Exi, Eyi, B0, dt);
      x[i] = mymod(x[i] + vx[i]*dt,1)
      y[i] = mymod(y[i] + vy[i]*dt,1)
      @assert 0 < x[i] <= 1
      @assert 0 < y[i] <= 1
      deposit!(n, x[i], y[i], w)
    end
  end
  wait(hphi0)
  wait(hparticles)
  @threads for j in 1:size(phi, 2)
    for k in axes(phi, 3), i in axes(phi, 1)
      phi[i, j] += ns[i, j, k]
    end
  end
  hns0 = @spawn @tturbo @. ns = 0 # faster outside loop above
  pfft * phi;
  @threads for j in axes(phi, 2)
    for i in axes(phi, 1)
      phiij = phi[i, j]
      @assert isfinite(phiij)
      Ex[i, j] = phi[i, j] * kx[i] * minvkk[i, j]
      Ey[i, j] = phi[i, j] * ky[j] * minvkk[i, j]
    end
  end
  hex = @spawn pifft * Ex;
  hey = @spawn pifft * Ey;
  wait(hns0)
  wait(hex)
  wait(hey)
  if t % NS == 0
    ti = (t ÷ NS)
    K[ti,1] = mean(((real.(Ex)).^2 .+ (real.(Ey)).^2))
    K[ti,2] = sum((vx.^2 + vy.^2).*w);
    K[ti,3]=sum(K[ti,1:2]);
    K[ti,4]=sum(vx)/P;
    K[ti,5]=sum(vy)/P;
    Exs[:,:,ti] .= real.(Ex);
    Eys[:,:,ti] .= real.(Ey);
    phis[:,:,ti] .= real.(phi);
  end
end
return NX, NY, vth, n0, B0, T, dt, Exs, Eys, phis, NS, K
end

NX, NY, vth, n0, B0, T, dt, Exs, Eys, phis, NS, K = foo()
if true
x = (1:NX) ./ NX ./ (vth / B0)
y = (1:NY) ./ NY ./ (vth / B0)
t = (1:size(Exs, 3)) .* ((T * dt / size(Eys, 3)) / (2pi/B0))
heatmap(x, t, Eys[:,1,:]')
xlabel!("Space [vth/Omega_c] ");ylabel!("Time [tau_c]")
savefig("AreaElectrostatic2D3V_TX.png")
heatmap(y, t, Eys[1,:,:]')
xlabel!("Space [vth/Omega_c] ");ylabel!("Time [tau_c]")
savefig("AreaElectrostatic2D3V_TY.png")
filter = sin.(((1:size(Eys,3)) .- 0.5) ./ size(Eys,3) .* pi)'
ws = 2π/(T * dt) .* (1:size(Eys,3)) ./ (B0);
kxs = 2π .* (0:NX-1) ./ (B0/vth);
kys = 2π .* (0:NY-1) ./ (B0/vth);
wind = length(ws)÷2#findlast(ws .< max(5.1, 1.1 *sqrt(n0)/B0));

@views for (F, FS) in ((Exs, "Ex"), (Eys, "Ey"), (phis, "phi"))
  heatmap(kxs[2:end÷2-1], ws[1:wind],
    log10.(sum(i->abs.(fft(F[:, i, :])[2:end÷2-1, 1:wind]'), 1:size(F, 2))))
  xlabel!(L"Wavenumber $[\Omega_c / v_{th}]$");
  ylabel!(L"Frequency $[\Omega_c]$")
  savefig("AreaElectrostatic2D3V_$(FS)_WKsumy.png")
  heatmap(kys[2:end÷2-1], ws[1:wind],
    log10.(sum(i->abs.(fft(F[i, :, :])[2:end÷2-1, 1:wind]'), 1:size(F, 1))))
  xlabel!(L"Wavenumber $[\Omega_c / v_{th}]$");
  ylabel!(L"Frequency $[\Omega_c]$")
  savefig("AreaElectrostatic2D3V_$(FS)_WKsumx.png")
end

end
# heatmap(kxs[1:end], ws, log10.(abs.(sum(i->abs.(fft(Exs[:, i, :])), 1:size(Eys, 2))))[1:end÷2-1, 1:end÷2]')
#
# heatmap(kxs[1:end], ws, log10.(abs.(sum(i->abs.(fft(Exs[i, :, :])),
# 1:size(Eys, 1))))[1:end÷2-1, 1:end÷2]')
#
# heatmap(kxs[1:end], ws, log10.(abs.(sum(i->abs.(fft(Eys[i, :, :])),
# 1:size(Eys, 1))))[1:end÷2-1, 1:end÷2]')
#
# heatmap(kxs[1:end], ws, log10.(abs.(sum(i->abs.(fft(Eys[:, i, :])), 1:size(Eys, 2))))[1:end÷2-1, 1:end÷2]')

