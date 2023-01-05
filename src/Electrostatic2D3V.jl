using FFTW,Plots, SpecialFunctions, StaticArrays, LinearAlgebra, Random
using LoopVectorization, Base.Threads, ThreadsX, Base.Iterators, Statistics
using ProgressMeter, LaTeXStrings, MuladdMacro, CommonSubexpressions
using TimerOutputs

FFTW.set_num_threads(Threads.nthreads())
Plots.gr()
Random.seed!(0)
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
#sq = 2 * 3 * 5 * 7 * 11
#NP1 = ceil(Int, NX * NY/sq) * sq
to = TimerOutput()
NX=NY=128;P=NX*NY*2^5;T=2^13;NG=sqrt(NX^2 + NY^2)
n0=4*pi^2;vth=sqrt(n0)/NG;dt=1/NG/6vth;B0=sqrt(n0)/4;
NS=2;TO=T÷NS;Δx=1/NX;Δy=1/NY;w=n0/P/(Δx*Δy);
@show NX, NY, P, T, TO, NS, n0, vth, B0, dt, w
@show vth * dt / Δx, vth / B0 / Δy
@show 2pi/sqrt(n0) / (NS * dt), 2pi/B0 / (NS * dt)
@show T * dt / (2pi/sqrt(n0)), T * dt / (2pi/B0)
@show 2 * pi^2 * (vth/B0)^2

tvec = @SArray [B0 * dt/2, 0, 0]
tscale = 2 / (1 + dot(tvec, tvec))

function boris(vx, vy, vz, Ex, Ey, dt)
  dt_2 = dt/2
  Edt_2 = @SArray [Ex * dt_2, Ey * dt_2, 0.0]
  v⁻ = (@SArray [vx, vy, vz]) + Edt_2
  v⁺ = v⁻ + cross(v⁻ + cross(v⁻, tvec), tvec) * tscale
  return v⁺ + Edt_2
end

K=zeros(TO,5); Exs=zeros(NX,NY,TO);
Eys=zeros(NX,NY,TO); phis=zeros(NX,NY,TO);
x  = rand(P)      ;#        halton.(0:P-1, 2, 1/sqrt(2));#rand(P);
y  = rand(P)      ;#        halton.(0:P-1, 3, 1/sqrt(2));#rand(P);
vx = erfinv.(2rand(P) .- 1) * vth;#erfinv.(halton.(0:P-1, 5, 1/sqrt(2)));#rand(P));
vy = erfinv.(2rand(P) .- 1) * vth;#erfinv.(halton.(0:P-1, 7, 1/sqrt(2)));#rand(P));
vz = erfinv.(2rand(P) .- 1) * vth;#erfinv.(halton.(0:P-1, 11, 1/sqrt(2)));#rand(P));
vx .-= mean(vx)
vy .-= mean(vy)
vz .-= mean(vz)
vx .*= (vth / sqrt(2)) / std(vx);
vy .*= (vth / sqrt(2)) / std(vy);
vz .*= (vth / sqrt(2)) / std(vz);
p  = collect(1:P)
function sortparticles!()
  sortperm!(p, [(x[i], y[i]) for i in 1:P],
    by=x->(ceil(Int, x[1]*NX), ceil(Int, y[2]*NY)))
  x .= x[p]; y .= y[p]; vx .= vx[p]; vy .= vy[p]; vz .= vz[p];
  return nothing
end
#th=2pi.*rand(P÷10)
#vx[1:P÷10] .= 0
#vy[1:P÷10] = 4*vth * sin.(th)
#vz[1:P÷10] = 4*vth * cos.(th)
phi=zeros(ComplexF64, NX, NY);
Ex=zeros(ComplexF64,NX, NY);
Ey=zeros(ComplexF64, NX, NY);
kx=2π*vcat(0:NX÷2-1,-NX÷2:-1);
ky=2π*vcat(0:NY÷2-1,-NY÷2:-1);
ns=zeros(NX, NY, nthreads())
#E = -∇ ϕ
#∇ . E = -∇.∇ ϕ = -∇^2 ϕ = ρ
#-i^2 (kx^2 + ky^2) ϕ = ρ
#ϕ = ρ / (kx^2 + ky^2)
# Ex = - ∇_x ϕ = - i kx ϕ = - i kx ρ / (kx^2 + ky^2)
# Ey = - ∇_y ϕ = - i ky ϕ = - i ky ρ / (kx^2 + ky^2)
minvkk = -im ./ (kx.^2 .+ (ky').^2) # - i / (kx^2 + ky^2)
minvkk[1, 1] = 0
@assert all(isfinite, minvkk)

unimod(x, n) = 0 < x <= n ? x : x > n ? x - n : x + n
@inline function g(z, NZ)
  zNZ = z * NZ
  i = unimod(ceil(Int, zNZ), NZ)
  r = i - zNZ;
  @assert 0 < r <= 1
  return ((i, 1-r), (unimod(i+1, NZ), r))
  #return ((i, 0.5), (i, 0.5))
  #return ((i, 1),)
end

@inline function eval(F1, F2, xi, yi)
  F1o = zero(real(eltype(F1)))
  F2o = zero(real(eltype(F2)))
  for (j, wy) in g(yi, NY), (i, wx) in g(xi, NX)
    wxy = wx * wy
    @muladd F1o += real(F1[i,j]) * wxy
    @muladd F2o += real(F2[i,j]) * wxy
  end
  return (F1o, F2o)
end

function depositcharge!(F, x, y, w)
  for (j, wy) in g(y, NY), (i, wx) in g(x, NX)
    F[i,j] += wx * wy * w
  end
end

pfft = plan_fft!(phi; flags=FFTW.ESTIMATE, timelimit=Inf)
pifft = plan_ifft!(phi; flags=FFTW.ESTIMATE, timelimit=Inf)

chunks = collect(Iterators.partition(1:P, ceil(Int, P/nthreads())))
@assert maximum(maximum.(chunks)) == P
@assert length(chunks) == size(ns, 3)
t1 = t2 = t3 = t4 = t5 = t6 = 0.0
@show w

@showprogress 1 for t in 1:T;
  @timeit to "zero" begin
    @inbounds @threads for i in eachindex(phi); phi[i] = 0;end
    @tturbo @. ns = 0.0 # faster outside loop above
  end
  @timeit to "Particle loop" begin
    @threads for j in axes(ns, 3)
      n = @view ns[:, :, j]
      for i in chunks[j]
        Exi, Eyi = eval(Ex, Ey, x[i], y[i])
        vx[i], vy[i], vz[i] = boris(vx[i], vy[i], vz[i], Exi, Eyi, dt);
        x[i] = unimod(x[i] + vx[i]*dt,1)
        y[i] = unimod(y[i] + vy[i]*dt,1)
        @assert 0 < x[i] <= 1
        @assert 0 < y[i] <= 1
        depositcharge!(n, x[i], y[i], w)
      end
    end
  end
  @timeit to "reduction" begin
    phi .= sum(ns, dims=3)
  end
  @timeit to "field invert" begin
    pfft * phi;
    phi[1, 1] = 0
    @threads for j in axes(phi, 2)
      for i in axes(phi, 1)
        tmp = phi[i, j] * minvkk[i, j]
        @assert isfinite(tmp)
        Ex[i, j] = tmp * kx[i]
        Ey[i, j] = tmp * ky[j]
      end
    end
  end
  @timeit to "field solve" begin
    pifft * Ex
    pifft * Ey
  end

#  @timeit to "particle sort" begin
#    t % 128 == 0 &&  sortparticles!()
#  end

  @timeit to "reduction" begin
    if t % NS == 0
      ti = (t ÷ NS)
      K[ti,1] = mean(((real.(Ex)).^2 .+ (real.(Ey)).^2))
      K[ti,2] = sum((vx.^2 + vy.^2).*w);
      K[ti,3] = sum(K[ti,1:2]);
      K[ti,4] = sum(vx)/P;
      K[ti,5] = sum(vy)/P;
      Exs[:,:,ti] .= real.(Ex);
      Eys[:,:,ti] .= real.(Ey);
      phis[:,:,ti] .= real.(pifft * phi);
    end
  end
end
show(to)
return NX, NY, vth, n0, B0, T, dt, Exs, Eys, phis, NS, K
end

NX, NY, vth, n0, B0, T, dt, Exs, Eys, phis, NS, K = pic()
if true
xs = (1:NX) ./ NX ./ (vth / B0)
ys = (1:NY) ./ NY ./ (vth / B0)
ts = (1:size(Exs, 3)) .* ((T * dt / size(Eys, 3)) / (2pi/B0))
heatmap(xs, ts, Eys[:,1,:]')
xlabel!("Space [vth/Omega_c] ");ylabel!("Time [tau_c]")
savefig("AreaElectrostatic2D3V_TX.png")
heatmap(ys, ts, Eys[1,:,:]')
xlabel!("Space [vth/Omega_c] ");ylabel!("Time [tau_c]")
savefig("AreaElectrostatic2D3V_TY.png")
filter = sin.(((1:size(Eys,3)) .- 0.5) ./ size(Eys,3) .* pi)'
ws = 2π/(T * dt) .* (1:size(Eys,3)) ./ (B0);
kxs = 2π .* (0:NX÷2) ./ (B0/vth);
kys = 2π .* (0:NY÷2) ./ (B0/vth);
wind = findlast(ws .< max(5.1, 2 * sqrt(n0)/B0));

@views for (F, FS) in ((Exs, "Ex"), (Eys, "Ey"), (phis, "phi"))
  if false
  maxabsF = maximum(abs, F)
    nsx = ceil(Int, size(F,1) / 128)
    nsy = ceil(Int, size(F,2) / 128)
    anim = @animate for i in axes(F, 3)
      heatmap(xs[1:nsx:end], ys[1:nsy:end], F[1:nsx:end, 1:nsy:end, i] ./ maxabsF)
      xlabel!(L"Position x $[v_{th} / \Omega]$");
      ylabel!(L"Position y $[v_{th} / \Omega]$")
    end
    gif(anim, "AreaElectrostatic2D3V_$(FS)_XY.gif", fps=10)
  end

  heatmap(xs, ys, F[:, :, 1])
  xlabel!(L"Position x $[v_{th} / \Omega]$");
  ylabel!(L"Position y $[v_{th} / \Omega]$")
  savefig("AreaElectrostatic2D3V_$(FS)_XY_1.png")
  heatmap(xs, ys, F[:, :, end])
  xlabel!(L"Position x $[v_{th} / \Omega]$");
  ylabel!(L"Position y $[v_{th} / \Omega]$")
  savefig("AreaElectrostatic2D3V_$(FS)_XY_end.png")

  Z = log10.(sum(i->abs.(fft(F[:, i, :])[2:end÷2, 1:wind]), 1:size(F, 2)))'
  heatmap(kxs[2:end-1], ws[1:wind], Z)
  xlabel!(L"Wavenumber x $[\Omega_c / v_{th}]$");
  ylabel!(L"Frequency $[\Omega_c]$")
  savefig("AreaElectrostatic2D3V_$(FS)_WKsumy_c.png")
  xlabel!(L"Wavenumber x $[\Pi / v_{th}]$");
  ylabel!(L"Frequency $[\Pi]$")
  heatmap(kxs[2:end-1] .* B0 / sqrt(n0), ws[1:wind] .* B0 / sqrt(n0), Z)
  savefig("AreaElectrostatic2D3V_$(FS)_WKsumy_p.png")

  Z = log10.(sum(i->abs.(fft(F[i, :, :])[2:end÷2, 1:wind]), 1:size(F, 1)))'
  heatmap(kys[2:end-1], ws[1:wind], Z)
  xlabel!(L"Wavenumber y $[\Omega_c / v_{th}]$");
  ylabel!(L"Frequency $[\Omega_c]$")
  savefig("AreaElectrostatic2D3V_$(FS)_WKsumx_c.png")
  heatmap(kys[2:end-1] .* B0 / sqrt(n0), ws[1:wind] .* B0 / sqrt(n0), Z)
  xlabel!(L"Wavenumber y $[\Pi / v_{th}]$");
  ylabel!(L"Frequency $[\Pi]$")
  savefig("AreaElectrostatic2D3V_$(FS)_WKsumx_p.png")
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

