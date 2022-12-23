using FFTW,Plots, SpecialFunctions, StaticArrays, LinearAlgebra, Random
using LoopVectorization, Base.Threads, ThreadsX, Base.Iterators, Statistics
using ProgressMeter, LaTeXStrings, OffsetArrays, StructArrays
doturbo = try; parse(Bool, ARGS[1]);catch; true; end
SOArray(x) = StructArray(OffsetArray(x,zeros(Int, length(size(x)))...))
TurboArray(x) = doturbo ? SOArray(x) : x
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

function foo()
sq = 2 * 3 * 5 * 7 * 11
NX=NY=64;
NP1 = ceil(Int, NX * NY/sq) * sq
P=NX*NY*2^4;T=2^15;NS=16;TO=T÷NS;NG=sqrt(NX^2 + NY^2)
n0=4*pi^2;vth=sqrt(n0)/NG;dt=1/NG/10vth;B0=sqrt(n0)/8;w=n0/P;
Δ=1/NG;Δx=1/NX;Δy=1/NY
@show NX, NY, P, T, TO, NS, n0, vth, B0, dt
@show vth * dt / Δ, vth / B0 / Δy
@show 2pi/sqrt(n0) / (NS * dt), 2pi/B0 / (NS * dt)
@show T * dt / (2pi/sqrt(n0)), T * dt / (2pi/B0)
@show 2 * pi^2 * (vth/B0)^2

t = @SArray [B0 * dt/2, 0, 0]
tscale = 2 / (1 + dot(t, t))
function boris(vx, vy, vz, Ex, Ey, dt)
  v⁻ = @SArray [vx + Ex * dt/2, vy + Ey * dt/2, vz]
  v⁺ = v⁻ + cross(v⁻ + cross(v⁻, t), t) * tscale
  return (v⁺[1] + Ex * dt/2, v⁺[2] + Ey * dt/2, v⁺[3])
end

K=zeros(TO,5); Exs=zeros(NX,NY,TO);
Eys=zeros(NX,NY,TO); phis=zeros(NX,NY,TO);
x  =       rand(P);#        halton.(0:P-1, 2, 1/sqrt(2));#rand(P);
y  =       rand(P);#        halton.(0:P-1, 3, 1/sqrt(2));#rand(P);
vx = vth * rand(P);#erfinv.(halton.(0:P-1, 5, 1/sqrt(2)));#rand(P));
vy = vth * rand(P);#erfinv.(halton.(0:P-1, 7, 1/sqrt(2)));#rand(P));
vz = vth * rand(P);#erfinv.(halton.(0:P-1, 11, 1/sqrt(2)));#rand(P));
#th=2pi.*rand(P÷10)
#vx[1:P÷10] .= 0
#vy[1:P÷10] = 4*vth * sin.(th)
#vz[1:P÷10] = 4*vth * cos.(th)
phi=TurboArray(zeros(ComplexF64, NX, NY));
Ex=TurboArray(zeros(ComplexF64,NX, NY));
Ey=TurboArray(zeros(ComplexF64, NX, NY));
kx=TurboArray(im * 2π*vcat(0:NX/2,-NX/2+1:-1));
ky=TurboArray(im * 2π*vcat(0:NY/2,-NY/2+1:-1));
ns=zeros(NX, NY, nthreads())

minvkk = real.(-1 ./ (kx.^2 .+ (ky').^2))
minvkk[1, 1] = 0
@assert all(isfinite, minvkk)


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
    wxy = wx * wy
    F1o += real(F1[i,j]) * wxy
    F2o += real(F2[i,j]) * wxy
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
@show size(ns, 3)
@show length(chunks)
@assert length(chunks) == size(ns, 3)
t1 = t2 = t3 = t4 = t5 = 0.0
t3b = 0.0

@showprogress 1 for t in 1:T;
  #hphi0 = @spawn if doturbo
  t1 += @elapsed if doturbo
    @tturbo for i in eachindex(phi);phi.re[i]=0.;phi.im[i]=0.;end
  else
    @inbounds @threads for i in eachindex(phi); phi[i] = 0;end
  end
  #hparticles = @spawn @threads for j in axes(ns, 3)
  t2 += @elapsed @threads for j in axes(ns, 3)
    n = @view ns[:, :, j]
    for i in chunks[j]
      Exi, Eyi = eval(Ex, Ey, x[i], y[i])
      vx[i], vy[i], vz[i] = boris(vx[i], vy[i], vz[i], Exi, Eyi, dt);
      x[i] = mymod(x[i] + vx[i]*dt,1)
      y[i] = mymod(y[i] + vy[i]*dt,1)
      @assert 0 < x[i] <= 1
      @assert 0 < y[i] <= 1
      depositcharge!(n, x[i], y[i], w)
    end
  end
  #wait(hphi0)
  #wait(hparticles)
  t3 += @elapsed @threads for j in 1:size(phi, 2)
    for k in axes(ns, 3), i in axes(phi, 1)
      phi[i, j] += ns[i, j, k]
    end
  end
  if !doturbo
    t3b += @elapsed phi .= sum(ns, dims=3)
  end
  #hns0 = @spawn @tturbo @. ns = zero(eltype(ns)) # faster outside loop above
  t4 += @elapsed @tturbo @. ns = 0.0 # faster outside loop above
  #hexy = if doturbo
  t5 += @elapsed hexy = if doturbo
    phi .= pfft * phi;# OffsetArrays don't update in place
    begin
      @tturbo for j in axes(Ex, 2), i in axes(Ex, 1)
        phi.re[i, j] *= minvkk[i, j]
        phi.im[i, j] *= minvkk[i, j]
        Ex.re[i, j] = - phi.im[i, j] * kx.im[i]
        Ex.im[i, j] =   phi.re[i, j] * kx.im[i]
      end
      @tturbo for j in axes(Ex, 2)
        kyimj = ky.im[j]
        for i in axes(Ex, 1)
          Ey.re[i, j] = - phi.im[i, j] * kyimj
          Ey.im[i, j] =   phi.re[i, j] * kyimj
        end
      end
    end
    #@spawn ((Ex .= pifft * Ex), (Ey .= pifft * Ey))
    ((Ex .= pifft * Ex), (Ey .= pifft * Ey))
  else
    pfft * phi;
    @threads for j in axes(phi, 2)
      for i in axes(phi, 1)
        phi[i, j] = phi[i, j] * minvkk[i, j]
        Ex[i, j] = phi[i,j] * kx[i] * im
        Ey[i, j] = phi[i,j] * ky[j] * im
      end
    end
    #@spawn ((pifft * Ex), (pifft * Ey))
    ((pifft * Ex), (pifft * Ey))
  end
  #wait(hns0)
  #wait(hexy)
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
@show t1
@show t2
@show t3
@show t3b
@show t4
@show t5
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

