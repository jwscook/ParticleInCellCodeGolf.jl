using FFTW,Plots, SpecialFunctions, StaticArrays, LinearAlgebra, Roots
using LoopVectorization, Base.Threads, ThreadsX, Base.Iterators, Statistics
FFTW.set_num_threads(Threads.nthreads())

function foo()

NX=NY=256;P=NX*NY*16;T=2^17;NS=8;TO=T÷NS;NG=sqrt(NX^2 + NY^2)
n0=4*pi^2;vth=sqrt(n0)/NG;dt=1/NG/60vth;B0=sqrt(n0)/16;w=n0/P;
@show NX, NY, P, T, TO, n0, vth, B0, dt
igr(d) = 1 / Roots.find_zero(x->x^(d+1) - x - 1, 1.5)
R(α, N, s0=0) = [rem(s0 + n * α, 1) for n in 1:N]

function boris(vx, vy, vz, Ex, Ey, B, dt)
#  return (vx + Ex * dt/2, vy + Ey * dt/2, vz)
  v⁻ = @SArray [vx + Ex * dt/2, vy + Ey * dt/2, vz]
  t = @SArray [B * dt/2, 0, 0]
  v⁺ = v⁻ + 2 * cross(v⁻ + cross(v⁻, t), t) / (1 + dot(t, t))
  return (v⁺[1] + Ex * dt/2, v⁺[2] + Ey * dt/2, v⁺[3])
end

K=zeros(TO,5); Exs=zeros(NX,NY,TO); Eys=zeros(NX,NY,TO);
x=R(igr(1),P);y=R(igr(2),P);
vx=vth * erfinv.(R(igr(3),P));
vy=vth * erfinv.(R(igr(4),P));
vz=vth * erfinv.(R(igr(5),P));
phi=zeros(ComplexF64, NX, NY);Ex=zeros(ComplexF64,NX, NY);Ey=zeros(ComplexF64, NX, NY);
ns=zeros(NX, NY, nthreads());
kx=im.*2π*vcat(0:NX/2,-NX/2+1:-1);
ky=im.*2π*vcat(0:NY/2,-NY/2+1:-1);

@inline function g(z, NZ)
  zNZ = z * NZ
  i = mod1(ceil(Int, zNZ), NZ)
  r = i - zNZ;
  @assert 0 <= r <= 1
  return ((i, 1-r), (mod1(i+1, NZ), r))# ((i, 0.5), (i, 0.5)) #
end


@inline function eval(F1, F2, xi, yi)
  output = sum((@SArray [real(F1[i,j]), real(F2[i,j])]) .* wx * wy
    for (j, wy) in g(yi, NY), (i, wx) in g(xi, NX))
  if !all(isfinite, output)
    @show xi, yi
    for k in 1:2
      @show k
      i, wx = g(xi, NX)[k]
      j, wy = g(yi, NY)[k]
      @show i, wx
      @show j, wy
      @show F1[i, j]
      @show F2[i, j]
    end
  end
  @assert all(isfinite, output)
  return output
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

for t in 1:T;
  @threads for j in 1:nthreads()
    n = @view ns[:, :, threadid()]
    for i in chunks[j]
      Exi, Eyi = eval(Ex, Ey, x[i], y[i])
      #@show i, x[i], y[i], vx[i], vy[i], vz[i], Exi, Eyi
      vx[i], vy[i], vz[i] = boris(vx[i], vy[i], vz[i], Exi, Eyi, B0, dt);
      x[i] = mod(x[i] + vx[i]*dt,1)
      y[i] = mod(y[i] + vy[i]*dt,1)
      #@show i, x[i], y[i], vx[i], vy[i], vz[i], Exi, Eyi
      deposit!(n, x[i], y[i], w)
    end
  end
  @threads for j in 1:size(phi, 2)
    for k in axes(phi, 3), i in axes(phi, 1)
      phi[i, j] += ns[i, j, k]
    end
  end
  @tturbo @. ns = 0
  pfft * phi;
  @assert all(isfinite, phi)
  @threads for j in axes(phi, 2)
    for i in axes(phi, 1)
      Ex[i, j] = phi[i, j] * kx[i] * minvkk[i, j]
      Ey[i, j] = phi[i, j] * ky[j] * minvkk[i, j]
      phi[i, j] = 0
    end
  end
  hex = @spawn pifft * Ex;
  hey = @spawn pifft * Ey;
  wait(hex)
  wait(hey)
  #@assert all(isfinite, Ey)
  #@assert all(isfinite, Ex)
  if t % NS == 0
    ti = (t ÷ NS)
    K[ti,1] = mean(((real.(Ex)).^2 .+ (real.(Ey)).^2))
    K[ti,2] = sum((vx.^2 + vy.^2).*w);
    @show ti, TO, K[ti, 1:2]
    K[ti,3]=sum(K[ti,1:2]);
    K[ti,4]=sum(vx)/P;
    K[ti,5]=sum(vy)/P;
    Exs[:,:,ti] .= real.(Ex);
    Eys[:,:,ti] .= real.(Ey);
  end
end
return NX, NY, vth, n0, B0, T, dt, Exs, Eys, NS, K
end

NX, NY, vth, n0, B0, T, dt, Exs, Eys, NS, K = foo()
if true
x = (1:NX) ./ NX ./ (vth / B0)
y = (1:NY) ./ NY ./ (vth / B0)
t = (1:size(Exs, 3)) .* ((T * dt / size(Eys, 3)) / (2pi/B0))
heatmap(x, t, Eys[:,1,:]')
xlabel!("Space [vth/Omega_c] ");ylabel!("Time [tau_c]")
savefig("AreaElectrostatic2D3V_TX.png")
heatmap(x, t, Eys[1,:,:]')
xlabel!("Space [vth/Omega_c] ");ylabel!("Time [tau_c]")
savefig("AreaElectrostatic2D3V_TY.png")
filter = sin.(((1:size(Eys,3)) .- 0.5) ./ size(Eys,3) .* pi)'
ws = 2π/(T * dt) .* (1:size(Eys,3)) ./ (B0);
kxs = 2π .* (0:NX-1) ./ (B0/vth);
kys = 2π .* (0:NY-1) ./ (B0/vth);
Zx = log10.(abs.(fft((Eys[:,1,:] .* filter)')))
heatmap(kxs[1:end÷2], ws[1:200], Zx[1:end÷2, 2:200])
xlabel!("Wavenumber");ylabel!("Frequency")
savefig("AreaElectrostatic2D3V_WKx.png")
Zy = log10.(abs.(fft((Eys[:,1,:] .* filter)')))[1:end÷2, 2:end÷2];
heatmap(kys[1:end÷2], ws[1:200], Zy[1:end÷2, 2:200])
xlabel!("Wavenumber");ylabel!("Frequency")
savefig("AreaElectrostatic2D3V_WKy.png")

heatmap(kxs[2:end÷2-1], ws[1:200], log10.(abs.(sum(i->abs.(fft(Eys[:, i, :])), 1:size(Eys, 2))))[2:end÷2-1, 1:200]')

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

