module PIC2D3V

using FFTW,Plots, SpecialFunctions, StaticArrays, LinearAlgebra, Random
using LoopVectorization, Base.Threads, ThreadsX, Base.Iterators, Statistics
using ProgressMeter, LaTeXStrings, MuladdMacro, CommonSubexpressions
using TimerOutputs, StaticNumbers, OffsetArrays

unimod(x, n) = x > n ? x - n : x > 0 ? x : x + n

function applyperiodicity!(a, oa::OffsetArray)
  for j in axes(oa, 2), i in axes(oa, 1)
    a[unimod(i, NX), unimod(j, NY)] .+= oa[i, j]
  end
end

function applyperiodicity!(oa::OffsetArray, a)
  for j in axes(OA, 2), i in axes(OA, 1)
     oa[i, j] = a[unimod(i, NX), unimod(j, NY)]
  end
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


struct ElectrostaticBoris
  t::SVector{3, Float64}
  t²::Float64
  dt_2::Float64
end
function ElectrostaticBoris(B::AbstractVector, dt::Float64)
  t = (@SArray [B[1], B[2], B[3]]) * dt / 2
  t² = dot(t, t)
  return ElectrostaticBoris(t, t², dt / 2)
end
function (boris::ElectrostaticBoris)(vx, vy, vz, Ex, Ey, q_m)
  Ē₂ = (@SArray [Ex, Ey, 0.0]) * boris.dt_2 * q_m
  v⁻ = (@SArray [vx, vy, vz]) + Ē₂
  v⁺ = v⁻ + cross(v⁻ + cross(v⁻, boris.t), boris.t) * q_m^2 * 2 / (1 + q_m^2 * boris.t²)
  return v⁺ + Ē₂
end

struct ElectromagneticBoris
  dt_2::Float64
  ElectromagneticBoris(dt::Float64) = new(dt / 2)
end

function (boris::ElectromagneticBoris)(vx, vy, vz, Ex, Ey, Ez, Bx, By, Bz, q_m)
  θ = boris.dt_2 * q_m
  t = (@SArray [Bx, By, Bz]) * θ
  tscale = 2 / (1 + dot(t, t))
  Ē₂ = (@SArray [Ex, Ey, Ez]) * θ
  v⁻ = (@SArray [vx, vy, vz]) + Ē₂
  v⁺ = v⁻ + cross(v⁻ + cross(v⁻, t), t) * tscale
  return v⁺ + Ē₂
end


abstract type AbstractDiagnostics end

struct ElectrostaticDiagnostics <: AbstractDiagnostics
  kineticenergy::Vector{Float64}
  fieldenergy::Vector{Float64}
  particlemomentum::Vector{Vector{Float64}}
  Exs::Array{Float64, 3}
  Eys::Array{Float64, 3}
  ϕs::Array{Float64, 3}
  ntskip::Int
  ngskip::Int
  ti::Ref{Int64}
  makegifs::Bool
end

function generatestorage(NX, NY, ND, nscalar, nmomentum, nstorage)
  scalarstorage = (zeros(ND) for _ in 1:nscalar)
  momentumstorage = ([zeros(3) for _ in 1:ND] for _ in 1:nmomentum)
  fieldstorage = (zeros(NX, NY, ND) for _ in 1:nstorage)
  return (scalarstorage, momentumstorage, fieldstorage)
end

function ElectrostaticDiagnostics(NX, NY, NT, ntskip, ngskip=1; makegifs=false)
  @assert NT >= ntskip
  @assert ispow2(ngskip)
  scalarstorage, momentumstorage, fieldstorage = generatestorage(
    NX÷ngskip, NY÷ngskip, NT÷ntskip, 2, 1, 3)
  return ElectrostaticDiagnostics(scalarstorage..., momentumstorage...,
    fieldstorage..., ntskip, ngskip, Ref(0), makegifs)
end

struct LorenzGaugeDiagnostics <: AbstractDiagnostics
  kineticenergy::Array{Float64, 1}
  fieldenergy::Array{Float64, 1}
  particlemomentum::Vector{Vector{Float64}}
  fieldmomentum::Vector{Vector{Float64}}
  Exs::Array{Float64, 3}
  Eys::Array{Float64, 3}
  Ezs::Array{Float64, 3}
  Bxs::Array{Float64, 3}
  Bys::Array{Float64, 3}
  Bzs::Array{Float64, 3}
  Axs::Array{Float64, 3}
  Ays::Array{Float64, 3}
  Azs::Array{Float64, 3}
  ϕs::Array{Float64, 3}
  ρs::Array{Float64, 3}
  Jxs::Array{Float64, 3}
  Jys::Array{Float64, 3}
  Jzs::Array{Float64, 3}
  ntskip::Int
  ngskip::Int
  ti::Ref{Int64}
  makegifs::Bool
end

function LorenzGaugeDiagnostics(NX, NY, NT::Int, ntskip::Int, ngskip=1;
                                makegifs=false)
  @assert NT >= ntskip
  @assert ispow2(ngskip)
  scalarstorage, momentumstorage, fieldstorage = generatestorage(
    NX÷ngskip, NY÷ngskip, NT÷ntskip, 2, 2, 14)
  return LorenzGaugeDiagnostics(scalarstorage..., momentumstorage...,
    fieldstorage..., ntskip, ngskip, Ref(0), makegifs)
end


abstract type AbstractShape end
struct NGPWeighting <: AbstractShape end
struct AreaWeighting <: AbstractShape end
struct BSpline2Weighting <: AbstractShape end
struct BSplineWeighting{N} <: AbstractShape end


struct Species{S<:AbstractShape}
  charge::Float64
  mass::Float64
  weight::Float64
  shape::S
  xyv::Matrix{Float64}
  p::Vector{Int}
  chunks::Vector{UnitRange{Int}}
end
positions(s::Species) = (@view s.xyv[1:2, :])
velocities(s::Species) = (@view s.xyv[3:5, :])
xyvchunk(s::Species, i::Int) = @view s.xyv[:, s.chunks[i]]

kineticenergy(s::Species) = sum(abs2, velocities(s)) * s.mass / 2 * s.weight

momentum(s::Species) = sum(velocities(s), dims=2)[:] * s.mass * s.weight

calculateweight(n0, P) = n0 / P;

function Species(P, vth, density, shape::AbstractShape; Lx, Ly,
    charge=1, mass=1)
  x  = Lx * halton.(0:P-1, 2, 1/sqrt(2));#rand(P);
  y  = Ly * halton.(0:P-1, 3, 1/sqrt(2));#rand(P);
  vx = vth * erfinv.(2halton.(0:P-1,  5, 1/sqrt(2)) .- 1);#rand(P));erfinv.(2rand(P) .- 1) * vth;#
  vy = vth * erfinv.(2halton.(0:P-1,  7, 1/sqrt(2)) .- 1);#rand(P));erfinv.(2rand(P) .- 1) * vth;#
  vz = vth * erfinv.(2halton.(0:P-1, 11, 1/sqrt(2)) .- 1);#rand(P))erfinv.(2rand(P) .- 1) * vth;#;
  vx .-= mean(vx)
  vy .-= mean(vy)
  vz .-= mean(vz)
  vx .*= (vth / sqrt(2)) / std(vx);
  vy .*= (vth / sqrt(2)) / std(vy);
  vz .*= (vth / sqrt(2)) / std(vz);
  p  = collect(1:P)
  xyv = Matrix(hcat(x, y, vx, vy, vz)')
  chunks = collect(Iterators.partition(1:P, ceil(Int, P/nthreads())))
  weight = calculateweight(density, P)
  return Species(Float64(charge), Float64(mass), weight, shape, xyv, p, chunks)
end

function Base.sort!(s::Species, Δx, Δy)
  sortperm!(s.p, eachindex(s.p),
    by=i->(ceil(Int, s.xyv[1,i] / Δx), ceil(Int, s.xyv[2,i] / Δy), s.xyv[3,i]))
  s.xyv .= s.xyv[:, s.p]
  return nothing
end


abstract type AbstractField end
timestep(f::AbstractField) = f.boris.dt_2 * 2

struct GridParameters
  Lx::Float64
  Ly::Float64
  NX::Int
  NY::Int
  ΔX::Float64
  ΔY::Float64
  NX_Lx::Float64
  NY_Ly::Float64
  xs::Vector{Float64}
  ys::LinearAlgebra.Adjoint{Float64, Vector{Float64}}
end
function GridParameters(Lx, Ly, NX, NY)
  xs = collect(Lx .* (0.5:NX));
  ys = collect(Ly .* (0.5:NY))';
  return GridParameters(Lx, Ly, NX, NY, Lx/NX, Ly/NY, NX / Lx, NY / Ly, xs, ys)
end

cellvolume(g::GridParameters) = g.ΔX * g.ΔY

struct FFTHelper{T, U, V}
  kx::Vector{Float64}
  ky::LinearAlgebra.Adjoint{Float64, Vector{Float64}}
  k²::Matrix{Float64}
  im_k⁻²::Matrix{ComplexF64}
  pfft!::T
  pifft!::U
  pifft::V
end
function FFTHelper(NX, NY, Lx, Ly)
  kx = 2π / Lx * vcat(0:NX÷2-1, -NX÷2:-1);
  ky = 2π / Ly * vcat(0:NY÷2-1, -NY÷2:-1)';
  k² = (kx.^2 .+ ky.^2)
  im_k⁻² = -im ./ k²
  im_k⁻²[1, 1] = 0
  z = zeros(ComplexF64, NX, NY)
  pfft! = plan_fft!(z; flags=FFTW.ESTIMATE, timelimit=Inf)
  pifft! = plan_ifft!(z; flags=FFTW.ESTIMATE, timelimit=Inf)
  pifft = plan_ifft(z; flags=FFTW.ESTIMATE, timelimit=Inf)
  return FFTHelper(kx, ky, k², im_k⁻², pfft!, pifft!, pifft)
end


struct ElectrostaticField{T} <: AbstractField
  ρs::Array{Float64, 3} # offset array
  ϕ::Array{ComplexF64, 2}
  Ex::Array{ComplexF64, 2}
  Ey::Array{ComplexF64, 2}
  Exy::Array{Float64, 3} # offset array
  B0::NTuple{3, Float64}
  gridparams::GridParameters
  ffthelper::T
  boris::ElectrostaticBoris
end

function ElectrostaticField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0)
  ρs=zeros(NX, NY, nthreads())
  Exy=zeros(2, NX, NY);
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  gps = GridParameters(Lx, Ly, NX, NY)
  boris = ElectrostaticBoris([B0x, B0y, B0z], dt)
  return ElectrostaticField(ρs, (zeros(ComplexF64, NX, NY) for _ in 1:3)...,
    Exy, Float64.((B0x, B0y, B0z)), gps, ffthelper, boris)
end

function update!(f::ElectrostaticField)
  # expand offset arrays, applyperiodicity!(oa, a)
  @. f.Exy[1, :, :] = real(f.Ex)
  @. f.Exy[2, :, :] = real(f.Ey)
end

abstract type AbstractImEx end
struct Explicit <: AbstractImEx end
struct Implicit <: AbstractImEx end
struct ImEx <: AbstractImEx
  θ::Float64
end


struct LorenzGaugeField{T, U} <: AbstractField
  imex::T
  ρJs::Array{Float64, 4} # would need to be offset array
  ϕ::Array{ComplexF64, 2}
  ϕ⁻::Array{ComplexF64, 2}
  Ax::Array{ComplexF64, 2}
  Ay::Array{ComplexF64, 2}
  Az::Array{ComplexF64, 2}
  Ax⁻::Array{ComplexF64, 2}
  Ay⁻::Array{ComplexF64, 2}
  Az⁻::Array{ComplexF64, 2}
  Ex::Array{ComplexF64, 2}
  Ey::Array{ComplexF64, 2}
  Ez::Array{ComplexF64, 2}
  ρ::Array{ComplexF64, 2}
  Jx::Array{ComplexF64, 2}
  Jy::Array{ComplexF64, 2}
  Jz::Array{ComplexF64, 2}
  Bx::Array{ComplexF64, 2}
  By::Array{ComplexF64, 2}
  Bz::Array{ComplexF64, 2}
  EBxyz::Array{Float64, 3} # would need to be offset array
  B0::NTuple{3, Float64}
  gridparams::GridParameters
  ffthelper::U
  boris::ElectromagneticBoris
  dt::Float64
end

function LorenzGaugeField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0,
    imex::AbstractImEx=Explicit())
  EBxyz=zeros(6, NX, NY);
  gps = GridParameters(Lx, Ly, NX, NY)
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  boris = ElectromagneticBoris(dt)
  return LorenzGaugeField(imex, zeros(4, NX, NY, nthreads()),
    (zeros(ComplexF64, NX, NY) for _ in 1:18)..., EBxyz,
    Float64.((B0x, B0y, B0z)), gps, ffthelper, boris, dt)
end

theta(::Explicit) = 0
theta(imex::ImEx) = imex.θ
theta(::Implicit) = 1

function update!(f::LorenzGaugeField)
  # expand offset arrays, applyperiodicity!(oa, a)
  f.EBxyz[1, :, :] .= real.(f.Ex)
  f.EBxyz[2, :, :] .= real.(f.Ey)
  f.EBxyz[3, :, :] .= real.(f.Ez)
  f.EBxyz[4, :, :] .= real.(f.Bx) .+ f.B0[1]
  f.EBxyz[5, :, :] .= real.(f.By) .+ f.B0[2]
  f.EBxyz[6, :, :] .= real.(f.Bz) .+ f.B0[3]
end

function reduction!(a, z)
  @. a = 0.0
  for k in axes(z, 3), j in axes(a, 2), i in axes(a, 1)
    a[i, j] += z[i, j, k]
    z[i, j, k] = 0.0
  end
end

function reduction!(a, b, c, d, z)
  @assert size(z, 1) == 4
  @. a = 0.0
  @. b = 0.0
  @. c = 0.0
  @. d = 0.0
  for k in axes(z, 4), j in axes(a, 2), i in axes(a, 1)
    a[i, j] += z[1, i, j, k]
    b[i, j] += z[2, i, j, k]
    c[i, j] += z[3, i, j, k]
    d[i, j] += z[4, i, j, k]
    for h in axes(z, 1)
      z[h, i, j, k] = 0.0
    end
  end
end

# E = -∇ ϕ
# ∇ . E = -∇.∇ ϕ = -∇^2 ϕ = ρ
# -i^2 (kx^2 + ky^2) ϕ = ρ
# ϕ = ρ / (kx^2 + ky^2)
# Ex = - ∇_x ϕ = - i kx ϕ = - i kx ρ / (kx^2 + ky^2)
# Ey = - ∇_y ϕ = - i ky ϕ = - i ky ρ / (kx^2 + ky^2)
function loop!(plasma, field::ElectrostaticField, to, t)
  dt = timestep(field)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  ΔV = cellvolume(field.gridparams)
  @timeit to "Particle Loop" begin
    @threads for k in axes(field.ρs, 3)
      ρ = @view field.ρs[:, :, k]
      for species in plasma
        qw_ΔV = species.charge * species.weight / ΔV
        q_m = species.charge / species.mass
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        for i in species.chunks[k]
          Exi, Eyi = field(species.shape, x[i], y[i])
          vxi, vyi = vx[i], vy[i]
          vx[i], vy[i], vz[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, q_m);
          x[i] = unimod(x[i] + (vxi + vx[i])/2*dt, Lx)
          y[i] = unimod(y[i] + (vyi + vy[i])/2*dt, Ly)
          deposit!(ρ, species.shape, x[i], y[i], NX_Lx, NY_Ly, qw_ΔV)
        end
      end
    end
  end

  @timeit to "Field Reduction" begin
    reduction!(field.ϕ, field.ρs)
  end
  @timeit to "Field Forward FT" begin
    field.ffthelper.pfft! * field.ϕ;
    field.ϕ[1, 1] = 0
  end
  @timeit to "Field Solve" begin
    @threads for j in axes(field.ϕ, 2)
      for i in axes(field.ϕ, 1)
        tmp = field.ϕ[i, j] * field.ffthelper.im_k⁻²[i, j]
        @assert isfinite(tmp)
        field.Ex[i, j] = tmp * field.ffthelper.kx[i]
        field.Ey[i, j] = tmp * field.ffthelper.ky[j]
      end
    end
  end
  @timeit to "Field Inverse FT" begin
    field.ffthelper.pifft! * field.Ex
    field.ffthelper.pifft! * field.Ey
  end
  @timeit to "Field Update" update!(field)
end

@inline denominator(::Explicit, dt², k²) = 1
@inline denominator(::Implicit, dt², k²) = 1 + dt² * k² / 2
@inline denominator(imex::ImEx, dt², k²) = 1 + dt² * k² * theta(imex) / 2
@inline numerator(::Explicit, dt², k²) = 2 - dt² * k²
@inline numerator(::Implicit, dt², k²) = 2
@inline numerator(imex::ImEx, dt², k²) = 2 - dt² * k² * (1 - theta(imex))
# ∇^2 f - ∂ₜ² f = - s
# -k² f - (f⁺ - 2f⁰ + f⁻)/Δt^2 = - s
# Explicit f⁺ = (2 - k²Δt^2)f⁰ - f⁻ + Δt^2 s
function lorenzgauge!(imex::AbstractImEx, xⁿ, xⁿ⁻¹, sⁿ, k², dt²)
  @threads for i in eachindex(xⁿ)
    num = numerator(imex, dt², k²[i])
    den = denominator(imex, dt², k²[i])
    xⁿ⁺¹ = (num * xⁿ[i] + dt² * sⁿ[i]) / den - xⁿ⁻¹[i]
    xⁿ⁻¹[i] = xⁿ[i]
    xⁿ[i] = xⁿ⁺¹
  end
end

# E = -∇ϕ - ∂ₜA
# B = ∇xA
# ∇² A - 1/c^2 ∂ₜ² A = -μ₀ J⁰
# ∇² ϕ - 1/c^2 ∂ₜ² ϕ = -ρ⁰ / ϵ₀
# im^2 k^2 * ϕ - 1/dt^2/c^2 (ϕ⁺ - 2ϕ⁰ + ϕ⁻) =  -ρ / ϵ₀
# 1) Calculate ϕ⁺ and A⁺ (ϕ and A at next timestep, n+1)
# ϕ⁺ = 2ϕ⁰ - ϕ⁻ + (ρ / ϵ₀ - k^2 * ϕ⁰)*c^2*dt^2
# A⁺ = 2A⁰ - A⁻ + (J μ₀ - k^2 * A⁰)*c^2*dt^2
# 2) calculate the half timstep n+1/2
# Eʰ = -∇(ϕ⁰ + ϕ⁺) / 2 - (A⁺ - A⁰)/dt
# Bʰ = ∇x(A⁺ + A⁰)/2
# 3) push particles from n to n+1 with fields at n+1/2
# push x⁰ to xʰ with v⁰
# push v⁰ to v⁺ with Eʰ and Bʰ
# push xʰ to x⁺ with v⁺
# 4) copy fields into buffers for next loop
function loop!(plasma, field::LorenzGaugeField, to, t)
  dt = timestep(field)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  ΔV = cellvolume(field.gridparams)
  @timeit to "Particle Loop" begin
    @threads for j in axes(field.ρJs, 4)
      ρJ = @view field.ρJs[:, :, :, j]
      for species in plasma
        qw_ΔV = species.charge * species.weight / ΔV
        q_m = species.charge / species.mass
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        for i in species.chunks[j]
          # x and v are known at the nth timestep, E and B fields are (n+1/2)th
          Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(species.shape, x[i], y[i])
          vxi = vx[i] # store for later use in v at (n+1/2)
          vyi = vy[i] # store for later use in v at (n+1/2)
          # accelerate velocities from n to (n+1)
          vx[i], vy[i], vz[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, Ezi,
            Bxi, Byi, Bzi, q_m);
          # spatial advection, effectively using v at (n+1/2)
          x[i] = unimod(x[i] + (vxi + vx[i]) / 2 * dt, Lx)
          y[i] = unimod(y[i] + (vyi + vy[i]) / 2 * dt, Ly)
          # now deposit at (n+1)th timestep
          deposit!(ρJ, species.shape, x[i], y[i], NX_Lx, NY_Ly,
                   qw_ΔV, vx[i] * qw_ΔV, vy[i] * qw_ΔV, vz[i] * qw_ΔV)
        end
      end
    end
  end
  @timeit to "Field Reduction" begin
    reduction!(field.ρ, field.Jx, field.Jy, field.Jz, field.ρJs)
   # mρ = mean(field.ρ)
   # t > 0 && (field.ρ .= 0.0)
   # if t == 0
   # #  for i in eachindex(field.ρ)
   # #    field.ρ[i] = (10  +  (rand() - 0.5))
   # #  end
   # field.ρ .= 0.0
   # #field.ρ[1, 1] = 1.0
   # field.ϕ⁻[1, 1] = 1.0 # one pixel
   # field.ffthelper.pfft! * field.ϕ⁻
   # end
   # field.Jx .= 0.0
   # field.Jy .= 0.0
   # field.Jz .= 0.0
  end
  @timeit to "Field Forward FT" begin
    field.ffthelper.pfft! * field.ρ;
    field.ffthelper.pfft! * field.Jx;
    field.ffthelper.pfft! * field.Jy;
    field.ffthelper.pfft! * field.Jz;
  end
  @timeit to "Field Solve" begin
    field.ρ[1, 1] = 0
    field.Jx[1, 1] = 0
    field.Jy[1, 1] = 0
    field.Jz[1, 1] = 0
    #if t == 0
    #  field.ϕ⁻ .= 0.0
    #  field.ϕ .= 0.0
    #  field.ϕ⁻[end÷2, end÷2] = 1.0
    #  #field.ϕ[end÷2, end÷2] = 1.0
    #  #field.ffthelper.pfft! * field.ϕ⁻
    #  #field.ffthelper.pfft! * field.ϕ
    #  #field.ϕ⁻[1, 1] *= 0.0
    #  #field.ϕ⁻[10, 10] = 1.0
    #  #field.ϕ[10, 10] = 1.0
    #end
    #if t == 0
    #  # warmup
    #  for η in 10.0.^(-16:-1)
    #    lorenzgauge!(field.imex, field.ϕ, field.ϕ⁻, η .* field.ρ, field.ffthelper.k², η * dt^2)
    #    lorenzgauge!(field.imex, field.Ax, field.Ax⁻, η .* field.Jx, field.ffthelper.k², η * dt^2)
    #    lorenzgauge!(field.imex, field.Ay, field.Ay⁻, η .* field.Jy, field.ffthelper.k², η * dt^2)
    #    lorenzgauge!(field.imex, field.Az, field.Az⁻, η .* field.Jz, field.ffthelper.k², η * dt^2)
    #  end
    #end
    # at this point ϕ stores the nth timestep value and ϕ⁻ the (n-1)th
    lorenzgauge!(field.imex, field.ϕ, field.ϕ⁻, field.ρ, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ax, field.Ax⁻, field.Jx, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ay, field.Ay⁻, field.Jy, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Az, field.Az⁻, field.Jz, field.ffthelper.k², dt^2)
    # at this point (ϕ, Ai) stores the (n+1)th timestep value and (ϕ⁻, Ai⁻) the nth
    # Now calculate the value of E and B at n+1/2
    # Eʰ = -∇(ϕ⁰ + ϕ⁺) / 2 - (A⁺ - A⁰)/dt
    # Bʰ = ∇x(A⁺ + A⁰)/2
    @. field.Ex = -im * field.ffthelper.kx * (field.ϕ + field.ϕ⁻) / 2
    @. field.Ey = -im * field.ffthelper.ky * (field.ϕ + field.ϕ⁻) / 2
    @. field.Ex -= (field.Ax - field.Ax⁻)/dt
    @. field.Ey -= (field.Ay - field.Ay⁻)/dt
    @. field.Ez = -(field.Az - field.Az⁻)/dt
    @. field.Bx = im * field.ffthelper.ky * (field.Az + field.Az⁻) / 2
    @. field.By = -im * field.ffthelper.kx * (field.Az + field.Az⁻) / 2
    @. field.Bz = im * field.ffthelper.kx * (field.Ay + field.Ay⁻) / 2
    @. field.Bz -= im * field.ffthelper.ky * (field.Ax + field.Ax⁻) / 2
  end
  @timeit to "Field Inverse FT" begin
    field.ffthelper.pifft! * field.Ex
    field.ffthelper.pifft! * field.Ey
    field.ffthelper.pifft! * field.Ez
    field.ffthelper.pifft! * field.Bx
    field.ffthelper.pifft! * field.By
    field.ffthelper.pifft! * field.Bz
  end
  @timeit to "Field Update" update!(field)
end


@inline function depositindicesfractions(s::AbstractShape, z::Float64, NZ::Int,
    NZ_Lz::Float64)
  zNZ = z * NZ_Lz # floating point position in units of cells
  # no need for unimod with offset arrays
  i = unimod(ceil(Int, zNZ), NZ) # cell number
  r = i - zNZ; # distance into cell i in units of cell width
  @assert 0 < r <= 1 "$z, $NZ, $NZ_Lz, $i"
  return gridinteractiontuple(s, i, r, NZ)
end

@inline gridinteractiontuple(::NGPWeighting, i, r, NZ) = ((i, 1), )
# no need for unimod with offset arrays
@inline gridinteractiontuple(::AreaWeighting, i, r, NZ) = ((i, 1-r), (unimod(i+1, NZ), r))

bspline(::BSplineWeighting{@stat N}, x) where N = bspline(BSplineWeighting{Int(N)}(), x)

@inline bspline(::BSplineWeighting{0}, x) = ((1.0),)
@inline bspline(::BSplineWeighting{1}, x) = (x, 1-x)
function bspline(::BSplineWeighting{2}, x)
  @cse begin
    a = 9/8 + 3/2*(x-1.5) + 1/2*(x-1.5)^2
    b = 3/4               -     (x-0.5)^2
    c = 9/8 - 3/2*(x+0.5) + 1/2*(x+0.5)^2
  end
  return (a, b, c)
end
function bspline(::BSplineWeighting{3}, x)
  @cse begin
    a = 4/3 + 2*(x-2) + (x-2)^2 + 1/6*(x-2)^3
    b = 2/3           - (x-1)^2 - 1/2*(x-1)^3
    c = 2/3           - (x  )^2 + 1/2*(x  )^3
    d = 4/3 - 2*(x+1) + (x+1)^2 - 1/6*(x+1)^3
  end
  return (a, b, c, d)
end
function bspline(::BSplineWeighting{4}, x)
  @cse begin
    a = 625/384 + 125/48*(x-2.5) + 25/16*(x-2.5)^2 + 5/12*(x-2.5)^3 + 1/24*(x-2.5)^4
    b = 55/96   -   5/24*(x-1.5) -   5/4*(x-1.5)^2 -  5/6*(x-1.5)^3 -  1/6*(x-1.5)^4
    c = 115/192                  -   5/8*(x-0.5)^2                  +  1/4*(x-0.5)^4
    d = 55/96   +   5/24*(x+0.5) -   5/4*(x+0.5)^2 +  5/6*(x+0.5)^3 -  1/6*(x+0.5)^4
    e = 625/384 - 125/48*(x+1.5) + 25/16*(x+1.5)^2 - 5/12*(x+1.5)^3 + 1/24*(x+1.5)^4
  end
  return (a, b, c, d, e)
end
function bspline(::BSplineWeighting{5}, x)
  @cse begin
  a = 243/120 + 81/24*(x-3) + 9/4*(x-3)^2 + 3/4*(x-3)^3 + 1/8*(x-3)^4 + 1/120*(x-3)^5
  b = 17/40   -   5/8*(x-2) - 7/4*(x-2)^2 - 5/4*(x-2)^3 - 3/8*(x-2)^4 -  1/24*(x-2)^5
  c = 22/40                 - 1/2*(x-1)^2               + 1/4*(x-1)^4 +  1/12*(x-1)^5
  d = 22/40                 - 1/2*(x+0)^2               + 1/4*(x-0)^4 -  1/12*(x-0)^5
  e = 17/40   +   5/8*(x+1) - 7/4*(x+1)^2 + 5/4*(x+1)^3 - 3/8*(x+1)^4 +  1/24*(x+1)^5
  f = 243/120 - 81/24*(x+2) + 9/4*(x+2)^2 - 3/4*(x+2)^3 + 1/8*(x+2)^4 - 1/120*(x+2)^5
  end
  return (a, b, c, d, e, f)
end

@inline indices(::BSplineWeighting{N}, i) where N = (i-fld(N, 2)):(i+cld(N, 2))

@inline function gridinteractiontuple(s::BSplineWeighting{N}, i, centre::T, NZ
    ) where {N,T}
  (j, z) = if isodd(N)
    (i, 1 - centre)
  else
    q = centre > 0.5
    (i + q, q + 0.5 - centre)
  end
  inds = indices(s, j)
  fractions = bspline(s, z)
  #@assert sum(fractions) ≈ 1 "$(sum(fractions)), $fractions"
  if (inds[1] < 1) || (inds[end] > NZ)
    # no need for unimod with offset arrays
    return zip(unimod.(inds, NZ), fractions)
  else
    return zip(inds, fractions)
  end
end

# NG = 64
# x = 1/NG/2:1/NG:1-1/NG/2
# h = plot(xticks=0:23)
# for N in 0:5
#   for i in 1:N+1
#     plot!(h, x .+ i .- 1, [bspline(BS{N}(), xi)[i] for xi in x], label="$N,$i")
#   end
# end
#
# for N in 0:5
#   for _ in 1:4
#     cell = rand(10:20)
#     dist = rand()
#     plot!(h, [cell + dist, cell + dist], [0, 1])
#     xc = (N+1)/2
#     for i in 1:N+1
#       plot!(h, x .+ i .- 1 .- xc .+ cell .+ dist,
#        [bspline(BS{N}(), xi)[i] for xi in x], label="$N,$i", legend=false)
#     end
#     for (a, b) in git(BS{N}(), cell, dist, 32)
#       scatter!(h, [a], [b])
#     end
#   end
# end
# h


@inline function (f::ElectrostaticField)(s::AbstractShape, xi, yi)
  NX, NY = f.gridparams.NX, f.gridparams.NY
  NX_Lx, NY_Ly = f.gridparams.NX_Lx, f.gridparams.NY_Ly
  Ex = Ey = zero(eltype(f.Exy))
  for (j, wy) in depositindicesfractions(s, yi, NY, NY_Ly)
    for (i, wx) in depositindicesfractions(s, xi, NX, NX_Lx)
      wxy = wx * wy
      @muladd Ex = Ex + f.Exy[1,i,j] * wxy
      @muladd Ey = Ey + f.Exy[2,i,j] * wxy
    end
  end
  return (Ex, Ey)
end

@inline function (f::LorenzGaugeField)(s::AbstractShape, xi, yi)
  NX, NY = f.gridparams.NX, f.gridparams.NY
  NX_Lx, NY_Ly = f.gridparams.NX_Lx, f.gridparams.NY_Ly
  Ex = Ey = Ez = Bx = By = Bz = zero(real(eltype(f.EBxyz)))
  for (j, wy) in depositindicesfractions(s, yi, NY, NY_Ly)
    for (i, wx) in depositindicesfractions(s, xi, NX, NX_Lx)
      wxy = wx * wy
      @muladd Ex = Ex + f.EBxyz[1,i,j] * wxy
      @muladd Ey = Ey + f.EBxyz[2,i,j] * wxy
      @muladd Ez = Ez + f.EBxyz[3,i,j] * wxy
      @muladd Bx = Bx + f.EBxyz[4,i,j] * wxy
      @muladd By = By + f.EBxyz[5,i,j] * wxy
      @muladd Bz = Bz + f.EBxyz[6,i,j] * wxy
    end
  end
  return (Ex, Ey, Ez, Bx, By, Bz)
end

function deposit!(z, s::AbstractShape, x, y, NX_Lx, NY_Ly, w::Number)
  NX, NY = size(z)
  for (j, wy) in depositindicesfractions(s, y, NY, NY_Ly)
    for (i, wx) in depositindicesfractions(s, x, NX, NX_Lx)
      z[i,j] += wx * wy * w
    end
  end
end

function deposit!(z, s::AbstractShape, x, y, NX_Lx, NY_Ly, w1, w2, w3, w4)
  NV, NX, NY = size(z)
  @assert NV == 4
  for (j, wy) in depositindicesfractions(s, y, NY, NY_Ly)
    for (i, wx) in depositindicesfractions(s, x, NX, NX_Lx)
      wxy = wx * wy
      @muladd z[1,i,j] = z[1,i,j] + wxy * w1
      @muladd z[2,i,j] = z[2,i,j] + wxy * w2
      @muladd z[3,i,j] = z[3,i,j] + wxy * w3
      @muladd z[4,i,j] = z[4,i,j] + wxy * w4
    end
  end
end


function diagnose!(d::AbstractDiagnostics, plasma, to)
  @timeit to "Plasma" begin
    ti = d.ti[]
    d.kineticenergy[ti] = sum(kineticenergy(s) for s in plasma)
    d.particlemomentum[ti] = sum(momentum(s) for s in plasma)
  end
end

function diagnose!(d::ElectrostaticDiagnostics, f::ElectrostaticField, plasma,
                   t, to)
  @timeit to "Diagnostics" begin
    t % d.ntskip == 0 && (d.ti[] += 1)
    if t % d.ntskip == 0
      diagnose!(d, plasma, to)
    end
    @timeit to "Fields" begin
      ti = d.ti[]
      if t % d.ntskip == 0
        d.fieldenergy[ti] = mean(abs2, f.Exy) / 2
      end
      a = 1:d.ngskip:size(f.Ex, 1)
      b = 1:d.ngskip:size(f.Ex, 2)
      @views d.Exs[:, :, ti] .+= real.(f.Ex[a, b]) ./ d.ntskip
      @views d.Eys[:, :, ti] .+= real.(f.Ey[a, b]) ./ d.ntskip
      @views d.ϕs[:, :, ti] .+= real.(f.ffthelper.pifft! * f.ϕ)[a, b] ./ d.ntskip
      f.ffthelper.pfft! * f.ϕ
    end
  end
end

function diagnose!(d::LorenzGaugeDiagnostics, f::LorenzGaugeField, plasma, t, to)
  @timeit to "Diagnostics" begin
    t % d.ntskip == 0 && (d.ti[] += 1)
    if t % d.ntskip == 0
      diagnose!(d, plasma, to)
    end
    @timeit to "Fields" begin
      ti = d.ti[]
      if t % d.ntskip == 0
        @timeit to "Energy" begin
          d.fieldenergy[ti] = mean(abs2, f.EBxyz) / 2
        end
        @timeit to "Momentum" begin
          px, py, pz = 0.0, 0.0, 0.0
          for i in eachindex(f.Ex)
            px += real(f.Ey[i]) * real(f.Bz[i]) - real(f.Ez[i]) * real(f.By[i])
            py += real(f.Ez[i]) * real(f.Bx[i]) - real(f.Ex[i]) * real(f.Bz[i])
            pz += real(f.Ex[i]) * real(f.By[i]) - real(f.Ey[i]) * real(f.Bx[i])
          end
          d.fieldmomentum[ti] .= (px, py, pz) ./ length(f.Ex)
        end
      end
      @timeit to "Field ifft!" begin
        f.ffthelper.pifft! * f.Ax
        f.ffthelper.pifft! * f.Ay
        f.ffthelper.pifft! * f.Az
        f.ffthelper.pifft! * f.ϕ
        f.ffthelper.pifft! * f.ρ;
        f.ffthelper.pifft! * f.Jx;
        f.ffthelper.pifft! * f.Jy;
        f.ffthelper.pifft! * f.Jz;
      end
      @timeit to "Field averaging" begin
        function average!(lhs, rhs)
          a = 1:d.ngskip:size(rhs, 1)
          b = 1:d.ngskip:size(rhs, 2)
          factor = 1 / (d.ntskip * d.ngskip^2)
          for (jl, jr) in enumerate(b), (il, ir) in enumerate(a)
            for jj in 0:d.ngskip-1, ii in 0:d.ngskip-1
              lhs[il, jl, ti] += real(rhs[ir+ii, jr+jj]) * factor
            end
          end
        end
        average!(d.Exs, f.Ex)
        average!(d.Eys, f.Ey)
        average!(d.Ezs, f.Ez)
        average!(d.Bxs, f.Bx)
        average!(d.Bys, f.By)
        average!(d.Bzs, f.Bz)
        average!(d.Axs, f.Ax)
        average!(d.Ays, f.Ay)
        average!(d.Azs, f.Az)
        average!(d.ϕs, f.ϕ)
        average!(d.ρs, f.ρ)
        average!(d.Jxs, f.Jx)
        average!(d.Jys, f.Jy)
        average!(d.Jzs, f.Jz)
      end
      @timeit to "Field fft!" begin
        f.ffthelper.pfft! * f.Ax
        f.ffthelper.pfft! * f.Ay
        f.ffthelper.pfft! * f.Az
        f.ffthelper.pfft! * f.ϕ
        #f.ffthelper.pfft! * f.ρ; # not necessary to transform back - they're overwritten
        #f.ffthelper.pfft! * f.Jx;
        #f.ffthelper.pfft! * f.Jy;
        #f.ffthelper.pfft! * f.Jz;
      end
    end
  end
end

function diagnosticfields(d::ElectrostaticDiagnostics)
  return ((d.Exs, "Ex"), (d.Eys, "Ey"), (d.ϕs, "ϕ"))
end

function diagnosticfields(d::LorenzGaugeDiagnostics)
  return ((d.Exs, "Ex"), (d.Eys, "Ey"), (d.Ezs, "Ez"),
          (d.Bxs, "Bx"), (d.Bys, "By"), (d.Bzs, "Bz"),
          (d.Axs, "Ax"), (d.Ays, "Ay"), (d.Azs, "Az"),
          (d.Jxs, "Jx"), (d.Jys, "Jy"), (d.Jzs, "Jz"),
          (d.ϕs, "ϕ"), (d.ρs, "ρ"))
end


function plotfields(d::AbstractDiagnostics, field, n0, vth, NT; cutoff=Inf)
  B0 = norm(field.B0)
  w0 = iszero(B0) ? sqrt(n0) : B0
  dt = timestep(field)
  g = field.gridparams
  NXd = g.NX÷d.ngskip
  NYd = g.NY÷d.ngskip
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  xs = collect(1:NXd) ./ NXd ./ (vth / w0) * Lx
  ys = collect(1:NYd) ./ NYd ./ (vth / w0) * Ly
  ndiags = d.ti[]
  ts = collect(1:ndiags) .* ((NT * dt / ndiags) / (2pi/w0))

  filter = sin.((collect(1:ndiags) .- 0.5) ./ ndiags .* pi)'
  ws = 2π / (NT * dt) .* (1:ndiags) ./ (w0);
  
  kxs = 2π/Lx .* collect(0:NXd-1) ./ (w0/vth);
  kys = 2π/Ly .* collect(0:NYd-1) ./ (w0/vth);

  k0 = d.fieldenergy[1] + d.kineticenergy[1]

  plot(ts, d.fieldenergy, label="Fields")
  plot!(ts, d.kineticenergy, label="Particles")
  plot!(ts, d.fieldenergy + d.kineticenergy, label="Total")
  savefig("Energies.png")
  
  wind = findlast(ws .< max(cutoff, 6 * sqrt(n0)/w0));
  isnothing(wind) && (wind = length(ws)÷2)
  kxind = min(length(kxs)÷2-1, 128)
  kyind = min(length(kys)÷2-1, 128)
  @views for (F, FS) in diagnosticfields(d)
    all(iszero, F) && (println("$FS is empty"); continue)
    if d.makegifs
      maxabsF = maximum(abs, F)
      maxabsF = iszero(maxabsF) ? 1.0 : maxabsF
      nsx = ceil(Int, size(F,1) / 128)
      nsy = ceil(Int, size(F,2) / 128)
      anim = @animate for i in axes(F, 3)
        heatmap(xs[1:nsx:end], ys[1:nsy:end], F[1:nsx:end, 1:nsy:end, i] ./ maxabsF)
        #heatmap(F[1:nsx:end, 1:nsy:end, i] ./ maxabsF)
        xlabel!(L"Position x $[v_{th} / \Omega]$");
        ylabel!(L"Position y $[v_{th} / \Omega]$")
      end
      gif(anim, "PIC2D3V_$(FS)_XY.gif", fps=10)
    end

    heatmap(xs, ys, F[:, :, 1])
    #heatmap(F[:, :, 1])
    xlabel!(L"Position x $[v_{th} / \Omega]$");
    ylabel!(L"Position y $[v_{th} / \Omega]$")
    savefig("PIC2D3V_$(FS)_XY_ic.png")

    heatmap(xs, ys, F[:, :, end])
    #heatmap(F[:, :, end])
    xlabel!(L"Position x $[v_{th} / \Omega]$");
    ylabel!(L"Position y $[v_{th} / \Omega]$")
    savefig("PIC2D3V_$(FS)_XY_final.png")

#    Z = log10.(abs.(fft(F[:, 1, :] .* filter)[2:kxind, 1:wind]))'
#    heatmap(kxs[2:kxind], ws[1:wind], Z)
#    xlabel!(L"Wavenumber x $[\Omega_c / v_{th}]$");
#    ylabel!(L"Frequency $[\Omega_c]$")
#    savefig("PIC2D3V_$(FS)_WKy1_c.png")
#    xlabel!(L"Wavenumber x $[\Pi / v_{th}]$");
#    ylabel!(L"Frequency $[\Pi]$")
#    heatmap(kxs[2:kxind] .* w0 / sqrt(n0), ws[1:wind] .* w0 / sqrt(n0), Z)
#    savefig("PIC2D3V_$(FS)_WKy1_p.png")
#
#    Z = log10.(abs.(fft(F[1, :, :] .* filter)[2:kyind, 1:wind]))'
#    heatmap(kys[2:kyind], ws[1:wind], Z)
#    xlabel!(L"Wavenumber y $[\Omega_c / v_{th}]$");
#    ylabel!(L"Frequency $[\Omega_c]$")
#    savefig("PIC2D3V_$(FS)_WKx1_c.png")
#    heatmap(kys[2:kyind] .* w0 / sqrt(n0), ws[1:wind] .* w0 / sqrt(n0), Z)
#    xlabel!(L"Wavenumber y $[\Pi / v_{th}]$");
#    ylabel!(L"Frequency $[\Pi]$")
#    savefig("PIC2D3V_$(FS)_WKx1_p.png")

    Z = log10.(sum(i->abs.(fft(F[:, i, :] .* filter)[2:kxind, 1:wind]), 1:size(F, 2)))'
    heatmap(kxs[2:kxind], ws[1:wind], Z)
    #heatmap(Z)
    xlabel!(L"Wavenumber x $[\Omega_c / v_{th}]$");
    ylabel!(L"Frequency $[\Omega_c]$")
    savefig("PIC2D3V_$(FS)_WKsumy_c.png")
    xlabel!(L"Wavenumber x $[\Pi / v_{th}]$");
    ylabel!(L"Frequency $[\Pi]$")
    heatmap(kxs[2:kxind] .* w0 / sqrt(n0), ws[1:wind] .* w0 / sqrt(n0), Z)
    #heatmap(Z)
    savefig("PIC2D3V_$(FS)_WKsumy_p.png")
   
    Z = log10.(sum(i->abs.(fft(F[i, :, :] .* filter)[2:kyind, 1:wind]), 1:size(F, 1)))'
    #heatmap(Z)
    heatmap(kys[2:kyind], ws[1:wind], Z)
    xlabel!(L"Wavenumber y $[\Omega_c / v_{th}]$");
    ylabel!(L"Frequency $[\Omega_c]$")
    savefig("PIC2D3V_$(FS)_WKsumx_c.png")
    #heatmap(Z)
    heatmap(kys[2:kyind] .* w0 / sqrt(n0), ws[1:wind] .* w0 / sqrt(n0), Z)
    xlabel!(L"Wavenumber y $[\Pi / v_{th}]$");
    ylabel!(L"Frequency $[\Pi]$")
    savefig("PIC2D3V_$(FS)_WKsumx_p.png")
  end

end

end


