module PIC2D3V

using FFTW,Plots, SpecialFunctions, StaticArrays, LinearAlgebra, Random
using LoopVectorization, Base.Threads, ThreadsX, Base.Iterators, Statistics
using ProgressMeter, LaTeXStrings, MuladdMacro, CommonSubexpressions
using TimerOutputs, StaticNumbers, OffsetArrays, FastPow, ThreadsX
using QuasiMonteCarlo

unimod(x, n) = x > n ? x - n : x > 0 ? x : x + n

function applyperiodicity!(a::Array, oa)
  NX, NY = size(a)
  @assert length(size(a)) == 2
  @assert length(size(oa)) == 2
  for j in axes(oa, 2), i in axes(oa, 1)
    a[unimod(i, NX), unimod(j, NY)] += oa[i, j]
  end
end

function applyperiodicity!(oa, a::Array)
  NX, NY = size(a)
  @assert length(size(a)) == 2
  @assert length(size(oa)) == 2
  for j in axes(oa, 2), i in axes(oa, 1)
     oa[i, j] += real(a[unimod(i, NX), unimod(j, NY)])
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
  characteristicmomentum::Vector{Vector{Float64}}
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
    NX÷ngskip, NY÷ngskip, NT÷ntskip, 2, 2, 3)
  return ElectrostaticDiagnostics(scalarstorage..., momentumstorage...,
    fieldstorage..., ntskip, ngskip, Ref(0), makegifs)
end

struct LorenzGaugeDiagnostics <: AbstractDiagnostics
  kineticenergy::Array{Float64, 1}
  fieldenergy::Array{Float64, 1}
  particlemomentum::Vector{Vector{Float64}}
  fieldmomentum::Vector{Vector{Float64}}
  characteristicmomentum::Vector{Vector{Float64}}
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
    NX÷ngskip, NY÷ngskip, NT÷ntskip, 2, 3, 14)
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
  xyvwork::Matrix{Float64}
end
function positions(s::Species; work=false)
  return work ? (@view s.xyvwork[1:2, :]) : (@view s.xyv[1:2, :])
end
function velocities(s::Species; work=false)
  return work ? (@view s.xyvwork[3:5, :]) : (@view s.xyv[3:5, :])
end
xyvchunk(s::Species, i::Int) = @view s.xyv[:, s.chunks[i]]

function copyto!(dest::Species, src::Species)
  @tturbo dest.xyv .= src.xyv
  @tturbo dest.p .= src.p
  return dest
end

kineticenergy(s::Species) = sum(abs2, velocities(s)) * s.mass / 2 * s.weight

function momentum(s::Species, op::F=identity) where F
  #output = sum(op.(velocities(s)), dims=2)[:] * s.mass * s.weight
  output = @MVector [0.0, 0.0, 0.0]
  for v in eachcol(velocities(s))
    output .+= op.(v)
  end
  output .*= s.mass * s.weight
  return output
end
characteristicmomentum(s::Species) = momentum(s, abs)

calculateweight(n0, P, Lx, Ly) = n0 * Lx * Ly / P;

sample(P, i) = halton.(0:P-1, i, 1/sqrt(2));#
#sample(P, _) = unimod.(rand() .+ reshape(QuasiMonteCarlo.sample(P,1,GoldenSample()), P), 1)
#sample(P, _) = unimod.(rand() .+ rand(P), 1)

function Species(P, vth, density, shape::AbstractShape; Lx, Ly, charge=1, mass=1)
  x  = Lx * sample(P, 2);
  y  = Ly * sample(P, 3);
  vx = vth * erfinv.(2sample(P, 5) .- 1) * vth;
  vy = vth * erfinv.(2sample(P, 7) .- 1) * vth;
  vz = vth * erfinv.(2sample(P, 9) .- 1) * vth;
  vx .-= mean(vx)
  vy .-= mean(vy)
  vz .-= mean(vz)
  vx .*= (vth / sqrt(2)) / std(vx);
  vy .*= (vth / sqrt(2)) / std(vy);
  vz .*= (vth / sqrt(2)) / std(vz);
  p  = collect(1:P)
  xyv = Matrix(hcat(x, y, vx, vy, vz)')
  chunks = collect(Iterators.partition(1:P, ceil(Int, P/nthreads())))
  weight = calculateweight(density, P, Lx, Ly)
  return Species(Float64(charge), Float64(mass), weight, shape, xyv, p, chunks, deepcopy(xyv))
end

function Base.sort!(s::Species, Δx, Δy)
  sortperm!(s.p, eachindex(s.p),
    by=i->(ceil(Int, s.xyv[1,i] / Δx), ceil(Int, s.xyv[2,i] / Δy), s.xyv[3,i]))
  s.xyv .= s.xyv[:, s.p]
  return nothing
end


abstract type AbstractField end
abstract type AbstractLorenzGaugeField <: AbstractField end
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

struct FFTHelper{T, U}
  kx::Vector{Float64}
  ky::LinearAlgebra.Adjoint{Float64, Vector{Float64}}
  k²::Matrix{Float64}
  im_k⁻²::Matrix{ComplexF64}
  smoothingkernel::Matrix{ComplexF64}
  pfft!::T
  pifft!::U
end
function FFTHelper(NX, NY, Lx, Ly)
  kx = 2π / Lx * vcat(0:NX÷2-1, -NX÷2:-1);
  ky = 2π / Ly * vcat(0:NY÷2-1, -NY÷2:-1)';
  k² = (kx.^2 .+ ky.^2)
  im_k⁻² = -im ./ k²
  im_k⁻²[1, 1] = 0
  z = zeros(ComplexF64, NX, NY)
  kernel = exp.(-(-6:6).^2)
  smoothingkernel = zeros(ComplexF64, NX, NY)
  smoothingkernel[1:size(kernel,1), 1:size(kernel, 1)] .= sqrt.(kernel .* kernel')
  smoothingkernel ./= sum(smoothingkernel)

  pfft! = plan_fft!(z; flags=FFTW.ESTIMATE, timelimit=Inf)
  pifft! = plan_ifft!(z; flags=FFTW.ESTIMATE, timelimit=Inf)
  pfft! * smoothingkernel
  return FFTHelper(kx, ky, k², im_k⁻², smoothingkernel, pfft!, pifft!)
end


struct ElectrostaticField{T} <: AbstractField
#  ρs::Array{Float64, 3}
  ρs::OffsetArray{Float64, 3, Array{Float64, 3}}# Array{Float64, 3} # offset array
  ϕ::Array{ComplexF64, 2}
  Ex::Array{ComplexF64, 2}
  Ey::Array{ComplexF64, 2}
#  Exy::Array{Float64, 3}
  Exy::OffsetArray{Float64, 3, Array{Float64, 3}}#Array{Float64, 3} # offset array
  B0::NTuple{3, Float64}
  gridparams::GridParameters
  ffthelper::T
  boris::ElectrostaticBoris
end

function ElectrostaticField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0, buffer=3)
  ρs = OffsetArray(zeros(NX+2buffer, NY+2buffer, nthreads()), -(buffer-1):NX+buffer, -(buffer-1):NY+buffer, 1:nthreads());
  Exy = OffsetArray(zeros(2, NX+2buffer, NY+2buffer), 1:2, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer);
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  gps = GridParameters(Lx, Ly, NX, NY)
  boris = ElectrostaticBoris([B0x, B0y, B0z], dt)
  return ElectrostaticField(ρs, (zeros(ComplexF64, NX, NY) for _ in 1:3)...,
    Exy, Float64.((B0x, B0y, B0z)), gps, ffthelper, boris)
end

function update!(f::ElectrostaticField)
  applyperiodicity!((@view f.Exy[1, :, :]), f.Ex)
  applyperiodicity!((@view f.Exy[2, :, :]), f.Ey)
end

abstract type AbstractImEx end
struct Explicit <: AbstractImEx end
struct Implicit <: AbstractImEx end
struct ImEx <: AbstractImEx
  θ::Float64
end


struct LorenzGaugeField{T, U} <: AbstractLorenzGaugeField
  imex::T
  ρJs::OffsetArray{Float64, 4, Array{Float64, 4}} # would need to be offset array
  ϕ⁺::Array{ComplexF64, 2}
  ϕ⁰::Array{ComplexF64, 2}
  Ax⁺::Array{ComplexF64, 2}
  Ay⁺::Array{ComplexF64, 2}
  Az⁺::Array{ComplexF64, 2}
  Ax⁰::Array{ComplexF64, 2}
  Ay⁰::Array{ComplexF64, 2}
  Az⁰::Array{ComplexF64, 2}
  Ex::Array{ComplexF64, 2}
  Ey::Array{ComplexF64, 2}
  Ez::Array{ComplexF64, 2}
  ρ⁺::Array{ComplexF64, 2}
  Jx⁺::Array{ComplexF64, 2}
  Jy⁺::Array{ComplexF64, 2}
  Jz⁺::Array{ComplexF64, 2}
  Bx::Array{ComplexF64, 2}
  By::Array{ComplexF64, 2}
  Bz::Array{ComplexF64, 2}
  EBxyz::OffsetArray{Float64, 3, Array{Float64, 3}}# would need to be offset array
  B0::NTuple{3, Float64}
  gridparams::GridParameters
  ffthelper::U
  boris::ElectromagneticBoris
  dt::Float64
end

function LorenzGaugeField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0,
    imex::AbstractImEx=Explicit(), buffer=0)
  EBxyz = OffsetArray(zeros(6, NX+2buffer, NY+2buffer), 1:6, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer);
  gps = GridParameters(Lx, Ly, NX, NY)
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  boris = ElectromagneticBoris(dt)
  ρJs = OffsetArray(zeros(4, NX+2buffer, NY+2buffer, nthreads()),
    1:4, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer, 1:nthreads());
  return LorenzGaugeField(imex, ρJs, (zeros(ComplexF64, NX, NY) for _ in 1:18)...,
    EBxyz, Float64.((B0x, B0y, B0z)), gps, ffthelper, boris, dt)
end
struct LorenzGaugeStaggeredField{T, U} <: AbstractLorenzGaugeField
  imex::T
  ρJs⁰::OffsetArray{Float64, 4, Array{Float64, 4}}
  ϕ⁺::Array{ComplexF64, 2}
  ϕ⁰::Array{ComplexF64, 2}
  ϕ⁻::Array{ComplexF64, 2}
  Ax⁺::Array{ComplexF64, 2}
  Ay⁺::Array{ComplexF64, 2}
  Az⁺::Array{ComplexF64, 2}
  Ax⁻::Array{ComplexF64, 2}
  Ay⁻::Array{ComplexF64, 2}
  Az⁻::Array{ComplexF64, 2}
  Ax⁰::Array{ComplexF64, 2}
  Ay⁰::Array{ComplexF64, 2}
  Az⁰::Array{ComplexF64, 2}
  Ex::Array{ComplexF64, 2}
  Ey::Array{ComplexF64, 2}
  Ez::Array{ComplexF64, 2}
 # ρ⁻::Array{ComplexF64, 2}
  ρ⁰::Array{ComplexF64, 2}
#  ρ⁺::Array{ComplexF64, 2}
#  Jx⁻::Array{ComplexF64, 2}
#  Jy⁻::Array{ComplexF64, 2}
#  Jz⁻::Array{ComplexF64, 2}
  Jx⁰::Array{ComplexF64, 2}
  Jy⁰::Array{ComplexF64, 2}
  Jz⁰::Array{ComplexF64, 2}
#  Jx⁺::Array{ComplexF64, 2}
#  Jy⁺::Array{ComplexF64, 2}
#  Jz⁺::Array{ComplexF64, 2}
  Bx::Array{ComplexF64, 2}
  By::Array{ComplexF64, 2}
  Bz::Array{ComplexF64, 2}
  EBxyz::OffsetArray{Float64, 3, Array{Float64, 3}}
  B0::NTuple{3, Float64}
  gridparams::GridParameters
  ffthelper::U
  boris::ElectromagneticBoris
  dt::Float64
end

function LorenzGaugeStaggeredField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0,
    imex::AbstractImEx=Explicit(), buffer=0)
  EBxyz = OffsetArray(zeros(6, NX+2buffer, NY+2buffer), 1:6, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer);
  gps = GridParameters(Lx, Ly, NX, NY)
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  boris = ElectromagneticBoris(dt)
  ρJs = OffsetArray(zeros(4, NX+2buffer, NY+2buffer, nthreads()),
    1:4, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer, 1:nthreads());
  return LorenzGaugeStaggeredField(imex, ρJs,
    (zeros(ComplexF64, NX, NY) for _ in 1:22)..., EBxyz,
    Float64.((B0x, B0y, B0z)), gps, ffthelper, boris, dt)
end



struct LorenzGaugeSemiImplicitField{T, U, V} <: AbstractLorenzGaugeField
  fieldimex::T
  sourceimex::U
  ρJs⁻::OffsetArray{Float64, 4, Array{Float64, 4}}
  ρJs⁰::OffsetArray{Float64, 4, Array{Float64, 4}}
  ρJs⁺::OffsetArray{Float64, 4, Array{Float64, 4}}
  ρJsᵗ::OffsetArray{Float64, 4, Array{Float64, 4}}
  ϕ⁺::Array{ComplexF64, 2}
  ϕ⁰::Array{ComplexF64, 2}
  ϕ⁻::Array{ComplexF64, 2}
  Ax⁺::Array{ComplexF64, 2}
  Ay⁺::Array{ComplexF64, 2}
  Az⁺::Array{ComplexF64, 2}
  Ax⁻::Array{ComplexF64, 2}
  Ay⁻::Array{ComplexF64, 2}
  Az⁻::Array{ComplexF64, 2}
  Ax⁰::Array{ComplexF64, 2}
  Ay⁰::Array{ComplexF64, 2}
  Az⁰::Array{ComplexF64, 2}
  Ex::Array{ComplexF64, 2}
  Ey::Array{ComplexF64, 2}
  Ez::Array{ComplexF64, 2}
  ρ⁻::Array{ComplexF64, 2}
  ρ⁰::Array{ComplexF64, 2}
  ρ⁺::Array{ComplexF64, 2}
  Jx⁻::Array{ComplexF64, 2}
  Jy⁻::Array{ComplexF64, 2}
  Jz⁻::Array{ComplexF64, 2}
  Jx⁰::Array{ComplexF64, 2}
  Jy⁰::Array{ComplexF64, 2}
  Jz⁰::Array{ComplexF64, 2}
  Jx⁺::Array{ComplexF64, 2}
  Jy⁺::Array{ComplexF64, 2}
  Jz⁺::Array{ComplexF64, 2}
  Bx::Array{ComplexF64, 2}
  By::Array{ComplexF64, 2}
  Bz::Array{ComplexF64, 2}
  EBxyz::OffsetArray{Float64, 3, Array{Float64, 3}}
  B0::NTuple{3, Float64}
  gridparams::GridParameters
  ffthelper::V
  boris::ElectromagneticBoris
  dt::Float64
  rtol::Float64
  maxiters::Int
end

function LorenzGaugeSemiImplicitField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0,
    fieldimex::AbstractImEx=Explicit(), sourceimex::AbstractImEx=Explicit(),
    buffer=0, rtol=sqrt(eps()), maxiters=10)
  EBxyz = OffsetArray(zeros(6, NX+2buffer, NY+2buffer), 1:6, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer);
  gps = GridParameters(Lx, Ly, NX, NY)
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  boris = ElectromagneticBoris(dt)
  ρJs = OffsetArray(zeros(4, NX+2buffer, NY+2buffer, nthreads()),
    1:4, -(buffer-1):NX+buffer, -(buffer-1):NY+buffer, 1:nthreads());
  return LorenzGaugeSemiImplicitField(fieldimex, sourceimex, ρJs, deepcopy(ρJs), deepcopy(ρJs),
    deepcopy(ρJs), (zeros(ComplexF64, NX, NY) for _ in 1:30)..., EBxyz,
    Float64.((B0x, B0y, B0z)), gps, ffthelper, boris, dt, rtol, maxiters)
end

theta(::Explicit) = 0
theta(imex::ImEx) = imex.θ
theta(::Implicit) = 1

function update!(f::AbstractLorenzGaugeField)
  f.EBxyz .= 0.0
  applyperiodicity!((@view f.EBxyz[1, :, :]), f.Ex)
  applyperiodicity!((@view f.EBxyz[2, :, :]), f.Ey)
  applyperiodicity!((@view f.EBxyz[3, :, :]), f.Ez)
  applyperiodicity!((@view f.EBxyz[4, :, :]), f.Bx)
  applyperiodicity!((@view f.EBxyz[5, :, :]), f.By)
  applyperiodicity!((@view f.EBxyz[6, :, :]), f.Bz)
  @views for k in axes(f.EBxyz, 3), j in axes(f.EBxyz, 2), i in 1:3
    f.EBxyz[i+3, j, k] += f.B0[i]
  end
end

function reduction!(a, z)
  @. a = 0.0
  @views for k in axes(z, 3)
    applyperiodicity!(a, z[:, :, k])
  end
end

function reduction!(a, b, c, z)
  @assert size(z, 1) == 4
  @. a = 0.0
  @. b = 0.0
  @. c = 0.0
  @views for k in axes(z, 4)
    applyperiodicity!(a, z[1, :, :, k])
  end
  @views for k in axes(z, 4)
    applyperiodicity!(b, z[2, :, :, k])
  end
  @views for k in axes(z, 4)
    applyperiodicity!(c, z[3, :, :, k])
  end
end

function reduction!(a, b, c, d, z)
  @assert size(z, 1) == 4
  @. a = 0.0
  @. b = 0.0
  @. c = 0.0
  @. d = 0.0
  @views for k in axes(z, 4)
    applyperiodicity!(a, z[1, :, :, k])
  end
  @views for k in axes(z, 4)
    applyperiodicity!(b, z[2, :, :, k])
  end
  @views for k in axes(z, 4)
    applyperiodicity!(c, z[3, :, :, k])
  end
  @views for k in axes(z, 4)
    applyperiodicity!(d, z[4, :, :, k])
  end
end

# E = -∇ ϕ
# ∇ . E = -∇.∇ ϕ = -∇^2 ϕ = ρ
# -i^2 (kx^2 + ky^2) ϕ = ρ
# ϕ = ρ / (kx^2 + ky^2)
# Ex = - ∇_x ϕ = - i kx ϕ = - i kx ρ / (kx^2 + ky^2)
# Ey = - ∇_y ϕ = - i ky ϕ = - i ky ρ / (kx^2 + ky^2)
function loop!(plasma, field::ElectrostaticField, to, t, _)
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
    field.ρs .= 0
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

function smooth!(a, ffthelper)
  ffthelper.pfft! * a
  a .*= ffthelper.smoothingkernel
  ffthelper.pifft! * a
  a .= real(a)
end

function smooth!(a, b, c, d, ffthelper)
  smooth!(a, ffthelper)
  smooth!(b, ffthelper)
  smooth!(c, ffthelper)
  smooth!(d, ffthelper)
end

## -∇² lhs = rhs
function neglaplacesolve!(lhs, rhs, ffthelper)
  ffthelper.pfft! * rhs
  @threads for j in axes(lhs, 2)
    for i in axes(lhs, 1)
      lhs[i, j] = rhs[i, j] / ffthelper.k²[i, j]
    end
  end
  lhs[1, 1] = 0
  ffthelper.pifft! * rhs
  ffthelper.pifft! * lhs
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

function lorenzgauge!(imex::AbstractImEx, xⁿ⁺¹, xⁿ, xⁿ⁻¹, sⁿ, k², dt²)
  θ = theta(imex)
  @threads for i in eachindex(xⁿ)
    num = numerator(imex, dt², k²[i])
    den = denominator(imex, dt², k²[i])
    xⁿ⁺¹[i] = (num * xⁿ[i] + dt² * sⁿ[i]) / den - xⁿ⁻¹[i]
  end
end

function lorenzgauge!(fieldimex::AbstractImEx, xⁿ⁺¹, xⁿ, xⁿ⁻¹, sⁿ⁺¹, sⁿ, sⁿ⁻¹, k², dt², sourceimex=fieldimex)
  θ = theta(sourceimex)
  @threads for i in eachindex(xⁿ)
    num = numerator(fieldimex, dt², k²[i])
    den = denominator(fieldimex, dt², k²[i])
    #xⁿ⁺¹[i] = (num * xⁿ[i] + dt² * (θ/2 * sⁿ⁻¹[i] + (1 - θ) * sⁿ[i] + θ/2 * sⁿ⁺¹[i])) / den - xⁿ⁻¹[i]
    xⁿ⁺¹[i] = (num * xⁿ[i] + dt² * (θ/2 * sⁿ⁻¹[i] + (1 - θ) * sⁿ[i] + θ/2 * sⁿ⁺¹[i])) / den - xⁿ⁻¹[i]
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
function loop!(plasma, field::LorenzGaugeField, to, t, _)
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
    reduction!(field.ρ⁺, field.Jx⁺, field.Jy⁺, field.Jz⁺, field.ρJs)
    field.ρJs .= 0
  end
  @timeit to "Field Forward FT" begin
    field.ffthelper.pfft! * field.ρ⁺;
    field.ffthelper.pfft! * field.Jx⁺;
    field.ffthelper.pfft! * field.Jy⁺;
    field.ffthelper.pfft! * field.Jz⁺;
  end
  @timeit to "Field Solve" begin
    field.ρ⁺[1, 1] = 0
    field.Jx⁺[1, 1] = 0
    field.Jy⁺[1, 1] = 0
    field.Jz⁺[1, 1] = 0
    # at this point ϕ stores the nth timestep value and ϕ⁻ the (n-1)th
    lorenzgauge!(field.imex, field.ϕ⁺, field.ϕ⁰, field.ρ⁺, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ax⁺, field.Ax⁰, field.Jx⁺, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ay⁺, field.Ay⁰, field.Jy⁺, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Az⁺, field.Az⁰, field.Jz⁺, field.ffthelper.k², dt^2)
    # at this point (ϕ, Ai) stores the (n+1)th timestep value and (ϕ⁻, Ai⁻) the nth
    # Now calculate the value of E and B at n+1/2
    # Eʰ = -∇(ϕ⁰ + ϕ⁺) / 2 - (A⁺ - A⁰)/dt
    # Bʰ = ∇x(A⁺ + A⁰)/2
    @. field.Ex = -im * field.ffthelper.kx * (field.ϕ⁺ + field.ϕ⁰) / 2
    @. field.Ey = -im * field.ffthelper.ky * (field.ϕ⁺ + field.ϕ⁰) / 2
    @. field.Ex -= (field.Ax⁺ - field.Ax⁰)/dt
    @. field.Ey -= (field.Ay⁺ - field.Ay⁰)/dt
    @. field.Ez = -(field.Az⁺ - field.Az⁰)/dt
    @. field.Bx = im * field.ffthelper.ky * (field.Az⁺ + field.Az⁰) / 2
    @. field.By = -im * field.ffthelper.kx * (field.Az⁺ + field.Az⁰) / 2
    @. field.Bz = im * field.ffthelper.kx * (field.Ay⁺ + field.Ay⁰) / 2
    @. field.Bz -= im * field.ffthelper.ky * (field.Ax⁺ + field.Ax⁰) / 2
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

warmup!(field::AbstractField, plasma, to) = field
function warmup!(ρ, Jx, Jy, Jz, ρJs, plasma, gridparams, dt, to)
  Lx, Ly = gridparams.Lx, gridparams.Ly
  NX_Lx, NY_Ly = gridparams.NX_Lx, gridparams.NY_Ly
  ΔV = cellvolume(gridparams)
  ρJs .= 0
  @timeit to "Particle Loop" begin
    @threads for j in axes(ρJs, 4)
      ρJ = @view ρJs[:, :, :, j]
      for species in plasma
        qw_ΔV = species.charge * species.weight / ΔV
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        for i in species.chunks[j]
          x[i] = unimod(x[i] + vx[i] * dt, Lx)
          y[i] = unimod(y[i] + vy[i] * dt, Ly)
          deposit!(ρJ, species.shape, x[i], y[i], NX_Lx, NY_Ly,
            qw_ΔV, vx[i] * qw_ΔV, vy[i] * qw_ΔV, vz[i] * qw_ΔV)
          x[i] = unimod(x[i] - vx[i] * dt, Lx)
          y[i] = unimod(y[i] - vy[i] * dt, Ly)
        end
      end
    end
  end
  @timeit to "Field Reduction" begin
    reduction!(ρ, Jx, Jy, Jz, ρJs)
  end
end
function advect!(plasma, gridparams, dt, to)
  Lx, Ly = gridparams.Lx, gridparams.Ly
  @timeit to "Advect Loop" begin
    for species in plasma
      @threads for j in eachindex(species.chunks)
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        for i in species.chunks[j]
          x[i] = unimod(x[i] + vx[i] * dt, Lx)
          y[i] = unimod(y[i] + vy[i] * dt, Ly)
        end
      end
    end
  end
end
function defaultdepositcallback(qw_ΔV, vx, vy, vz)
    return (1.0, vx, vy, vz) .* qw_ΔV
end
function deposit!(ρ, Jx, Jy, Jz, ρJs, plasma, gridparams, dt, to,
        cb::F=defaultdepositcallback) where F
  Lx, Ly = gridparams.Lx, gridparams.Ly
  NX_Lx, NY_Ly = gridparams.NX_Lx, gridparams.NY_Ly
  ΔV = cellvolume(gridparams)
  ρJs .= 0
  @timeit to "Particle Loop" begin
    @threads for j in axes(ρJs, 4)
      ρJ = @view ρJs[:, :, :, j]
      for species in plasma
        qw_ΔV = species.charge * species.weight / ΔV
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        for i in species.chunks[j]
          deposit!(ρJ, species.shape, x[i], y[i], NX_Lx, NY_Ly,
                   cb(qw_ΔV, vx[i], vy[i], vz[i])...)
        end
      end
    end
  end
  @timeit to "Field Reduction" begin
    reduction!(ρ, Jx, Jy, Jz, ρJs)
  end
end


function warmup!(field::LorenzGaugeStaggeredField, plasma, to)
  ρcallback(qw_ΔV, vx, vy, vz) = (qw_ΔV,)
  Jcallback(qw_ΔV, vx, vy, vz) = qw_ΔV .* (vx, vy, vz)
  @timeit to "Warmup" begin
    dt = timestep(field)
#    advect!(plasma, field.gridparams, -3dt/2, to) #n - 3/2
#    deposit!(field.ρ⁻, field.Jx⁻, field.Jy⁻, field.Jz⁻, field.ρJs⁺, plasma, field.gridparams, -dt, to, ρcallback)
#    #neglaplacesolve!(field.ϕ⁻, field.ρ⁻, field.ffthelper)
#    advect!(plasma, field.gridparams, dt/2, to) #(n-1)
#    deposit!(field.ρ⁻, field.Jx⁻, field.Jy⁻, field.Jz⁻, field.ρJs⁺, plasma, field.gridparams, -dt, to, Jcallback)
#    #neglaplacesolve!(field.Ax⁻, field.Jx⁻, field.ffthelper)
#    #neglaplacesolve!(field.Ay⁻, field.Jy⁻, field.ffthelper)
#    #neglaplacesolve!(field.Az⁻, field.Jz⁻, field.ffthelper)
#    field.ρJs⁺ .= 0

    advect!(plasma, field.gridparams, -dt/2, to)
    deposit!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.ρJs⁰, plasma, field.gridparams, dt, to, ρcallback)
    #neglaplacesolve!(field.ϕ⁰, field.ρ⁰, field.ffthelper)
    advect!(plasma, field.gridparams, dt/2, to) # back to start, n
    deposit!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.ρJs⁰, plasma, field.gridparams, dt, to, Jcallback)
    #neglaplacesolve!(field.Ax⁰, field.Jx⁰, field.ffthelper)
    #neglaplacesolve!(field.Ay⁰, field.Jy⁰, field.ffthelper)
    #neglaplacesolve!(field.Az⁰, field.Jz⁰, field.ffthelper)
    field.ρJs⁰ .= 0

#    advect!(plasma, field.gridparams, dt/2, to) # n + 1/2
#    deposit!(field.ρ⁺, field.Jx⁺, field.Jy⁺, field.Jz⁺, field.ρJs⁰, plasma, field.gridparams, dt, to, ρcallback)
#    #neglaplacesolve!(field.ϕ⁺, field.ρ⁺, field.ffthelper)
#    advect!(plasma, field.gridparams, dt/2, to) # n+1
#    deposit!(field.ρ⁺, field.Jx⁺, field.Jy⁺, field.Jz⁺, field.ρJs⁰, plasma, field.gridparams, dt, to, Jcallback)
#    advect!(plasma, field.gridparams, -dt, to) # advect back to start
#    #neglaplacesolve!(field.Ax⁺, field.Jx⁺, field.ffthelper)
#    #neglaplacesolve!(field.Ay⁺, field.Jy⁺, field.ffthelper)
#    #neglaplacesolve!(field.Az⁺, field.Jz⁺, field.ffthelper)
#    field.ρJs⁺ .= 0
  end
end

function warmup!(field::LorenzGaugeSemiImplicitField, plasma, to)
  @timeit to "Warmup" begin
    dt = timestep(field)
    warmup!(field.ρ⁻, field.Jx⁻, field.Jy⁻, field.Jz⁻, field.ρJs⁻, plasma, field.gridparams, -dt, to)
    warmup!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.ρJs⁰, plasma, field.gridparams, dt, to)
    warmup!(field.ρ⁺, field.Jx⁺, field.Jy⁺, field.Jz⁺, field.ρJs⁺, plasma, field.gridparams, dt, to)
    advect!(plasma, field.gridparams, -dt, to) # advect back to start
  end
end


function loop!(plasma, field::LorenzGaugeStaggeredField, to, t, _)
  dt = timestep(field)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  ΔV = cellvolume(field.gridparams)
  Lx, Ly, NX_Lx, NY_Ly
  # assume ρ and J are up to date at the current time
  @timeit to "Field Solve" begin
    # at this point ϕ stores the nth timestep value and ϕ⁻ the (n-1)th
    lorenzgauge!(field.imex, field.ϕ⁺,  field.ϕ⁰,  field.ϕ⁻,  field.ρ⁰,  field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ax⁺, field.Ax⁰, field.Ax⁻, field.Jx⁰, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ay⁺, field.Ay⁰, field.Ay⁻, field.Jy⁰, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Az⁺, field.Az⁰, field.Az⁻, field.Jz⁰, field.ffthelper.k², dt^2)
  end

  @timeit to "Field Forward FT" begin
    field.ffthelper.pfft! * field.ρ⁰;
    field.ffthelper.pfft! * field.Jx⁰;
    field.ffthelper.pfft! * field.Jy⁰;
    field.ffthelper.pfft! * field.Jz⁰;
    field.ρ⁰ .*= field.ffthelper.smoothingkernel
    field.Jx⁰ .*= field.ffthelper.smoothingkernel
    field.Jy⁰ .*= field.ffthelper.smoothingkernel
    field.Jz⁰ .*= field.ffthelper.smoothingkernel
    field.ρ⁰[1, 1] = 0
    field.Jx⁰[1, 1] = 0
    field.Jy⁰[1, 1] = 0
    field.Jz⁰[1, 1] = 0
  end

  # at this point (ϕ, Ai) stores the (n+1)th timestep value and (ϕ⁻, Ai⁻) the nth
  # Now calculate the value of E and B at n+1/2
  # Eʰ = -∇ϕ⁺ - (A⁺ - A⁰)/dt
  # Bʰ = ∇x(A⁺ + A⁰)/2

  #  E.....E.....E
  #  B.....B.....B
  #  ϕ.....ϕ.....ϕ
  #  -..A..0..A..+..A
  #  ρ.....ρ.....ρ
  #  -..J..0..J..+..J
  #  x.....x.....x
  #  -..v..0..v..+..v
  @timeit to "Calculate E, B" begin
    @. field.Ex = -im * field.ffthelper.kx * field.ϕ⁺
    @. field.Ey = -im * field.ffthelper.ky * field.ϕ⁺
    @. field.Ex -= (field.Ax⁺ - field.Ax⁰)/dt
    @. field.Ey -= (field.Ay⁺ - field.Ay⁰)/dt
    @. field.Ez = -(field.Az⁺ - field.Az⁰)/dt
    @. field.Bx =  im * field.ffthelper.ky * (field.Az⁺ + field.Az⁰)/2
    @. field.By = -im * field.ffthelper.kx * (field.Az⁺ + field.Az⁰)/2
    @. field.Bz =  im * field.ffthelper.kx * (field.Ay⁺ + field.Ay⁰)/2
    @. field.Bz -= im * field.ffthelper.ky * (field.Ax⁺ + field.Ax⁰)/2
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

  @timeit to "Particle Loop" begin
    @threads for j in axes(field.ρJs⁰, 4)
      ρJ⁰ = @view field.ρJs⁰[:, :, :, j]
      for species in plasma
        qw_ΔV = species.charge * species.weight / ΔV
        q_m = species.charge / species.mass
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        #  E.....E.....E
        #  B.....B.....B
        #  ...ϕ.....ϕ.....ϕ
        #  A..0..A..+..A
        #  ...ρ.....ρ.....ρ
        #  J..0..J..+..J
        #  x.....x.....x
        #  v..0..v..+..v
        @inbounds for i in species.chunks[j]
          Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(species.shape, x[i], y[i])
          vx[i], vy[i], vz[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, Ezi,
            Bxi, Byi, Bzi, q_m);
          x[i] = unimod(x[i] + vx[i] * dt/2, Lx)
          y[i] = unimod(y[i] + vy[i] * dt/2, Ly)
          # deposit ρ at (n+1/2)th timestep
          deposit!(ρJ⁰, species.shape, x[i], y[i], NX_Lx, NY_Ly, qw_ΔV)
          x[i] = unimod(x[i] + vx[i] * dt/2, Lx)
          y[i] = unimod(y[i] + vy[i] * dt/2, Ly)
          # deposit J at the (n+1)th point
          deposit!(ρJ⁰, species.shape, x[i], y[i], NX_Lx, NY_Ly,
            vx[i] * qw_ΔV, vy[i] * qw_ΔV, vz[i] * qw_ΔV)
        end
      end
    end
  end
  @timeit to "Field Reduction" begin
    reduction!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.ρJs⁰)
    #smooth!(field.ρ⁺, field.Jx⁺, field.Jy⁺, field.Jz⁺, field.ffthelper)
    field.ρJs⁰ .= 0
  end

  @timeit to "Copy over buffers" begin
    field.ϕ⁻ .= field.ϕ⁰
    field.ϕ⁰ .= field.ϕ⁺
    field.Ax⁻ .= field.Ax⁰
    field.Ax⁰ .= field.Ax⁺
    field.Ay⁻ .= field.Ay⁰
    field.Ay⁰ .= field.Ay⁺
    field.Az⁻ .= field.Az⁰
    field.Az⁰ .= field.Az⁺
  end
end

function loop!(plasma, field::LorenzGaugeSemiImplicitField, to, t, plasmacopy = deepcopy(plasma))
  dt = timestep(field)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  ΔV = cellvolume(field.gridparams)

  copyto!.(plasmacopy, plasma)
  firstloop = true
  iters = 0
  while true
    if (iters > 0) && (iters > field.maxiters || isapprox(field.ρJsᵗ, field.ρJs⁺, rtol=field.rtol, atol=0))
      for species in plasma
        x = positions(species)
        v = velocities(species)
        xʷ = positions(species; work=true)
        vʷ = velocities(species; work=true)
        @tturbo x .= xʷ
        @tturbo v .= vʷ
      end
      break
    end
    iters += 1
    copyto!.(plasma, plasmacopy)
    @tturbo field.ρJsᵗ .= field.ρJs⁺
    @tturbo field.ρJs⁺ .= 0
    @timeit to "Particle Loop" begin
      @threads for j in axes(field.ρJs⁺, 4)
        ρJ⁺ = @view field.ρJs⁺[:, :, :, j]
        for species in plasma
          qw_ΔV = species.charge * species.weight / ΔV
          q_m = species.charge / species.mass
          x = @view positions(species)[1, :]
          y = @view positions(species)[2, :]
          vx = @view velocities(species)[1, :]
          vy = @view velocities(species)[2, :]
          vz = @view velocities(species)[3, :]
          xʷ = @view positions(species; work=true)[1, :]
          yʷ = @view positions(species; work=true)[2, :]
          vxʷ = @view velocities(species; work=true)[1, :]
          vyʷ = @view velocities(species; work=true)[2, :]
          vzʷ = @view velocities(species; work=true)[3, :]
          for i in species.chunks[j]
            Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(species.shape, x[i], y[i])
            xʷ[i] = unimod(x[i] + vx[i] * dt, Lx)
            yʷ[i] = unimod(y[i] + vy[i] * dt, Ly)
            vxʷ[i], vyʷ[i], vzʷ[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, Ezi,
              Bxi, Byi, Bzi, q_m);
            # now deposit ρ at (n+1)th timestep
            deposit!(ρJ⁺, species.shape, xʷ[i], yʷ[i], NX_Lx, NY_Ly,
              vxʷ[i] * qw_ΔV, vyʷ[i] * qw_ΔV,  vzʷ[i] * qw_ΔV)
          end
        end
      end
    end
    @timeit to "Field Reduction" begin
      reduction!(field.ρ⁺, field.Jx⁺, field.Jy⁺, field.Jz⁺, field.ρJs⁺)
      #field.ρJs⁺ .= 0 # dont zero it here!
    end
    @timeit to "Field Forward FT" begin
      field.ffthelper.pfft! * field.ρ⁺;
      field.ffthelper.pfft! * field.Jx⁺;
      field.ffthelper.pfft! * field.Jy⁺;
      field.ffthelper.pfft! * field.Jz⁺;
      field.ρ⁺[1, 1] = 0
      field.Jx⁺[1, 1] = 0
      field.Jy⁺[1, 1] = 0
      field.Jz⁺[1, 1] = 0
    end
    @timeit to "Field Solve" begin
      # at this point ϕ stores the nth timestep value and ϕ⁻ the (n-1)th
      lorenzgauge!(field.fieldimex, field.ϕ⁺, field.ϕ⁰,  field.ϕ⁻, field.ρ⁺, field.ρ⁰, field.ρ⁻, field.ffthelper.k², dt^2, field.sourceimex)
      lorenzgauge!(field.fieldimex, field.Ax⁺, field.Ax⁰, field.Ax⁻, field.Jx⁺, field.Jx⁰, field.Jx⁻, field.ffthelper.k², dt^2, field.sourceimex)
      lorenzgauge!(field.fieldimex, field.Ay⁺, field.Ay⁰, field.Ay⁻, field.Jy⁺, field.Jy⁰, field.Jy⁻, field.ffthelper.k², dt^2, field.sourceimex)
      lorenzgauge!(field.fieldimex, field.Az⁺, field.Az⁰, field.Az⁻, field.Jz⁺, field.Jz⁰, field.Jz⁻, field.ffthelper.k², dt^2, field.sourceimex)
    end
    @timeit to "Calculate E, B" begin
      θ = theta(field.fieldimex)
      @. field.Ex = -im * field.ffthelper.kx * (θ/2 * (field.ϕ⁺ + field.ϕ⁻) + (1-θ)*field.ϕ⁰)
      @. field.Ey = -im * field.ffthelper.ky * (θ/2 * (field.ϕ⁺ + field.ϕ⁻) + (1-θ)*field.ϕ⁰)
      @. field.Ex -= (field.Ax⁺ - field.Ax⁻)/2dt
      @. field.Ey -= (field.Ay⁺ - field.Ay⁻)/2dt
      @. field.Ez = -(field.Az⁺ - field.Az⁻)/2dt
      @. field.Bx =  im * field.ffthelper.ky * (θ/2 * (field.Az⁺ + field.Az⁻) + (1-θ)*field.Az⁰)
      @. field.By = -im * field.ffthelper.kx * (θ/2 * (field.Az⁺ + field.Az⁻) + (1-θ)*field.Az⁰)
      @. field.Bz =  im * field.ffthelper.kx * (θ/2 * (field.Ay⁺ + field.Ay⁻) + (1-θ)*field.Ay⁰)
      @. field.Bz -= im * field.ffthelper.ky * (θ/2 * (field.Ax⁺ + field.Ax⁻) + (1-θ)*field.Ax⁰)
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
  @timeit to "Copy over buffers" begin
    field.ρJs⁻ .= field.ρJs⁰
    field.ρJs⁰ .= field.ρJs⁺
    field.ϕ⁻ .= field.ϕ⁰
    field.ϕ⁰ .= field.ϕ⁺
    field.Ax⁻ .= field.Ax⁰
    field.Ax⁰ .= field.Ax⁺
    field.Ay⁻ .= field.Ay⁰
    field.Ay⁰ .= field.Ay⁺
    field.Az⁻ .= field.Az⁰
    field.Az⁰ .= field.Az⁺
  end
end



@inline function depositindicesfractions(s::AbstractShape, z::Float64, NZ::Int,
    NZ_Lz::Float64)
  zNZ = z * NZ_Lz # floating point position in units of cells
  # no need for unimod with offset arrays
  i = ceil(Int, zNZ) # cell number
  r = i - zNZ; # distance into cell i in units of cell width
  @assert 0 < r <= 1 "$z, $NZ, $NZ_Lz, $i"
  return gridinteractiontuple(s, i, r, NZ)
end

@inline gridinteractiontuple(::NGPWeighting, i, r, NZ) = ((i, 1), )
# no need for unimod with offset arrays
@inline function gridinteractiontuple(::AreaWeighting, i, r, NZ)
  return ((i, 1-r), (i+1, r))
end

bspline(::BSplineWeighting{@stat N}, x) where N = bspline(BSplineWeighting{Int(N)}(), x)

@inline bspline(::BSplineWeighting{0}, x) = ((1.0),)
@inline bspline(::BSplineWeighting{1}, x) = (x, 1-x)
function bspline(::BSplineWeighting{2}, x)
  @fastmath begin
    a = 9/8 + 3/2*(x-1.5) + 1/2*(x-1.5)^2
    b = 3/4               -     (x-0.5)^2
    c = 9/8 - 3/2*(x+0.5) + 1/2*(x+0.5)^2
  end
  return (a, b, c)
end
function bspline(::BSplineWeighting{3}, x)
  @fastmath begin
    a = 4/3 + 2*(x-2) + (x-2)^2 + 1/6*(x-2)^3
    b = 2/3           - (x-1)^2 - 1/2*(x-1)^3
    c = 2/3           - (x  )^2 + 1/2*(x  )^3
    d = 4/3 - 2*(x+1) + (x+1)^2 - 1/6*(x+1)^3
  end
  return (a, b, c, d)
end
function bspline(::BSplineWeighting{4}, x)
  @fastmath begin
    a = 625/384 + 125/48*(x-2.5) + 25/16*(x-2.5)^2 + 5/12*(x-2.5)^3 + 1/24*(x-2.5)^4
    b = 55/96   -   5/24*(x-1.5) -   5/4*(x-1.5)^2 -  5/6*(x-1.5)^3 -  1/6*(x-1.5)^4
    c = 115/192                  -   5/8*(x-0.5)^2                  +  1/4*(x-0.5)^4
    d = 55/96   +   5/24*(x+0.5) -   5/4*(x+0.5)^2 +  5/6*(x+0.5)^3 -  1/6*(x+0.5)^4
    e = 625/384 - 125/48*(x+1.5) + 25/16*(x+1.5)^2 - 5/12*(x+1.5)^3 + 1/24*(x+1.5)^4
  end
  return (a, b, c, d, e)
end
function bspline(::BSplineWeighting{5}, x)
  @fastmath begin
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

for N in 0:2:10
  @eval _bsplineinputs(::BSplineWeighting{@stat $(N+1)}, i, centre, ) = (i, 1 - centre)
  @eval function _bsplineinputs(::BSplineWeighting{@stat $N}, i, centre, )
    q = centre > 0.5
    return (i + q, q + 0.5 - centre)
  end
end

@inline function gridinteractiontuple(s::BSplineWeighting{N}, i, centre::T, NZ
    ) where {N,T}
#  (j, z) = if isodd(N)
#    (i, 1 - centre)
#  else
#    q = centre > 0.5
#    (i + q, q + 0.5 - centre)
#  end
  j, z = _bsplineinputs(s, i, centre)
  inds = indices(s, j)
  fractions = bspline(s, z)
  #@assert sum(fractions) ≈ 1 "$(sum(fractions)), $fractions"
  return zip(inds, fractions)
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

@inline function (f::AbstractLorenzGaugeField)(s::AbstractShape, xi, yi)
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

function deposit!(z::AbstractArray{<:Number, 2}, s::AbstractShape, x, y, NX_Lx, NY_Ly, w::Number)
  NX, NY = size(z)
  for (j, wy) in depositindicesfractions(s, y, NY, NY_Ly)
    for (i, wx) in depositindicesfractions(s, x, NX, NX_Lx)
      z[i,j] += wx * wy * w
    end
  end
end

function deposit!(z::AbstractArray{<:Number, 3}, s::AbstractShape, x, y, NX_Lx, NY_Ly, w1)
  NV, NX, NY = size(z)
  @assert NV == 4
  for (j, wy) in depositindicesfractions(s, y, NY, NY_Ly)
    for (i, wx) in depositindicesfractions(s, x, NX, NX_Lx)
      wxy = wx * wy
      @muladd z[1,i,j] = z[1,i,j] + wxy * w1
      #@muladd z[2,i,j] = z[2,i,j] + wxy * w2
      #@muladd z[3,i,j] = z[3,i,j] + wxy * w3
      #@muladd z[4,i,j] = z[4,i,j] + wxy * w4
    end
  end
end

function deposit!(z::AbstractArray{<:Number, 3}, s::AbstractShape, x, y, NX_Lx, NY_Ly, w2, w3, w4)
  NV, NX, NY = size(z)
  @assert NV == 4
  for (j, wy) in depositindicesfractions(s, y, NY, NY_Ly)
    for (i, wx) in depositindicesfractions(s, x, NX, NX_Lx)
      wxy = wx * wy
      #@muladd z[1,i,j] = z[1,i,j] + wxy * w1
      @muladd z[2,i,j] = z[2,i,j] + wxy * w2
      @muladd z[3,i,j] = z[3,i,j] + wxy * w3
      @muladd z[4,i,j] = z[4,i,j] + wxy * w4
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
    d.particlemomentum[ti] .= sum(momentum(s) for s in plasma)
    d.characteristicmomentum[ti] .= sum(characteristicmomentum(s) for s in plasma)
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

function diagnose!(d::LorenzGaugeDiagnostics, f::AbstractLorenzGaugeField, plasma,
    t, to)
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
          totenergy = (d.fieldenergy[ti] + d.kineticenergy[ti]) / (d.fieldenergy[1] + d.kineticenergy[1])
          @show ti, totenergy
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
        f.ffthelper.pifft! * f.Ax⁰
        f.ffthelper.pifft! * f.Ay⁰
        f.ffthelper.pifft! * f.Az⁰
        f.ffthelper.pifft! * f.ϕ⁰
        f.ffthelper.pifft! * f.ρ⁰
        f.ffthelper.pifft! * f.Jx⁰
        f.ffthelper.pifft! * f.Jy⁰
        f.ffthelper.pifft! * f.Jz⁰
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
        average!(d.Axs, f.Ax⁰)
        average!(d.Ays, f.Ay⁰)
        average!(d.Azs, f.Az⁰)
        average!(d.ϕs, f.ϕ⁰)
        average!(d.ρs, f.ρ⁰)
        average!(d.Jxs, f.Jx⁰)
        average!(d.Jys, f.Jy⁰)
        average!(d.Jzs, f.Jz⁰)
      end
      @timeit to "Field fft!" begin
        f.ffthelper.pfft! * f.Ax⁺
        f.ffthelper.pfft! * f.Ay⁺
        f.ffthelper.pfft! * f.Az⁺
        f.ffthelper.pfft! * f.ϕ⁺
        #f.ffthelper.pfft! * f.ρ⁺; # not necessary to transform back - they're overwritten
        #f.ffthelper.pfft! * f.Jx⁺;
        #f.ffthelper.pfft! * f.Jy⁺;
        #f.ffthelper.pfft! * f.Jz⁺;
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

function plotfields(d::AbstractDiagnostics, field, n0, vc, w0, NT; cutoff=Inf)
  B0 = norm(field.B0)
  dt = timestep(field)
  g = field.gridparams
  NXd = g.NX÷d.ngskip
  NYd = g.NY÷d.ngskip
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  xs = collect(1:NXd) ./ NXd ./ (vc / w0) * Lx
  ys = collect(1:NYd) ./ NYd ./ (vc / w0) * Ly
  ndiags = d.ti[]
  ts = collect(1:ndiags) .* ((NT * dt / ndiags) / (2pi/w0))

  filter = sin.((collect(1:ndiags) .- 0.5) ./ ndiags .* pi)'
  ws = 2π / (NT * dt) .* (1:ndiags) ./ (w0);

  kxs = 2π/Lx .* collect(0:NXd-1) .* (vc / w0);
  kys = 2π/Ly .* collect(0:NYd-1) .* (vc / w0);

  k0 = d.fieldenergy[1] + d.kineticenergy[1]

  plot(ts, d.fieldenergy, label="Fields")
  plot!(ts, d.kineticenergy, label="Particles")
  plot!(ts, d.fieldenergy + d.kineticenergy, label="Total")
  savefig("Energies.png")

  fieldmom = cat(d.fieldmomentum..., dims=2)'
  particlemom = cat(d.particlemomentum..., dims=2)'
  characteristicmom = cat(d.characteristicmomentum..., dims=2)'
  p0 = characteristicmom[1]
  plot(ts, fieldmom ./ p0, label="Fields")
  plot!(ts, characteristicmom ./ p0, label="Characteristic")
  plot!(ts, particlemom ./ p0, label="Particles")
  plot!(ts, (fieldmom .+ particlemom) ./ p0, label="Total")
  savefig("Momenta.png")

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
        xlabel!(L"Position x $[V_{A} / \Omega]$");
        ylabel!(L"Position y $[V_{A} / \Omega]$")
      end
      gif(anim, "PIC2D3V_$(FS)_XY.gif", fps=10)
    end

#    heatmap(xs, ys, F[:, :, 1])
#    xlabel!(L"Position x $[v_{A} / \Omega]$");
#    ylabel!(L"Position y $[v_{A} / \Omega]$")
#    savefig("PIC2D3V_$(FS)_XY_ic.png")
#
#    heatmap(xs, ys, F[:, :, end])
#    xlabel!(L"Position x $[v_{A} / \Omega]$");
#    ylabel!(L"Position y $[v_{A} / \Omega]$")
#    savefig("PIC2D3V_$(FS)_XY_final.png")

    Z = log10.(abs.(fft(F)[2:kxind, 1, 1:wind]))'
    heatmap(kxs[2:kxind], ws[1:wind], Z)
    xlabel!(L"Wavenumber x $[\Omega_c / V_{A}]$");
    ylabel!(L"Frequency $[\Omega_c]$")
    savefig("PIC2D3V_$(FS)_WKsumy_c.png")

    Z = log10.(abs.(fft(F)[1, 2:kyind, 1:wind]))'
    heatmap(kys[2:kyind], ws[1:wind], Z)
    xlabel!(L"Wavenumber y $[\Omega_c / V_{A}]$");
    ylabel!(L"Frequency $[\Omega_c]$")
    savefig("PIC2D3V_$(FS)_WKsumx_c.png")
  end

end

end


