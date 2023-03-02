module PIC2D3V

using FFTW,Plots, SpecialFunctions, StaticArrays, LinearAlgebra, Random
using LoopVectorization, Base.Threads, ThreadsX, Base.Iterators, Statistics
using ProgressMeter, LaTeXStrings, MuladdMacro, CommonSubexpressions
using TimerOutputs, StaticNumbers, OffsetArrays, FastPow, ThreadsX

unimod(x, n) = x > n ? x - n : x > 0 ? x : x + n

function applyperiodicity!(a::Array, oa)
  NX, NY = size(a)
  @assert length(size(a)) == 2
  @assert length(size(oa)) == 2
  @inbounds for j in axes(oa, 2), i in axes(oa, 1)
    a[mod1(i, NX), mod1(j, NY)] += oa[i, j]
  end
end

function applyperiodicity!(oa, a::Array)
  NX, NY = size(a)
  @assert length(size(a)) == 2
  @assert length(size(oa)) == 2
  @inbounds for j in axes(oa, 2), i in axes(oa, 1)
     oa[i, j] += real(a[mod1(i, NX), mod1(j, NY)])
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

abstract type AbstractBoris end
abstract type AbstractElectrostaticBoris end
abstract type AbstractElectromagneticBoris end

struct ElectrostaticBoris <: AbstractElectrostaticBoris
  t::SVector{3, Float64}
  t²::Float64
  dt_2::Float64
end
function ElectrostaticBoris(B::AbstractVector, dt::Float64)
  t = (@SArray [B[1], B[2], B[3]]) * dt / 2
  t² = dot(t, t)
  return ElectrostaticBoris(t, t², dt / 2)
end
function (boris::ElectrostaticBoris)(vx, vy, vz, _, Ex, Ey, q_m)
  dtq_2m = boris.dt_2 * q_m
  Ē₂ = (@SArray [Ex * dtq_2m, Ey * dtq_2m, 0.0])
  v⁻ = (@SArray [vx, vy, vz]) + Ē₂
  v⁺ = v⁻ + cross(v⁻ + cross(v⁻, boris.t), boris.t) * q_m^2 * 2 / (1 + q_m^2 * boris.t²)
  v = v⁺ + Ē₂
  return @SVector ([v[1], v[2], v[3], 1.0])
end

struct ElectromagneticBoris <: AbstractElectromagneticBoris
  dt_2::Float64
  ElectromagneticBoris(dt::Float64) = new(dt / 2)
end

function (boris::ElectromagneticBoris)(vx, vy, vz, _, Ex, Ey, Ez, Bx, By, Bz, q_m)
  θ = boris.dt_2 * q_m
  Ē₂ = (@SArray [Ex, Ey, Ez]) * θ
  v⁻ = Ē₂ + (@SArray [vx, vy, vz])
  t = (@SArray [Bx, By, Bz]) * θ
  tscale = 2 / (1 + dot(t, t))
  v⁺ = v⁻ + cross(v⁻ + cross(v⁻, t), t) * tscale
  v = v⁺ + Ē₂
  return @SVector ([v[1], v[2], v[3], 1.0])
end


struct RelativisticElectromagneticBoris <: AbstractElectromagneticBoris
  dt_2::Float64
  ElectromagneticBoris(dt::Float64) = new(dt / 2)
end

function (boris::RelativisticElectromagneticBoris)(vx, vy, vz, γ, Ex, Ey, Ez, Bx, By, Bz, q_m)
  θ = boris.dt_2 * q_m
  Ē₂ = (@SArray [Ex, Ey, Ez]) * θ
  u⁻ = Ē₂ + (@SArray [vx, vy, vz]) * γ
  invγ⁻ = 1 / sqrt(1 + dot(u⁻, u⁻))
  t = (@SArray [Bx, By, Bz]) * θ * invγ⁻
  tscale = 2 / (1 + dot(t, t))
  u⁺ = u⁻ + cross(u⁻ + cross(u⁻, t), t) * tscale
  u = u⁺ + Ē₂
  γ⁻¹ = 1 / sqrt(1 + dot(u, u))
  return @SVector ([u[1] * γ⁻¹, u[2] * γ⁻¹, u[3] * γ⁻¹, γ])
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
  xyvγ::Matrix{Float64}
  p::Vector{Int}
  chunks::Vector{UnitRange{Int}}
  xyvγwork::Matrix{Float64}
end
function positions(s::Species; work=false)
  return work ? (@view s.xyvγwork[1:2, :]) : (@view s.xyvγ[1:2, :])
end
function velocities(s::Species; work=false)
  return work ? (@view s.xyvγwork[3:5, :]) : (@view s.xyvγ[3:5, :])
end
function gammas(s::Species; work=false)
  return work ? (@view s.xyvγwork[6, :]) : (@view s.xyvγ[6, :])
end

copyto!(srcdest::Species) = (@turbo srcdest.xyvγ .= srcdest.xyvγwork)
function copyto!(dest::Species, src::Species)
  @turbo dest.xyvγ .= src.xyvγ
  #@tturbo dest.p .= src.p
  return dest
end

kineticenergy(s::Species) = sum(abs2, velocities(s)) * s.mass / 2 * s.weight

momentum(s::Species) = sum(velocities(s), dims=2)[:] * s.mass * s.weight

calculateweight(n0, P) = n0 / P;

function Species(P, vth, density, shape::AbstractShape; Lx, Ly, charge=1, mass=1)
  x  = Lx * rand(P);#halton.(0:P-1, 2, 1/sqrt(2));#
  y  = Ly * rand(P);#halton.(0:P-1, 3, 1/sqrt(2));#
  vx = vth * erfinv.(2rand(P) .- 1) * vth;#erfinv.(2halton.(0:P-1,  5, 1/sqrt(2)) .- 1);#rand(P));
  vy = vth * erfinv.(2rand(P) .- 1) * vth;#erfinv.(2halton.(0:P-1,  7, 1/sqrt(2)) .- 1);#rand(P));
  vz = vth * erfinv.(2rand(P) .- 1) * vth;#erfinv.(2halton.(0:P-1, 11, 1/sqrt(2)) .- 1);#rand(P));;
  vx .-= mean(vx)
  vy .-= mean(vy)
  vz .-= mean(vz)
  vx .*= (vth / sqrt(2)) / std(vx);
  vy .*= (vth / sqrt(2)) / std(vy);
  vz .*= (vth / sqrt(2)) / std(vz);
  γ = @. 1 / sqrt(1 - (vx^2 + vy^2 + vz^2))
  p  = collect(1:P)
  xyvγ = Matrix(hcat(x, y, vx, vy, vz, γ)')
  chunks = collect(Iterators.partition(1:P, ceil(Int, P/nthreads())))
  weight = calculateweight(density, P)
  return Species(Float64(charge), Float64(mass), weight, shape, xyvγ, p, chunks, deepcopy(xyvγ))
end

function Base.sort!(s::Species, Δx, Δy)
  sortperm!(s.p, eachindex(s.p),
    by=i->(ceil(Int, s.xyvγ[1,i] / Δx), ceil(Int, s.xyvγ[2,i] / Δy), s.xyvγ[3,i]))
  s.xyvγ .= s.xyvγ[:, s.p]
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


#struct ElectrostaticField{N, A<:AbstractArray{Float64, N}, T} <: AbstractField
struct ElectrostaticField{N, A<:AbstractArray{Float64, N}, T} <: AbstractField
  Exy::OffsetArray{Float64, 3, Array{Float64, 3}} # offset array
  ρs::OffsetArray{Float64, 3, Array{Float64, 3}} # offset array
  ϕ::Array{ComplexF64, 2}
  Ex::Array{ComplexF64, 2}
  Ey::Array{ComplexF64, 2}
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
  return ElectrostaticField(Exy, ρs, (zeros(ComplexF64, NX, NY) for _ in 1:3)...,
    Float64.((B0x, B0y, B0z)), gps, ffthelper, boris)
end

function update!(f::ElectrostaticField)
  #@sync begin
  t1 = @spawn applyperiodicity!((@view f.Exy[1, :, :]), f.Ex)
  t2 = @spawn applyperiodicity!((@view f.Exy[2, :, :]), f.Ey)
  #end
  wait(t1); wait(t2)
end

abstract type AbstractImEx end
struct Explicit <: AbstractImEx end
struct Implicit <: AbstractImEx end
struct ImEx <: AbstractImEx
  θ::Float64
end

#struct LorenzGaugeStaggeredField{N, A<:AbstractArray{Float64, N},
#    N1, B<:AbstractArray{Float64, N1}, T, U} <: AbstractLorenzGaugeField
struct LorenzGaugeStaggeredField{T, U} <: AbstractLorenzGaugeField
  EBxyz::OffsetArray{Float64, 3, Array{Float64, 3}}
  ρJs⁺::OffsetArray{Float64, 4, Array{Float64, 4}}
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
  B0::NTuple{3, Float64}
  imex::T
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
  return LorenzGaugeStaggeredField(EBxyz, ρJs,
    (zeros(ComplexF64, NX, NY) for _ in 1:30)..., 
    Float64.((B0x, B0y, B0z)), imex, gps, ffthelper, boris, dt)
end



#struct LorenzGaugeSemiImplicitField{N, A<:AbstractArray{Float64, N},
#    N1, B<:AbstractArray{Float64, N1}, T, U, V} <: AbstractLorenzGaugeField
struct LorenzGaugeSemiImplicitField{T, U, V} <: AbstractLorenzGaugeField
  EBxyz::OffsetArray{Float64, 3, Array{Float64, 3}}
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
  B0::NTuple{3, Float64}
  fieldimex::T
  sourceimex::U
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
  return LorenzGaugeSemiImplicitField(EBxyz, ρJs, deepcopy(ρJs), deepcopy(ρJs),
    deepcopy(ρJs), (zeros(ComplexF64, NX, NY) for _ in 1:30)..., 
    Float64.((B0x, B0y, B0z)), fieldimex, sourceimex, gps, ffthelper, boris, dt, rtol, maxiters)
end

theta(::Explicit) = 0
theta(imex::ImEx) = imex.θ
theta(::Implicit) = 1

function update!(f::AbstractLorenzGaugeField)
  f.EBxyz .= 0.0
  t1 = @spawn applyperiodicity!((@view f.EBxyz[1, :, :]), f.Ex)
  t2 = @spawn applyperiodicity!((@view f.EBxyz[2, :, :]), f.Ey)
  t3 = @spawn applyperiodicity!((@view f.EBxyz[3, :, :]), f.Ez)
  t4 = @spawn applyperiodicity!((@view f.EBxyz[4, :, :]), f.Bx)
  t5 = @spawn applyperiodicity!((@view f.EBxyz[5, :, :]), f.By)
  t6 = @spawn applyperiodicity!((@view f.EBxyz[6, :, :]), f.Bz)
  wait(t4); wait(t5); wait(t6);
  @inbounds for k in axes(f.EBxyz, 3), j in axes(f.EBxyz, 2), i in 1:3
    f.EBxyz[i+3, j, k] += f.B0[i]
  end
  wait(t1); wait(t2); wait(t3);
end

function reduction!(a, z)
  @. a = 0.0
  @inbounds for k in axes(z, 3)
    applyperiodicity!(a, (@view z[:, :, k]))
  end
end

function reduction!(a, b, c, z)
  @assert size(z, 1) == 4
  #@sync begin
  task1 = @spawn begin
    @. a = 0.0
    @views for k in axes(z, 4)
      applyperiodicity!(a, z[1, :, :, k])
    end
  end
  task2 = @spawn begin
    @. b = 0.0
    @views for k in axes(z, 4)
      applyperiodicity!(b, z[2, :, :, k])
    end
  end
  task3 = @spawn begin
    @. c = 0.0
    @views for k in axes(z, 4)
      applyperiodicity!(c, z[3, :, :, k])
    end
  end
  #end
  wait.(vcat(task1, task2, task3))
end

function reduction!(a, b, c, d, z)
  @assert size(z, 1) == 4
#@sync begin
  task1 = @spawn begin
    @. a = 0.0
    @views for k in axes(z, 4)
      applyperiodicity!(a, z[1, :, :, k])
    end
  end
  task2 = @spawn begin
    @. b = 0.0
    @views for k in axes(z, 4)
      applyperiodicity!(b, z[2, :, :, k])
    end
  end
  task3 = @spawn begin
  @. c = 0.0
  @views for k in axes(z, 4)
    applyperiodicity!(c, z[3, :, :, k])
  end
  end
  task4 = @spawn begin
    @. d = 0.0
    @views for k in axes(z, 4)
     applyperiodicity!(d, z[4, :, :, k])
    end
  end
#end
  wait.(vcat(task1, task2, task3, task4))
end

function particleloop!(plasma, field::ElectrostaticField, dt)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  ΔV = cellvolume(field.gridparams)
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
      γ = gammas(species)
      for i in species.chunks[k]
        Exi, Eyi = field(species.shape, x[i], y[i])
        vxi, vyi = vx[i], vy[i]
        vx[i], vy[i], vz[i], γ[i] = field.boris(vx[i], vy[i], vz[i], γ[i], Exi, Eyi, q_m);
        x[i] = unimod(x[i] + (vxi + vx[i])/2*dt, Lx)
        y[i] = unimod(y[i] + (vyi + vy[i])/2*dt, Ly)
        deposit!(ρ, species.shape, x[i], y[i], NX_Lx, NY_Ly, qw_ΔV)
      end
    end
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
  @timeit to "Particle Loop" particleloop!(plasma, field, dt)
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

function warmup!(field::LorenzGaugeStaggeredField, plasma, to)
  @timeit to "Warmup" begin
    dt = timestep(field)
    warmup!(field.ρ⁻, field.Jx⁻, field.Jy⁻, field.Jz⁻, field.ρJs⁺, plasma, field.gridparams, -dt, to)
    warmup!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.ρJs⁺, plasma, field.gridparams, 0, to)
    warmup!(field.ρ⁺, field.Jx⁺, field.Jy⁺, field.Jz⁺, field.ρJs⁺, plasma, field.gridparams, dt, to)
  end
end

function warmup!(field::LorenzGaugeSemiImplicitField, plasma, to)
  @timeit to "Warmup" begin
    dt = timestep(field)
    warmup!(field.ρ⁻, field.Jx⁻, field.Jy⁻, field.Jz⁻, field.ρJs⁻, plasma, field.gridparams, -dt, to)
    warmup!(field.ρ⁰, field.Jx⁰, field.Jy⁰, field.Jz⁰, field.ρJs⁰, plasma, field.gridparams, 0, to)
    warmup!(field.ρ⁺, field.Jx⁺, field.Jy⁺, field.Jz⁺, field.ρJs⁺, plasma, field.gridparams, dt, to)
  end
end

function particleloop!(plasma, field::LorenzGaugeStaggeredField, dt)
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  ΔV = cellvolume(field.gridparams)
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
      γ = gammas(species)
      #  E.....E.....E
      #  B.....B.....B
      #  ϕ.....ϕ.....ϕ
      #  -..A..0..A..+..A
      #  ρ.....ρ.....ρ
      #  -..J..0..J..+..J
      #  x.....x.....x
      #  -..v..0..v..+..v
      for i in species.chunks[j]
        Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(species.shape, x[i], y[i])
        vx[i], vy[i], vz[i], γ[i] = field.boris(vx[i], vy[i], vz[i], γ[i], Exi, Eyi, Ezi,
          Bxi, Byi, Bzi, q_m);
        x[i] = unimod(x[i] + vx[i] * dt/2, Lx)
        y[i] = unimod(y[i] + vy[i] * dt/2, Ly)
        # deposit J at the (n+1/2)th point
        deposit!(ρJ⁺, species.shape, x[i], y[i], NX_Lx, NY_Ly,
          vx[i] * qw_ΔV, vy[i] * qw_ΔV, vz[i] * qw_ΔV)
        x[i] = unimod(x[i] + vx[i] * dt/2, Lx)
        y[i] = unimod(y[i] + vy[i] * dt/2, Ly)
        # now deposit ρ at (n+1)th timestep
        deposit!(ρJ⁺, species.shape, x[i], y[i], NX_Lx, NY_Ly, qw_ΔV)
      end
    end
  end
end

# E = -∇ϕ - ∂ₜA
# B = ∇xA
# ∇² A - 1/c^2 ∂ₜ² A = -μ₀ J⁰
# ∇² ϕ - 1/c^2 ∂ₜ² ϕ = -ρ⁰ / ϵ₀
# im^2 k^2 * ϕ - 1/dt^2/c^2 (ϕ⁺ - 2ϕ⁰ + ϕ⁻) =  -ρ / ϵ₀
function loop!(plasma, field::LorenzGaugeStaggeredField, to, t, _)
  dt = timestep(field)
  @timeit to "Particle Loop" particleloop!(plasma, field, dt)
  @timeit to "Field Reduction" begin
    reduction!(field.ρ⁺, field.Jx⁺, field.Jy⁺, field.Jz⁺, field.ρJs⁺)
    field.ρJs⁺ .= 0
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
    lorenzgauge!(field.imex, field.ϕ⁺,  field.ϕ⁰,  field.ϕ⁻,  field.ρ⁺,  field.ρ⁰, field.ρ⁻, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ax⁺, field.Ax⁰, field.Ax⁻, field.Jx⁺, field.Jx⁰, field.Jx⁻, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Ay⁺, field.Ay⁰, field.Ay⁻, field.Jy⁺, field.Jy⁰, field.Jy⁻, field.ffthelper.k², dt^2)
    lorenzgauge!(field.imex, field.Az⁺, field.Az⁰, field.Az⁻, field.Jz⁺, field.Jz⁰, field.Jz⁻, field.ffthelper.k², dt^2)
  end
  # at this point (ϕ, Ai) stores the (n+1)th timestep value and (ϕ⁻, Ai⁻) the nth
  # Now calculate the value of E and B at n+1/2
  # Eʰ = -∇ ϕ⁺ - (A⁺ - A⁰)/dt
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
function particleloop!(plasma, field::LorenzGaugeSemiImplicitField, dt, plasmacopy)
  copyto!.(plasma, plasmacopy)
  @turbo field.ρJsᵗ .= field.ρJs⁺
  @turbo field.ρJs⁺ .= 0
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  NX_Lx, NY_Ly = field.gridparams.NX_Lx, field.gridparams.NY_Ly
  ΔV = cellvolume(field.gridparams)
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
      γ = gammas(species)
      xʷ = @view positions(species; work=true)[1, :]
      yʷ = @view positions(species; work=true)[2, :]
      vxʷ = @view velocities(species; work=true)[1, :]
      vyʷ = @view velocities(species; work=true)[2, :]
      vzʷ = @view velocities(species; work=true)[3, :]
      @inbounds for i in species.chunks[j]
        Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(species.shape, x[i], y[i])
        xʷ[i] = unimod(x[i] + vx[i] * dt, Lx)
        yʷ[i] = unimod(y[i] + vy[i] * dt, Ly)
        vxʷ[i], vyʷ[i], vzʷ[i], _ = field.boris(vx[i], vy[i], vz[i], γ[i], Exi, Eyi, Ezi,
          Bxi, Byi, Bzi, q_m);
        # now deposit ρ at (n+1)th timestep
        deposit!(ρJ⁺, species.shape, xʷ[i], yʷ[i], NX_Lx, NY_Ly,
          vxʷ[i] * qw_ΔV, vyʷ[i] * qw_ΔV,  vzʷ[i] * qw_ΔV)
      end
    end
  end
end

function loop!(plasma, field::LorenzGaugeSemiImplicitField, to, t, plasmacopy = deepcopy(plasma))
  dt = timestep(field)
  copyto!.(plasmacopy, plasma)
  firstloop = true
  iters = 0
  while true
    if (iters > 0) && (iters > field.maxiters ||
        isapprox(sum(abs2, field.ρJsᵗ), sum(abs2, field.ρJs⁺), rtol=field.rtol, atol=0))
      copyto!.(plasma)
      break
    end
    iters += 1
    @timeit to "Particle Loop" particleloop!(plasma, field, dt, plasmacopy)
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
#  deposit!(z, s, x, y, NX_Lx, NY_Ly, 1:1, (w1, ))
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
#  deposit!(z, s, x, y, NX_Lx, NY_Ly, 2:4, (0, w2, w3, w4))
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
  #deposit!(z, s, x, y, NX_Lx, NY_Ly, 1:4, (w1, w2, w3, w4))
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

function deposit!(Z, X, Y, V::AbstractMatrix, s::AbstractShape, NX_Lx::Float64, NY_Ly::Float64)
  NV, NX, NY = size(z)
  @assert NV == 4
  for p in eahcindex(X, Y)
    for (j, wy) in depositindicesfractions(s, Y[p], NY, NY_Ly)
      for (i, wx) in depositindicesfractions(s, X[p], NX, NX_Lx)
        wxy = wx * wy
        for h in hs
          @inbounds z[h,i,j] += wxy * V[h, p]
        end
      end
    end
  end
end


function diagnose!(d::AbstractDiagnostics, plasma, to)
  @timeit to "Plasma" begin
    ti = d.ti[]
    d.kineticenergy[ti] = sum(kineticenergy(s) for s in plasma)
    d.particlemomentum[ti] .= sum(momentum(s) for s in plasma)
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
          ndiag = length(d.kineticenergy)
          totenergy = (d.fieldenergy[ti] + d.kineticenergy[ti]) / (d.fieldenergy[1] + d.kineticenergy[1])
          ndiag = length(d.kineticenergy)
          @show ndiag, d.ti, totenergy
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
        f.ffthelper.pifft! * f.Ax⁺
        f.ffthelper.pifft! * f.Ay⁺
        f.ffthelper.pifft! * f.Az⁺
        f.ffthelper.pifft! * f.ϕ⁺
        f.ffthelper.pifft! * f.ρ⁺;
        f.ffthelper.pifft! * f.Jx⁺;
        f.ffthelper.pifft! * f.Jy⁺;
        f.ffthelper.pifft! * f.Jz⁺;
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
        average!(d.Axs, f.Ax⁺)
        average!(d.Ays, f.Ay⁺)
        average!(d.Azs, f.Az⁺)
        average!(d.ϕs, f.ϕ⁺)
        average!(d.ρs, f.ρ⁺)
        average!(d.Jxs, f.Jx⁺)
        average!(d.Jys, f.Jy⁺)
        average!(d.Jzs, f.Jz⁺)
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

  fieldmom = cat(d.fieldmomentum..., dims=2)'
  particlemom = cat(d.particlemomentum..., dims=2)'
  plot(ts, fieldmom, label="Fields")
  plot!(ts, particlemom, label="Particles")
  plot!(ts, fieldmom .+ particlemom, label="Total")
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
        xlabel!(L"Position x $[v_{th} / \Omega]$");
        ylabel!(L"Position y $[v_{th} / \Omega]$")
      end
      gif(anim, "PIC2D3V_$(FS)_XY.gif", fps=10)
    end

    heatmap(xs, ys, F[:, :, 1])
    xlabel!(L"Position x $[v_{th} / \Omega]$");
    ylabel!(L"Position y $[v_{th} / \Omega]$")
    savefig("PIC2D3V_$(FS)_XY_ic.png")

    heatmap(xs, ys, F[:, :, end])
    xlabel!(L"Position x $[v_{th} / \Omega]$");
    ylabel!(L"Position y $[v_{th} / \Omega]$")
    savefig("PIC2D3V_$(FS)_XY_final.png")

    Z = log10.(abs.(fft(F)[2:kxind, 1, 1:wind]))'
    heatmap(kxs[2:kxind], ws[1:wind], Z)
    xlabel!(L"Wavenumber x $[\Omega_c / v_{th}]$");
    ylabel!(L"Frequency $[\Omega_c]$")
    savefig("PIC2D3V_$(FS)_WKsumy_c.png")
    xlabel!(L"Wavenumber x $[\Pi / v_{th}]$");
    ylabel!(L"Frequency $[\Pi]$")
    heatmap(kxs[2:kxind] .* w0 / sqrt(n0), ws[1:wind] .* w0 / sqrt(n0), Z)
    savefig("PIC2D3V_$(FS)_WKsumy_p.png")
   
    Z = log10.(abs.(fft(F)[1, 2:kyind, 1:wind]))'
    heatmap(kys[2:kyind], ws[1:wind], Z)
    xlabel!(L"Wavenumber y $[\Omega_c / v_{th}]$");
    ylabel!(L"Frequency $[\Omega_c]$")
    savefig("PIC2D3V_$(FS)_WKsumx_c.png")
    heatmap(kys[2:kyind] .* w0 / sqrt(n0), ws[1:wind] .* w0 / sqrt(n0), Z)
    xlabel!(L"Wavenumber y $[\Pi / v_{th}]$");
    ylabel!(L"Frequency $[\Pi]$")
    savefig("PIC2D3V_$(FS)_WKsumx_p.png")
  end

end

end


