module PIC2D3V

using FFTW,Plots, SpecialFunctions, StaticArrays, LinearAlgebra, Random
using LoopVectorization, Base.Threads, ThreadsX, Base.Iterators, Statistics
using ProgressMeter, LaTeXStrings, MuladdMacro, CommonSubexpressions
using TimerOutputs

FFTW.set_num_threads(Threads.nthreads())
Plots.gr()

Random.seed!(0)

unimod(x, n) = 0 < x <= n ? x : x > n ? x - n : x + n

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
  Exs::Vector{Array{Float64, 2}}
  Eys::Vector{Array{Float64, 2}}
  ϕs::Vector{Array{Float64, 2}}
  nskip::Int
  ti::Ref{Int64}
  makegifs::Bool
end

function generatestorage(NX, NY, ND, nstorage)
  scalarstorage = (zeros(ND) for _ in 1:2)
  momentumstorage = [zeros(3) for _ in 1:ND]
  fieldstorage = ([zeros(NX, NY) for _ in 1:ND] for _ in 1:nstorage)
  return (scalarstorage, momentumstorage, fieldstorage)
end

function ElectrostaticDiagnostics(NX, NY, NT, NS; makegifs=false)
  @assert NT >= NS
  scalarstorage, momentumstorage, fieldstorage = generatestorage(NX, NY, NT÷NS, 3)
  return ElectrostaticDiagnostics(scalarstorage..., momentumstorage,
    fieldstorage..., NS, Ref(0), makegifs)
end

struct LorenzGuageDiagnostics <: AbstractDiagnostics
  kineticenergy::Array{Float64, 1}
  fieldenergy::Array{Float64, 1}
  particlemomentum::Vector{Vector{Float64}}
  Exs::Vector{Array{Float64, 2}}
  Eys::Vector{Array{Float64, 2}}
  Ezs::Vector{Array{Float64, 2}}
  Bxs::Vector{Array{Float64, 2}}
  Bys::Vector{Array{Float64, 2}}
  Bzs::Vector{Array{Float64, 2}}
  Axs::Vector{Array{Float64, 2}}
  Ays::Vector{Array{Float64, 2}}
  Azs::Vector{Array{Float64, 2}}
  ϕs::Vector{Array{Float64, 2}}
  nskip::Int
  ti::Ref{Int64}
  makegifs::Bool
end

function LorenzGuageDiagnostics(NX, NY, NT::Int, NS::Int; makegifs=false)
  @assert NT >= NS
  scalarstorage, momentumstorage, fieldstorage = generatestorage(NX, NY, NT÷NS, 10)
  return LorenzGuageDiagnostics(scalarstorage..., momentumstorage,
    fieldstorage..., NS, Ref(0), makegifs)
end


abstract type AbstractShape end
struct AreaWeighting <: AbstractShape end
struct NGPWeighting <: AbstractShape end


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

function kineticenergy(s::Species)
  return mapreduce(u->u^2, +, velocities(s)) * s.mass / 2 * s.weight
end

momentum(s::Species) = sum(velocities(s), dims=2)[:] * s.mass * s.weight

calculateweight(s::AreaWeighting, n0, P) = n0 / P;
calculateweight(s::NGPWeighting, n0, P) = n0 / P;

function Species(P, vth, density, shape::AbstractShape; Lx, Ly,
    charge=1, mass=1)
  x  = Lx * rand(P);#        halton.(0:P-1, 2, 1/sqrt(2));#rand(P);
  y  = Ly * rand(P);#        halton.(0:P-1, 3, 1/sqrt(2));#rand(P);
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
  xyv = Matrix(hcat(x, y, vx, vy, vz)')
  chunks = collect(Iterators.partition(1:P, ceil(Int, P/nthreads())))
  weight = calculateweight(shape, density, P)
  return Species(Float64(charge), Float64(mass), weight, shape, xyv, p, chunks)
end

function Base.sort!(s::Species, Δx, Δy)
  sortperm!(s.p, eachindex(s.p),
    by=x->(ceil(Int, s.xyv[1] / Δx), ceil(Int, s.xyz[2] / Δy)))
  s.xyv .= s.xyv[:, p]
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
end

cellvolume(g::GridParameters) = g.ΔX * g.ΔY

struct FFTHelper{T, U}
  kx::Vector{Float64}
  ky::LinearAlgebra.Adjoint{Float64, Vector{Float64}}
  k²::Matrix{Float64}
  negative_im_k⁻²::Matrix{ComplexF64}
  pfft::T
  pifft::U
end
function FFTHelper(NX, NY, Lx, Ly)
  kx=2π/Lx*vcat(0:NX÷2-1,-NX÷2:-1);
  ky=2π/Ly*vcat(0:NY÷2-1,-NY÷2:-1)';
  kk = -(kx.^2 .+ ky.^2)
  negative_im_k⁻²= im ./ kk
  negative_im_k⁻²[1, 1] = 0
  z = zeros(ComplexF64, NX, NY)
  pfft = plan_fft!(z; flags=FFTW.ESTIMATE, timelimit=Inf)
  pifft = plan_ifft!(z; flags=FFTW.ESTIMATE, timelimit=Inf)
  return FFTHelper(kx, ky, kk, negative_im_k⁻², pfft, pifft)
end


struct ElectrostaticField{T} <: AbstractField
  ρs::Array{Float64, 3}
  ϕ::Array{ComplexF64, 2}
  Ex::Array{ComplexF64, 2}
  Ey::Array{ComplexF64, 2}
  Exy::Array{Float64, 3}
  B0::NTuple{3, Float64}
  gridparams::GridParameters
  ffthelper::T
  boris::ElectrostaticBoris
end

function ElectrostaticField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0)
  ρs=zeros(NX, NY, nthreads())
  Exy=zeros(2, NX, NY);
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  gps = GridParameters(Lx, Ly, NX, NY, 1/NX, 1/NY)
  boris = ElectrostaticBoris([B0x, B0y, B0z], dt)
  return ElectrostaticField(ρs, (zeros(ComplexF64, NX, NY) for _ in 1:3)...,
    Exy, Float64.((B0x, B0y, B0z)), gps, ffthelper, boris)
end

function update!(f::ElectrostaticField)
  f.Exy[1, :, :] .= real.(f.Ex)
  f.Exy[2, :, :] .= real.(f.Ey)
end

struct LorenzGuageField{T} <: AbstractField
  ρJs::Array{Float64, 4}
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
  EBxyz::Array{Float64, 3}
  B0::NTuple{3, Float64}
  gridparams::GridParameters
  ffthelper::T
  boris::ElectromagneticBoris
end

function LorenzGuageField(NX, NY=NX, Lx=1, Ly=1; dt, B0x=0, B0y=0, B0z=0)
  EBxyz=zeros(6, NX, NY);
  gps = GridParameters(Lx, Ly, NX, NY, 1/NX, 1/NY)
  ffthelper = FFTHelper(NX, NY, Lx, Ly)
  boris = ElectromagneticBoris(dt)
  return LorenzGuageField(zeros(4, NX, NY, nthreads()),
    (zeros(ComplexF64, NX, NY) for _ in 1:18)..., EBxyz,
    Float64.((B0x, B0y, B0z)), gps, ffthelper, boris)
end

function update!(f::LorenzGuageField)
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
    for h in 1:4
      z[h, i, j, k] = 0.0
    end
  end
end


#E = -∇ ϕ
#∇ . E = -∇.∇ ϕ = -∇^2 ϕ = ρ
#-i^2 (kx^2 + ky^2) ϕ = ρ
#ϕ = ρ / (kx^2 + ky^2)
# Ex = - ∇_x ϕ = - i kx ϕ = - i kx ρ / (kx^2 + ky^2)
# Ey = - ∇_y ϕ = - i ky ϕ = - i ky ρ / (kx^2 + ky^2)
function loop!(plasma, field::ElectrostaticField, to)
  dt = timestep(field)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  ΔV = cellvolume(field.gridparams)
  @timeit to "Particle Loop" begin
    @threads for k in axes(field.ρs, 3)
      ρ = @view field.ρs[:, :, k]
      for species in plasma
        qw_ΔV = species.charge * species.weight / ΔV
        q_m = species.charge * species.mass
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        @inbounds for i in species.chunks[k]
          Exi, Eyi = field(species.shape, x[i], y[i])
          vx[i], vy[i], vz[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, q_m);
          x[i] = unimod(x[i] + vx[i]*dt, Lx)
          y[i] = unimod(y[i] + vy[i]*dt, Ly)
          deposit!(ρ, species.shape, x[i], y[i], Lx, Ly, qw_ΔV)
        end
      end
    end
  end
  @timeit to "Field reduction" begin
    reduction!(field.ϕ, field.ρs)
  end
  @timeit to "Field Forward FT" begin
    field.ffthelper.pfft * field.ϕ;
    field.ϕ[1, 1] = 0
    @threads for j in axes(field.ϕ, 2)
      for i in axes(field.ϕ, 1)
        tmp = field.ϕ[i, j] * field.ffthelper.negative_im_k⁻²[i, j]
        @assert isfinite(tmp)
        field.Ex[i, j] = tmp * field.ffthelper.kx[i]
        field.Ey[i, j] = tmp * field.ffthelper.ky[j]
      end
    end
  end
  @timeit to "Field Inverse FT" begin
    field.ffthelper.pifft * field.Ex
    field.ffthelper.pifft * field.Ey
  end
  @timeit to "Field update" update!(field)
end

function lorenzguage!(xⁿ, xⁿ⁻¹, sⁿ, k², dt)
  @inbounds @threads for i in eachindex(xⁿ)
    xⁿ⁺¹ = 2xⁿ[i] - xⁿ⁻¹[i] + (sⁿ[i] + k²[i] * xⁿ[i]) * dt^2
    xⁿ⁻¹[i] = xⁿ[i]
    xⁿ[i] = xⁿ⁺¹
  end
end

# E = -∇ϕ - ∂ₜA
# B = ∇xA
# ∇² A - 1/c^2 ∂ₜ² A = -μ₀ J⁰
# ∇² ϕ - 1/c^2 ∂ₜ² ϕ = -ρ⁰ / ϵ₀
# im^2 k^2 * ϕ - dt^2/c^2 (ϕ⁺ - 2ϕ⁰ + ϕ⁻) =  -ρ / ϵ₀
# 1) Calculate ϕ⁺ and A⁺ (ϕ and A at next timestep, n+1)
# ϕ⁺ = 2ϕ⁰ - ϕ⁻ + (ρ / ϵ₀ - k^2 * ϕ⁰)*c^2/dt^2
# A⁺ = 2A⁰ - A⁻ + (J μ₀ - k^2 * A⁰)*c^2/dt^2
# 2) calculate the half timstep n+1/2
# Eʰ = -∇(ϕ⁰ + ϕ⁺) / 2 - (A⁺ - A⁰)/dt
# Bʰ = ∇x(A⁺ + A⁰)/2
# 3) push particles from n to n+1 with fields at n+1/2
# push x⁰ to xʰ with v⁰
# push v⁰ to v⁺ with Eʰ and Bʰ
# push xʰ to x⁺ with v⁺
# 4) copy fields into buffers for next loop
function loop!(plasma, field::LorenzGuageField, to)
  dt = timestep(field)
  Lx, Ly = field.gridparams.Lx, field.gridparams.Ly
  ΔV = cellvolume(field.gridparams)
  @timeit to "Particle loop" begin
    @threads for j in axes(field.ρJs, 4)
      ρJ = @view field.ρJs[:, :, :, j]
      for species in plasma
        qw_ΔV = species.charge * species.weight / ΔV
        q_m = species.charge * species.mass
        x = @view positions(species)[1, :]
        y = @view positions(species)[2, :]
        vx = @view velocities(species)[1, :]
        vy = @view velocities(species)[2, :]
        vz = @view velocities(species)[3, :]
        @inbounds for i in species.chunks[j]
          Exi, Eyi, Ezi, Bxi, Byi, Bzi = field(species.shape, x[i], y[i])
          vx[i], vy[i], vz[i] = field.boris(vx[i], vy[i], vz[i], Exi, Eyi, Ezi, Bxi,
                                      Byi, Bzi, q_m);
          x[i] = unimod(x[i] + vx[i]*dt, Lx)
          y[i] = unimod(y[i] + vy[i]*dt, Ly)
          deposit!(ρJ, species.shape, x[i], y[i], Lx, Ly, qw_ΔV, vx[i] * qw_ΔV, vy[i] * qw_ΔV, vz[i] * qw_ΔV)
        end
      end
    end
  end
  @timeit to "Field reduction" begin
    reduction!(field.ρ, field.Jx, field.Jy, field.Jz, field.ρJs)
  end
  @timeit to "Field invert" begin
    field.ffthelper.pfft * field.ρ;
    field.ffthelper.pfft * field.Jx;
    field.ffthelper.pfft * field.Jy;
    field.ffthelper.pfft * field.Jz;
    field.ρ[1, 1] = 0
    field.Jx[1, 1] = 0
    field.Jy[1, 1] = 0
    field.Jz[1, 1] = 0
    lorenzguage!(field.ϕ, field.ϕ⁻, field.ρ, field.ffthelper.k², dt)
    lorenzguage!(field.Ax, field.Ax⁻, field.Jx, field.ffthelper.k², dt)
    lorenzguage!(field.Ay, field.Ay⁻, field.Jy, field.ffthelper.k², dt)
    lorenzguage!(field.Az, field.Az⁻, field.Jz, field.ffthelper.k², dt)
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
  @timeit to "Field solve" begin
    field.ffthelper.pifft * field.Ex
    field.ffthelper.pifft * field.Ey
    field.ffthelper.pifft * field.Ez
    field.ffthelper.pifft * field.Bx
    field.ffthelper.pifft * field.By
    field.ffthelper.pifft * field.Bz
  end
  @timeit to "Field update" update!(field)
end

@inline function depositindicesfractions(s::AbstractShape, z::Float64, NZ::Int,
                                         Lz::Float64)
  zNZ = z / Lz * NZ
  i = unimod(ceil(Int, zNZ), NZ)
  r = i - zNZ;
  @assert 0 < r <= 1 "$z, $NZ, $Lz, $i"
  return gridinteractiontuple(s, i, r, NZ)
end

@inline gridinteractiontuple(::NGPWeighting, i, r, NZ) = ((i, 1), )
@inline gridinteractiontuple(::AreaWeighting, i, r, NZ) = ((i, 1-r), (unimod(i+1, NZ), r))

@inline function (f::ElectrostaticField)(s::AbstractShape, xi, yi)
  NX, NY = f.gridparams.NX, f.gridparams.NY
  Lx, Ly = f.gridparams.Lx, f.gridparams.Ly
  Ex = Ey = zero(eltype(f.Exy))
  @inbounds for (j, wy) in depositindicesfractions(s, yi, NY, Ly)
    for (i, wx) in depositindicesfractions(s, xi, NX, Lx)
      wxy = wx * wy
      @muladd Ex = Ex + f.Exy[1,i,j] * wxy
      @muladd Ey = Ey + f.Exy[2,i,j] * wxy
    end
  end
  return (Ex, Ey)
end

@inline function (f::LorenzGuageField)(s::AbstractShape, xi, yi)
  NX, NY = f.gridparams.NX, f.gridparams.NY
  Lx, Ly = f.gridparams.Lx, f.gridparams.Ly
  Ex = Ey = Ez = Bx = By = Bz = zero(eltype(f.EBxyz))
  @inbounds for (j, wy) in depositindicesfractions(s, yi, NY, Ly)
    for (i, wx) in depositindicesfractions(s, xi, NX, Lx)
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

function deposit!(z, s::AbstractShape, x, y, Lx, Ly, w::Number)
  NX, NY = size(z)
  @inbounds for (j, wy) in depositindicesfractions(s, y, NY, Lx)
    for (i, wx) in depositindicesfractions(s, x, NX, Lx)
      z[i,j] += wx * wy * w
    end
  end
end

function deposit!(z, s::AbstractShape, x, y, Lx, Ly, w1, w2, w3, w4)
  NV, NX, NY = size(z)
  @assert NV == 4
  @inbounds for (j, wy) in depositindicesfractions(s, y, NY, Lx)
    for (i, wx) in depositindicesfractions(s, x, NX, Lx)
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
    t % d.nskip == 0 && (d.ti[] += 1)
    if t % d.nskip == 0
      diagnose!(d, plasma, to)
    end
    @timeit to "Fields" begin
      ti = d.ti[]
      if t % d.nskip == 0
        d.fieldenergy[ti] = mean(abs2, f.Exy) / 2
      end
      d.Exs[ti] .+= real.(f.Ex) ./ d.nskip
      d.Eys[ti] .+= real.(f.Ey) ./ d.nskip
      d.ϕs[ti] .+= real.(f.ffthelper.pifft * f.ϕ) ./ d.nskip
    end
  end
end

function diagnose!(d::LorenzGuageDiagnostics, f::LorenzGuageField, plasma, t, to)
  @timeit to "Diagnostics" begin
    t % d.nskip == 0 && (d.ti[] += 1)
    if t % d.nskip == 0
      diagnose!(d, plasma, to)
    end
    @timeit to "Fields" begin
      ti = d.ti[]
      if t % d.nskip == 0
        d.fieldenergy[ti] = mean(abs2, f.EBxyz) / 2
      end
      @views d.Exs[ti] .+= real.(f.Ex) ./ d.nskip
      @views d.Eys[ti] .+= real.(f.Ey) ./ d.nskip
      @views d.Ezs[ti] .+= real.(f.Ez) ./ d.nskip
      @views d.Bxs[ti] .+= real.(f.Bx) ./ d.nskip
      @views d.Bys[ti] .+= real.(f.By) ./ d.nskip
      @views d.Bzs[ti] .+= real.(f.Bz) ./ d.nskip
      d.Axs[ti] .+= real.(f.ffthelper.pifft * f.Ax) ./ d.nskip;
      d.Ays[ti] .+= real.(f.ffthelper.pifft * f.Ay) ./ d.nskip;
      d.Azs[ti] .+= real.(f.ffthelper.pifft * f.Az) ./ d.nskip;
      d.ϕs[ti] .+= real.(f.ffthelper.pifft * f.ϕ) ./ d.nskip
      f.ffthelper.pfft * f.ϕ; # Fourier transpose back
      f.ffthelper.pfft * f.Ax; # Fourier transpose back
      f.ffthelper.pfft * f.Ay; # Fourier transpose back
      f.ffthelper.pfft * f.Az; # Fourier transpose back
    end
  end
end

function diagnosticfields(d::ElectrostaticDiagnostics)
  return ((d.Exs, "Ex"), (d.Eys, "Ey"), (d.ϕs, "ϕ"))
end

function diagnosticfields(d::LorenzGuageDiagnostics)
  return ((d.Exs, "Ex"), (d.Eys, "Ey"), (d.Ezs, "Ez"),
          (d.Bxs, "Bx"), (d.Bys, "By"), (d.Bzs, "Bz"),
          (d.Axs, "Ax"), (d.Ays, "Ay"), (d.Azs, "Az"),
          (d.ϕs, "ϕ"))
end


function plotfields(d::AbstractDiagnostics, field, n0, vth, NT)
  B0 = norm(field.B0)
  dt = timestep(field)
  g = field.gridparams
  xs = (1:g.NX) ./ g.NX ./ (vth / B0)
  ys = (1:g.NY) ./ g.NY ./ (vth / B0)
  ndiags = d.ti[]
  ts = (1:ndiags) .* ((NT * dt / ndiags) / (2pi/B0))

  filter = sin.(((1:ndiags) .- 0.5) ./ ndiags .* pi)'
  ws = 2π/(NT * dt) .* (1:ndiags) ./ (B0);
  
  kxs = 2π .* (0:g.NX-1) ./ (B0/vth);
  kys = 2π .* (0:g.NY-1) ./ (B0/vth);
  
  wind = findlast(ws .< max(5, 2 * sqrt(n0)/B0));
  isnothing(wind) && (wind = length(ws)÷2)
  @views for (fvec, FS) in diagnosticfields(d)
    F = cat(fvec..., dims=3)
    all(iszero, F) && (println("$FS is empty"); continue)
    if d.makegifs
      maxabsF = maximum(abs, F)
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
  
    Z = log10.(sum(i->abs.(fft(F[:, i, :])[2:end÷2-1, 1:wind]), 1:size(F, 2)))'
    heatmap(kxs[2:end÷2-1], ws[1:wind], Z)
    xlabel!(L"Wavenumber x $[\Omega_c / v_{th}]$");
    ylabel!(L"Frequency $[\Omega_c]$")
    savefig("PIC2D3V_$(FS)_WKsumy_c.png")
    xlabel!(L"Wavenumber x $[\Pi / v_{th}]$");
    ylabel!(L"Frequency $[\Pi]$")
    heatmap(kxs[2:end÷2-1] .* B0 / sqrt(n0), ws[1:wind] .* B0 / sqrt(n0), Z)
    savefig("PIC2D3V_$(FS)_WKsumy_p.png")
   
    Z = log10.(sum(i->abs.(fft(F[i, :, :])[2:end÷2-1, 1:wind]), 1:size(F, 1)))'
    heatmap(kys[2:end÷2-1], ws[1:wind], Z)
    xlabel!(L"Wavenumber y $[\Omega_c / v_{th}]$");
    ylabel!(L"Frequency $[\Omega_c]$")
    savefig("PIC2D3V_$(FS)_WKsumx_c.png")
    heatmap(kys[2:end÷2-1] .* B0 / sqrt(n0), ws[1:wind] .* B0 / sqrt(n0), Z)
    xlabel!(L"Wavenumber y $[\Pi / v_{th}]$");
    ylabel!(L"Frequency $[\Pi]$")
    savefig("PIC2D3V_$(FS)_WKsumx_p.png")
  end

end

end

using ProgressMeter, TimerOutputs, Plots

import .PIC2D3V

function pic()

  to = TimerOutput()

  @timeit to "Initialisation" begin
    NX = NY = 128
    Lx = Ly = 1.0
    P = NX * NY * 2^5
    NT = 2^15
    dl = min(Lx / NX, Ly / NY)
    n0 = 4*pi^2
    vth = sqrt(n0) * dl
    B0 = sqrt(n0)/4;
    NS = 2

    #dt = dl/6vth
    #field = PIC2D3V.ElectrostaticField(NX, NY, Lx, Ly, dt=dt, B0x=B0)
    #diagnostics = PIC2D3V.ElectrostaticDiagnostics(NX, NY, NT, NS; makegifs=false)
    dt = 0.1 * dl/1 #/6vth
    field = PIC2D3V.LorenzGuageField(NX, NY, Lx, Ly, dt=dt, B0x=B0)
    diagnostics = PIC2D3V.LorenzGuageDiagnostics(NX, NY, NT, NS)
    electrons = PIC2D3V.Species(P, vth, n0, PIC2D3V.AreaWeighting();
      Lx=Lx, Ly=Ly, charge=-1, mass=1)
    ions = PIC2D3V.Species(P, vth / sqrt(32), n0, PIC2D3V.AreaWeighting();
      Lx=Lx, Ly=Ly, charge=1, mass=32)
    plasma = [electrons]#, ions]

    @show NX, NY, P, NT, NT÷NS, NS, dl, n0, vth, B0, dt
    @show vth * (NT * dt)
    @show (NT * dt) / (2pi/B0), (2pi/B0) / (dt * NS)
    @show (NT * dt) / (2pi/sqrt(n0)),  (2pi/sqrt(n0)) / (dt * NS)
  end
  
  @showprogress 1 for t in 0:NT-1;
    PIC2D3V.loop!(plasma, field, to)
    PIC2D3V.diagnose!(diagnostics, field, plasma, t, to)
  end

  show(to)

  return diagnostics, field, plasma, n0, vth, NT
end

diagnostics, field, plasma, n0, vcharacteristic, NT = pic()

PIC2D3V.plotfields(diagnostics, field, n0, vcharacteristic, NT)

