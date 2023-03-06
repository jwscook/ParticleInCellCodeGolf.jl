using ProgressMeter, TimerOutputs, Plots, FFTW, Random, StaticNumbers
using ThreadPinning
if Base.Sys.isunix()
  pinthreads([0:2:(Threads.nthreads()-1)÷2])
end

include("PIC2D3V.jl")
import .PIC2D3V

FFTW.set_num_threads(Threads.nthreads())

using StatProfilerHTML
Random.seed!(0)
# vthe / c = 0.1 for 10 keV
# λD = 5e-5 m for 1e20 m^-3
# Ωe / Πe = 1 for 3T, 1e20m^-3
# λD = vthe / Πe = vthe / sqrt(n0)
# NG = 2π * sqrt(mi) * N_kres / vthe
# n0 = NG^2 * vthe^2
# B0 = sqrt(n0) *2 / 3

# λD = dl / debyeoverresolution therefore dl = debyeoverresolution * vthe / sqrt(n0)
# ... hence n0 = (dl / debyeoverresolution / vthe)^2

function pic()

  to = TimerOutput()

  #@timeit to "Initialisation" begin
    L = 1.0
    debyeoverresolution = 1
    #vth = debyeoverresolution * dl * sqrt(n0)
    vthe = 0.001 # 10 keV is 0.1
    N_kres = 8
    mi = 16
    n0 = (2π* N_kres)^2 * mi 
    n0 = 4π^2; vthe = sqrt(n0) / 256 * 2
    NG = nextpow(2, ceil(Int, sqrt(n0) / vthe))
    B0 = sqrt(n0) * 2 / 3; # 2T at 1e20m^-3
    NGT = 256^2
    NY = NG
    NX = max(1, NGT ÷ NY)
    Lx = 1.0
    Ly = Lx * NY / NX
    @show NG, NX, NY, n0, B0, L
    P = NX * NY * 2^4
    wci = B0 / mi
    tce = 2π / B0
    tci = tce * mi
    dl = min(Lx / NX, Ly / NY)
    dt = dl/6vthe / 4
    #dt = dl/2 * 0.75 #/6vthe
    @show tci, dt
    NT = nextpow(2, ceil(Int, tci / dt * 16))#2^15
    ntskip = prevpow(2, ceil(Int, tce/dt / 8))
    field = PIC2D3V.ElectrostaticField(NX, NY, Lx, Ly, dt=dt, B0x=B0)
    diagnostics = PIC2D3V.ElectrostaticDiagnostics(NX, NY, NT, ntskip, 1)
    @show dt * vthe / dl
    #field = PIC2D3V.LorenzGaugeStaggeredField(NX, NY, Lx, Ly, dt=dt, B0x=B0,
    #  imex=PIC2D3V.ImEx(0), buffer=10)
    #field = PIC2D3V.LorenzGaugeSemiImplicitField(NX, NY, Lx, Ly, dt=dt, B0x=B0,
    #  fieldimex=PIC2D3V.ImEx(1.0), sourceimex=PIC2D3V.ImEx(0.05), buffer=10, rtol=sqrt(eps()), maxiters=100)
    #diagnostics = PIC2D3V.LorenzGaugeDiagnostics(NX, NY, NT, ntskip, 1; makegifs=false)
    shape = PIC2D3V.BSplineWeighting{@stat 2}()
    #shape = PIC2D3V.NGPWeighting();#
    #shape = PIC2D3V.AreaWeighting();#
    electrons = PIC2D3V.Species(P, vthe, n0, shape;
      Lx=Lx, Ly=Ly, charge=-1, mass=1)
    ions = PIC2D3V.Species(P, vthe / sqrt(mi), n0, shape;
      Lx=Lx, Ly=Ly, charge=1, mass=mi)
    sort!(electrons, Lx / NX, Ly / NY)
    sort!(ions, Lx / NX, Ly / NY)
    plasma = [electrons, ions]

    @show NX, NY, P, NT, NT÷ntskip, ntskip, dl, n0, vthe, B0, dt
    #@show vth * (NT * dt)
    #@show (NT * dt) / (2pi/B0), (2pi/B0) / (dt * ntskip)
    #@show (NT * dt) / (2pi/sqrt(n0)),  (2pi/sqrt(n0)) / (dt * ntskip)
#  end
  plasmacopy = deepcopy(plasma)
  PIC2D3V.warmup!(field, plasma, to)
  for t in 0:NT-1;
    PIC2D3V.loop!(plasma, field, to, t, plasmacopy)
    PIC2D3V.diagnose!(diagnostics, field, plasma, t, to)
    if t % 256 == 0
      sort!(electrons, Lx / NX, Ly / NY)
      sort!(ions, Lx / NX, Ly / NY)
    end
  end

  show(to)

  return diagnostics, field, plasma, n0, 1.0, NT
end
diagnostics, field, plasma, n0, vcharacteristic, NT = pic()

PIC2D3V.plotfields(diagnostics, field, n0, vcharacteristic, NT; cutoff=1/16)

