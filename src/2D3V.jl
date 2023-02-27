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
    debyeoverresolution = 1
    #vth = debyeoverresolution * dl * sqrt(n0)
    vthe = 0.001 # 10 keV is 0.1
    N_kres = 8
    mi = 16
    #NG = nextpow(2, 2π * sqrt(mi) * N_kres / vthe)
    #n0 = NG^2 * vthe^2
    n0 = 1.0
    NG = nextpow(2, sqrt(n0) / vthe)
    L = N_kres * 2π / sqrt(n0/mi)
    B0 = sqrt(n0) * 2 / 3; # 2T at 1e20m^-3
    @show NG, n0, B0, L
    NGT = 256^2
    NY = NG
    NX = NGT ÷ NY
    Ly = 1.0
    Lx = Ly * NX / NY
    P = NX * NY * 2^4
    NT = 2^18#2^14
    dl = min(Lx / NX, Ly / NY)
    @show dl, 2pi/sqrt(n0)
    #ntskip = 4
    #dt = dl/6vth
    #field = PIC2D3V.ElectrostaticField(NX, NY, Lx, Ly, dt=dt, B0x=B0)
    #diagnostics = PIC2D3V.ElectrostaticDiagnostics(NX, NY, NT, ntskip, 2)
    ntskip = 64#prevpow(2, round(Int, 10 / 6vthe)) ÷ 4
    dt = dl/2 #/6vthe
    @show dt * vthe / dl
    #field = PIC2D3V.LorenzGaugeField(NX, NY, Lx, Ly, dt=dt, B0x=B0,
    #  imex=PIC2D3V.ImEx(1), buffer=10)
    field = PIC2D3V.LorenzGaugeStaggeredField(NX, NY, Lx, Ly, dt=dt, B0x=B0,
      imex=PIC2D3V.ImEx(1), buffer=5)
    #field = PIC2D3V.LorenzGaugeSemiImplicitField(NX, NY, Lx, Ly, dt=dt, B0x=B0,
    #  fieldimex=PIC2D3V.ImEx(1.0), sourceimex=PIC2D3V.ImEx(0.05), buffer=10, rtol=sqrt(eps()), maxiters=100)
    diagnostics = PIC2D3V.LorenzGaugeDiagnostics(NX, NY, NT, ntskip, 2; makegifs=false)
    shape = PIC2D3V.BSplineWeighting{@stat 0}()
    #shape = PIC2D3V.NGPWeighting();#
    #shape = PIC2D3V.AreaWeighting();#
    electrons = PIC2D3V.Species(P, vthe, n0, shape;
      Lx=Lx, Ly=Ly, charge=-1, mass=1)
    ions = PIC2D3V.Species(P, vthe / sqrt(mi), n0, shape;
      Lx=Lx, Ly=Ly, charge=1, mass=mi)
    sort!(electrons, Lx / NX, Ly / NY)
    sort!(ions, Lx / NX, Ly / NY)
    plasma = [electrons, ions]

    #@show NX, NY, P, NT, NT÷ntskip, ntskip, dl, n0, vth, B0, dt
    @show dt * NT / (2pi/(B0/mi))
    #@show vth * (NT * dt)
    #@show (NT * dt) / (2pi/B0), (2pi/B0) / (dt * ntskip)
    #@show (NT * dt) / (2pi/sqrt(n0)),  (2pi/sqrt(n0)) / (dt * ntskip)
#  end
  plasmacopy = deepcopy(plasma)
  PIC2D3V.warmup!(field, plasma, to)
  for t in 0:NT-1;
    PIC2D3V.loop!(plasma, field, to, t, plasmacopy)
    PIC2D3V.diagnose!(diagnostics, field, plasma, t, to)
  end

  show(to)

  return diagnostics, field, plasma, n0, vthe, NT
end
diagnostics, field, plasma, n0, vcharacteristic, NT = pic()

PIC2D3V.plotfields(diagnostics, field, n0, vcharacteristic, NT; cutoff=1.0)

