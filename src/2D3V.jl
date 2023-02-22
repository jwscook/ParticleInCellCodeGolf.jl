using ProgressMeter, TimerOutputs, Plots, FFTW, Random, StaticNumbers

include("PIC2D3V.jl")
import .PIC2D3V

FFTW.set_num_threads(Threads.nthreads())

Random.seed!(0)

function pic()

  to = TimerOutput()

  #@timeit to "Initialisation" begin
    NQ = 4
    NX = 128 ÷ NQ
    NY = 128 * NQ
    Lx = 1.0
    Ly = Lx * NY / NX
    P = NX * NY * 2^3
    NT = 2^13#2^14
    dl = min(Lx / NX, Ly / NY)
    n0 = 4 * pi^2
    debyeoverresolution = 1
    vth = debyeoverresolution * dl * sqrt(n0)
    B0 = sqrt(n0) / 4;
    #λ = vth / sqrt(n0)
    #rL = vth / B0 = 4λ

    #ntskip = 4
    #dt = dl/6vth
    #field = PIC2D3V.ElectrostaticField(NX, NY, Lx, Ly, dt=dt, B0x=B0)
    #diagnostics = PIC2D3V.ElectrostaticDiagnostics(NX, NY, NT, ntskip, 2)
    ntskip = 1#4#prevpow(2, round(Int, 10 / 6vth)) ÷ 4
    dt = dl/3 #/6vth
    #dt = dl / vth
    @show (vth * dt) / dl
    #field = PIC2D3V.LorenzGaugeField(NX, NY, Lx, Ly, dt=dt, B0x=B0,
    #  imex=PIC2D3V.ImEx(1), buffer=10)
    field = PIC2D3V.LorenzGaugeStaggeredField(NX, NY, Lx, Ly, dt=dt, B0x=B0,
      imex=PIC2D3V.ImEx(1), buffer=10)#, rtol=sqrt(eps()), maxiters=1)
    diagnostics = PIC2D3V.LorenzGaugeDiagnostics(NX, NY, NT, ntskip, 2; makegifs=false)
    #shape = PIC2D3V.BSplineWeighting{@stat 2}()#PIC2D3V.NGPWeighting();#
    shape = PIC2D3V.AreaWeighting();#
    electrons = PIC2D3V.Species(P, vth, n0, shape;
      Lx=Lx, Ly=Ly, charge=-1, mass=1)
    ions = PIC2D3V.Species(P, vth / sqrt(16), n0, shape;
      Lx=Lx, Ly=Ly, charge=1, mass=16)
    sort!(electrons, Lx / NX, Ly / NY)
    sort!(ions, Lx / NX, Ly / NY)
    plasma = [electrons, ions]

    #@show NX, NY, P, NT, NT÷ntskip, ntskip, dl, n0, vth, B0, dt
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

  return diagnostics, field, plasma, n0, vth, NT
end
using StatProfilerHTML
diagnostics, field, plasma, n0, vcharacteristic, NT = pic()

PIC2D3V.plotfields(diagnostics, field, n0, vcharacteristic, NT; cutoff=20)

