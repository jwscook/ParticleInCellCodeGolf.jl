using ProgressMeter, TimerOutputs, Plots, FFTW, Random, StaticNumbers

include("PIC2D3V.jl")
import .PIC2D3V

FFTW.set_num_threads(Threads.nthreads())

Random.seed!(0)
pow(a, b) = a^b
function pic()

  SPEED_OF_LIGHT = 299792458.0
  ELEMENTARY_MASS = 9.1093837e-31;
  ELEMENTARY_CHARGE = 1.60217663e-19;
  MU_0 = 4.0e-7 * π;
  EPSILON_0 = 1.0 / MU_0 / SPEED_OF_LIGHT / SPEED_OF_LIGHT;

  n = 1e16
  B = 2.1
  M = 2 * 1836
  TeeV = 1e4
  @show Va = B / sqrt(MU_0 * M * ELEMENTARY_MASS * n)
  @show Wp = ELEMENTARY_CHARGE * sqrt(n / ELEMENTARY_MASS / EPSILON_0)
  @show vthe = sqrt(TeeV * ELEMENTARY_CHARGE * 2 / ELEMENTARY_MASS)
  @show vthe / SPEED_OF_LIGHT
  @show lD0 = vthe / Wp
  @show Ωi0 = ELEMENTARY_CHARGE * B / M / ELEMENTARY_MASS
  kresolution = 5
  @show L = Va / Ωi0 * 2π * kresolution
  @show REQUIRED_GRID_CELLS = L / lD0
  @show vthe / Wp / (L / 512 / 4)

  @show m_lengthScale = L
  m_timeScale = m_lengthScale / SPEED_OF_LIGHT;
  m_electricPotentialScale = ELEMENTARY_MASS *
                             pow(m_lengthScale / m_timeScale, 2) /
                             ELEMENTARY_CHARGE;
  m_chargeDensityScale =
      m_electricPotentialScale * EPSILON_0 / pow(m_lengthScale, 2);
  m_numberDensityScale = m_chargeDensityScale / ELEMENTARY_CHARGE;
  m_magneticPotentialScale =
      m_timeScale * m_electricPotentialScale / m_lengthScale;
  m_currentDensityScale = m_chargeDensityScale * m_lengthScale / m_timeScale;
  @show EPSILON_0 * SPEED_OF_LIGHT^2 / m_lengthScale * m_electricPotentialScale * m_magneticPotentialScale
  @show SPEED_OF_LIGHT / m_lengthScale


  to = TimerOutput()

  NQ = 1
  NX = 1024 ÷ NQ
  NY = 1024 * NQ

  @show L0 = L / m_lengthScale
  @show dt = L / NX / SPEED_OF_LIGHT / m_timeScale / 8
  @show B0 = B / (m_magneticPotentialScale / m_lengthScale)
  @show n0 = n / m_numberDensityScale
  @show Πe = sqrt(ELEMENTARY_CHARGE^2 * n / EPSILON_0 / ELEMENTARY_MASS) * m_timeScale
  @show vth = sqrt(TeeV * ELEMENTARY_CHARGE * 2 / ELEMENTARY_MASS) / m_lengthScale * m_timeScale
  @show Va / SPEED_OF_LIGHT, B0 / sqrt(M * n0)
  @show Ωi = B0 / M
  @show Ωi, Ωi * m_timeScale
  @show ld = vth / Πe
  @show lD0 / m_lengthScale, ld
  Va = Va / SPEED_OF_LIGHT

  #@timeit to "Initialisation" begin
    Lx = L0
    Ly = Lx * NY / NX
    @show P = NX * NY * 16
    @show NT = 2^7 #2^10#2^14
    Δx = Lx / NX
    @show Δx = Lx / NX
    @show Δy = Ly / NY
    @show dl = min(Lx / NX, Ly / NY)
    @show vth / Πe / Δx
    #n0 = 3.5e6 #4 * pi^2
    #debyeoverresolution = 1
    #vth = 0.01 #debyeoverresolution * dl * sqrt(n0)
    #B0 = sqrt(n0) / 4;
    #λ = vth / sqrt(n0)
    #rL = vth / B0 = 4λ

    #ntskip = 4
    #dt = dl/6vth
    #field = PIC2D3V.ElectrostaticField(NX, NY, Lx, Ly, dt=dt, B0x=B0)
    #diagnostics = PIC2D3V.ElectrostaticDiagnostics(NX, NY, NT, ntskip, 2)
    ntskip = 1 #4#prevpow(2, round(Int, 10 / 6vth)) ÷ 4
    ngskip = 1
    @show NT ÷ ntskip
    #dt = 2dl #/6vth
    #dt = dl / vth
    M = 32
    @show (SPEED_OF_LIGHT * m_timeScale / m_lengthScale * dt) / dl
    @show (vth * dt) / dl, 2π/B0 / dt
    @show (2π/B0) / dt, NT * dt / (2π*M/B0)
    @show (2π / Πe) / dt
    @show (vth / Πe) / (Lx / NX)
    @show Va * 2π/Lx / Ωi, Va * 2π/(Lx / NX) / Ωi
    #field = PIC2D3V.LorenzGaugeField(NX, NY, Lx, Ly, dt=dt, B0x=B0,
    #  imex=PIC2D3V.ImEx(1), buffer=10)
    field = PIC2D3V.LorenzGaugeStaggeredField(NX, NY, Lx, Ly, dt=dt, B0x=B0,
      imex=PIC2D3V.ImEx(1), buffer=10)
    #field = PIC2D3V.LorenzGaugeSemiImplicitField(NX, NY, Lx, Ly, dt=dt, B0x=B0,
    #  fieldimex=PIC2D3V.ImEx(1.0), sourceimex=PIC2D3V.ImEx(0.05), buffer=10, rtol=sqrt(eps()), maxiters=1000)
    diagnostics = PIC2D3V.LorenzGaugeDiagnostics(NX, NY, NT, ntskip, ngskip; makegifs=false)
    shape = PIC2D3V.BSplineWeighting{@stat 2}()
    #shape = PIC2D3V.NGPWeighting();#
    #shape = PIC2D3V.AreaWeighting();#
    electrons = PIC2D3V.Species(P, vth, n0, shape;
      Lx=Lx, Ly=Ly, charge=-1, mass=1)
    ions = PIC2D3V.Species(P, vth / sqrt(M), n0, shape;
      Lx=Lx, Ly=Ly, charge=1, mass=M)
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

  return diagnostics, field, plasma, n0, Va, Ωi, NT
end
using StatProfilerHTML
diagnostics, field, plasma, n0, vcharacteristic, omegachar, NT = pic()

PIC2D3V.plotfields(diagnostics, field, n0, vcharacteristic, omegachar, NT; cutoff=20)

