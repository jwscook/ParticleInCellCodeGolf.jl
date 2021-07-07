using FFTW,Plots;N=128;P=64N;dt=1/4N;NT=1024;W=200;w=W/P*N;
x=rand(P);v=collect(1:P.>P/2).*2 .-1;u()=(x.=mod.(x.+v/2*dt,1));n=zeros(N);
k=im.*2π*vcat(1,1:N/2,-N/2+1:-1);f(x)=Int(mod1(round(x*N),N));Σ=sum;
@gif for t in 1:NT;
  u();E=real.(ifft((E=fft((n.*=0;for j∈x;n[f(j)]+=w;end;n))./k;E[1]*=0;E)));
  u();v+=E[f.(x)]*dt;scatter(x,v,ylims=(-3,3),legend=false);
end every 8
