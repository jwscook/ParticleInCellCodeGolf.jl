using FFTW,Plots;N=128;P=64N;dt=1/4N;NT=1024;W=200;w=W/P*N;n=zeros(N);
v=collect(1:P.>P/2).*2 .-1.;u()=(x.=mod.(x.+v/2*dt,1));F=fft;IF=ifft;x=rand(P);
k=im.*2π*vcat(1,1:N/2,-N/2+1:-1);sc(a,b)=scatter(a,b,ylims=(-3,3),legend=0==1);
g(y)=(o=Int(mod1(ceil(y),N));([o,mod1(o-1,N)],[1-o+y,o-y]));O(z)=(z[1]*=0;z);
@gif for t in 1:NT;
  u();E=real.(IF((E=F((n.*=0;for i∈x;j=g(i);n[j[1]].+=j[2];end;n*w))./k;O(E))));
  u();i=0;for y∈x;i+=1;j=g(y);v[i]+=E[j[1]]'*j[2]*dt;end;sc(x,v);
end every 8
