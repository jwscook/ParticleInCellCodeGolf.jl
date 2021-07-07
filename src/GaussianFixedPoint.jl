using FFTW,LinearAlgebra,Plots,SpecialFunctions;N=128;P=32N;dt=1/6N;T=1024;
W=400;w=W/P*N;ξ=im*zeros(N);x=rand(P);v=rand([-1.0,1.0],P);r=zeros(N);
E=zeros(N);x0=copy(x);v0=copy(v);ik=2π*im*vcat(1,1:N/2,-N/2+1:-1);D=zeros(T,4);
f(g,c)=erf((g-c)*N)/2;ff(i,c)=f((i+0.5)/N,c)-f((i-0.5)/N,c);l=1e-8;
d(c)=((mod1(i,N),ff(i,c)) for i∈(-6:6).+Int(round(c*N)));F=copy(E);
ρ(x,y)=(r.*=0;for j∈d.((x.+y)./2);for k∈j;r[k[1]]+=k[2]*w;end;end;r);
@gif for t ∈1:T;x0.=x;v0.=v;F.*=NaN;for _∈0:9;isapprox(F,E,rtol=l)&&break;F.=E;
  x.=mod.(x0.+(v.+v0)/2*dt,1);E.=real.(ifft((ξ.=fft(ρ(x,x0))./ik;ξ[1]*=0;ξ)));
  for j∈eachindex(v);v[j]=v0[j]+sum(k->E[k[1]]*k[2],d((x[j]+x0[j])/2))*dt;end;
  D[t,1:2].=(sum(E.^2)/N,sum(v.^2)*W/P)./2;D[t,3:4].=sum.((D[t,1:2],v/P));end;
  scatter(x,v,ylims=(-3,3));D[t,1:3].*=2/W;plot!((1:t)./T,D[1:t,:],legend=0==1)
end every 8
