using FFTW,LinearAlgebra,Plots,SpecialFunctions; function pic()
N=64;P=16N;ϵ=1/10;dt=ϵ^2/N/2;T=4048;
W=400;w=W/P*N;Z=zeros;ξ=im*Z(N);x=rand(P);a=2π*rand(P);v=randn(P,3);
O()=(ξ[1]*=false;ξ);r=Z(N);
Ex=Z(N);Ey=Z(N);Ez=Z(N);Bx=Z(N);By=Z(N);Bz=Z(N).+1;X=copy(x);V=copy(v);
Fx=Z(N);Fy=Z(N);Fz=Z(N);Cy=Z(N);Cz=Z(N);
ik=2π*im*vcat(0,1:N/2,-N/2+1:-1);D=Z(T,4);irn(c)=Int(mod1(round(c*N),N));
f(g,c)=erf((g-c)*N)/2;ff(i,c)=f((i+0.5)/N,c)-f((i-0.5)/N,c);η=1e-8;
s(c,u=1)=((mod1(i,N),u*ff(i,c)) for i∈(-6:6).+irn(c));
d(y,u=0*y.+1)=(r.*=0;for j∈s.(y,u);for k∈j;r[k[1]]+=k[2]*w;end;end;r);
fields = zeros(N,T,12);rif(x)=real.(ifft(x));
for t ∈1:T;
  X.=x;V.=v; Ex.=real.(ifft((ξ.=fft(d(X))./ik;O())));
  x.=X.+v[:,1]*dt; x.=mod.(x,1);p=sortperm(x);x=x[p];v=v[p,:];
  Cy.=By; Cz.=Bz; By.-=dt.*rif(-ik.*fft(Ez)); Bz.-=dt.*rif(ik.*fft(Ey));
  by=(By.+Cy)./2; bz=(Bz.+Cz)./2;
  for j∈1:P;
    θ=s(x[j]); a=sum(k->[Ex[k[1]],Ey[k[1]],Ez[k[1]]]*k[2],θ);
    v[j,:].+=a*dt/2
    v[j,:].+=cross(v[j,:],sum(k->[Bx[k[1]],by[k[1]],bz[k[1]]]*k[2],θ))*dt;
    v[j,:].+=a*dt/2
  end
  Ey.+=dt.*rif(-ik.*fft(Bz)/ϵ^2 .-fft(d((X.+x)/2,v[:,2])));
  Ez.+=dt.*rif(+ik.*fft(By)/ϵ^2 .-fft(d((X.+x)/2,v[:,3])));
  fields[:,t,1].=Ex;fields[:,t,2].=Ey;fields[:,t,3].=Ez;fields[:,t,4].=By;
  fields[:,t,5].=Bz;fields[:,t,6].=d(x);fields[:,t,7].=d(x,v[:,1]);
  fields[:,t,8].=d(x, v[:,2]);fields[:,t,9].=d(x, v[:,3]);
  fields[:,t,10].=d(x, v[:,1].^2);fields[:,t,11].=d(x, v[:,2].^2);
  fields[:,t,12].=d(x, v[:,3].^2);
end;return fields;end

