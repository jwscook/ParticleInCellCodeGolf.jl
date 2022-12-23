using
FFTW,LinearAlgebra,Plots,SpecialFunctions;N=64;P=16N;ϵ=1/10;dt=ϵ^2/N/2;T=4048;
W=400;w=W/P*N;Z=zeros;ξ=im*Z(N);x=rand(P);a=2π*rand(P);v=Z(P,3);v[:,1].=sin.(a)/4;
v[:,2].=cos.(a)/4;O()=(ξ[1]*=false;ξ);r=Z(N);
#v=Z(P,3);v[:,1].+=rand([-1.0,1.0],P);
Ex=Z(N);Ey=Z(N);Ez=Z(N);Bx=Z(N);By=Z(N);Bz=Z(N).+1;X=copy(x);V=copy(v);
Fx=Z(N);Fy=Z(N);Fz=Z(N);Cy=Z(N);Cz=Z(N);
ik=2π*im*vcat(0,1:N/2,-N/2+1:-1);D=Z(T,4);irn(c)=Int(mod1(round(c*N),N));
f(g,c)=erf((g-c)*N)/2;ff(i,c)=f((i+0.5)/N,c)-f((i-0.5)/N,c);η=1e-8;
s(c,u=1)=((mod1(i,N),u*ff(i,c)) for i∈(-6:6).+irn(c));
d(y,u=0*y.+1)=(r.*=0;for j∈s.(y,u);for k∈j;r[k[1]]+=k[2]*w;end;end;r);
fields = zeros(N,T,12);rif(x)=real.(ifft(x));
@gif for t ∈1:T;X.=x;V.=v;Fx.*=NaN;Fy.=Ey;Fz.*=Ez;Cy.=By;Cz.=Bz;
  for _∈0:2;isapprox(Fx,Ex,rtol=η)&&break;Fx.=Ex;
  x.=X.+(V[:,1].+v[:,1])/2*dt;
#  x   v   x
#  R   J   R
#  E   B   E

  Ex.=real.(ifft((ξ.=fft(d((X.+x)/2))./ik;O())));
  by=(By.+Cy)/2;bz=(Bz.+Cz)/2;
  Ey.=Fy.+dt.*rif(-ik.*fft(bz)/ϵ^2 .-fft(d((X.+x)/2,(V[:,2].+v[:,2])/2)));
  Ez.=Fz.+dt.*rif(+ik.*fft(by)/ϵ^2 .-fft(d((X.+x)/2,(V[:,3].+v[:,3])/2)));
  ex=(Ex.+Fx)/2;ey=(Ey.+Fy)/2;ez=(Ez.+Fz)/2;
  By.=Cy.-dt.*rif(-ik.*fft(ez));
  Bz.=Cz.-dt.*rif(ik.*fft(ey));
  by=(By.+Cy)/2;bz=(Bz.+Cz)/2;
  for j∈1:P;u=V[j,:];θ=s((x[j]+X[j])/2);a=sum(k->[ex[k[1]],ey[k[1]],ez[k[1]]]*k[2],θ);
  v[j,:].=u.+(a.+cross(u.+a*dt/2,sum(k->[Bx[k[1]],by[k[1]],bz[k[1]]]*k[2],θ)))*dt;end;

#  Ex.=real.(ifft((ξ.=fft(d((X.+x)/2))./ik;O())));
#  by=(By.+Cy)/2;bz=(Bz.+Cz)/2;
#  Ey.=Fy.+dt.*rif(-ik.*fft(bz)/ϵ^2 .-fft(d((X.+x)/2,(V[:,2].+v[:,2])/2)));
#  Ez.=Fz.+dt.*rif(+ik.*fft(by)/ϵ^2 .-fft(d((X.+x)/2,(V[:,3].+v[:,3])/2)));
#  ex=(Ex.+Fx)/2;ey=(Ey.+Fy)/2;ez=(Ez.+Fz)/2;
#  By.=Cy.-dt.*rif(-ik.*fft(ez));
#  Bz.=Cz.-dt.*rif(ik.*fft(ey));
#  by=(By.+Cy)/2;bz=(Bz.+Cz)/2;
#  for j∈1:P;u=V[j,:];θ=s((x[j]+X[j])/2);a=sum(k->[ex[k[1]],ey[k[1]],ez[k[1]]]*k[2],θ);
#  v[j,:].=u.+(a.+cross(u.+a*dt/2,sum(k->[Bx[k[1]],by[k[1]],bz[k[1]]]*k[2],θ)))*dt;end;
  end;x.=mod.(x,1);p=sortperm(x);x=x[p];v=v[p,:];
  
  #scatter(x, atan.(v[:,2],v[:,1]))#,ylims=(-1,1));
#  scatter(v[:,1], v[:,2], xlims=(-1,1), ylims=(-1,1))
  fields[:,t,1].=Ex;fields[:,t,2].=Ey;fields[:,t,3].=Ez;fields[:,t,4].=By;
  fields[:,t,5].=Bz;fields[:,t,6].=d(x);fields[:,t,7].=d(x,v[:,1]);
  fields[:,t,8].=d(x, v[:,2]);fields[:,t,9].=d(x, v[:,3]);
  fields[:,t,10].=d(x, v[:,1].^2);fields[:,t,11].=d(x, v[:,2].^2);
  fields[:,t,12].=d(x, v[:,3].^2);
end every 32
