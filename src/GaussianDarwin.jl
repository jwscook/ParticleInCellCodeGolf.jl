using FFTW,LinearAlgebra,Plots,SpecialFunctions,LoopVectorization,StaticArrays;
function pic();
N=128;P=32N;dt=1/6N;T=1024;μ=1e-2;
W=400;w=W/P*N;Z=zeros;ξ=im*Z(N);x=rand(P);a=2π*rand(P);v=randn(3,P)/10;
O()=(ξ[1]*=false;ξ);r=Z(N);
Ex=Z(N);Ey=Z(N);Ez=Z(N);Bx=Z(N);By=Z(N);Bz=Z(N).+1;X=copy(x);V=copy(v);
Fx=Z(N);Fy=Z(N);Fz=Z(N);Cy=Z(N);Cz=Z(N);
ik=2π*im*vcat(0,1:N/2,-N/2+1:-1);D=Z(T,4);irn(c)=Int(mod1(round(c*N),N));
f(g,c)=erf((g-c)*N)/2;ff(i,c)=f((i+0.5)/N,c)-f((i-0.5)/N,c);η=1e-8;
s(c,u=1)=((mod1(i,N),u*ff(i,c)) for i∈(-6:6).+irn(c));
d(y,u=0*y.+1)=(r.*=0;for j∈s.(y,u);for k∈j;r[k[1]]+=k[2]*w;end;end;r);
fields = zeros(N,T,12);rif(x)=real.(ifft(x));
for t ∈1:T;Cy.=By; Cz.=Bz; X.=x;V.=v; for _∈0:4;
    Ex.=real.(ifft((ξ.=fft(d(X))./ik;O())));
    x.=X.+v[1,:]*dt; x.=mod.(x,1);#p=sortperm(x);x=x[p];v=v[:,p];
    By.=rif((ξ.=fft(d((X.+x)/2,v[3,:]))./ik;-O()))*μ;
    Bz.=rif((ξ.=fft(-d((X.+x)/2,v[2,:]))./ik;O()))*μ.+1;
    Ey.=Fy.-dt*rif((ξ.=fft((Bz.-Cz))./ik;O()));
    Ez.=Fz.-dt*rif((ξ.=-fft((By.-Cy))./ik;O()));
    by=(By.+Cy)./2; bz=(Bz.+Cz)./2;
    Base.Threads.@threads for j∈1:Base.Threads.nthreads();
      Exc=deepcopy(Ex);
      Eyc=deepcopy(Ey);
      Ezc=deepcopy(Ez);
      Bxc=deepcopy(Bx);
      Bxc=deepcopy(by);
      Bxc=deepcopy(bz);
      for j in Int((tj-1)*P/4+1):Int(tj*P/4);
        θ=s(x[j]); a=sum(k->(@SArray [Exc[k[1]],Eyc[k[1]],Ezc[k[1]]]).*k[2],θ);
        v[:,j].=V[:,j].+a*dt/2
        v[:,j].+=cross(v[:,j],sum(k->(@SArray [Bxc[k[1]],Byc[k[1]],Bzc[k[1]]]).*k[2],θ))*dt;
        v[:,j].+=a*dt/2
      end
    end
  end
  fields[:,t,1].=Ex;fields[:,t,2].=Ey;fields[:,t,3].=Ez;fields[:,t,4].=By;
  fields[:,t,5].=Bz;fields[:,t,6].=d(x);fields[:,t,7].=d(x,v[1,:]);
  fields[:,t,8].=d(x, v[2,:]);fields[:,t,9].=d(x, v[3,:]);
  fields[:,t,10].=d(x, v[1,:].^2);fields[:,t,11].=d(x, v[2,:].^2);
  fields[:,t,12].=d(x, v[3,:].^2);
end;return fields;end;f=pic();

