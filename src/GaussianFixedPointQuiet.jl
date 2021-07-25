using FFTW,LinearAlgebra,Plots,SpecialFunctions;pic=()->(N=64;P=32N;dt=1/6N;
T=2^13;W=32π^2/3;w=W/P*N;ξ=im*zeros(N);x=(bitreverse.(0:P-1).+2.0^63)/2.0^64;
v=collect(1:P.>P/2).*2 .-1.;r=zeros(N);
E=zeros(N);X=copy(x);V=copy(v);ik=2π*im*vcat(1,1:N/2,-N/2+1:-1);D=zeros(T,4);
f(g,c)=erf((g-c)*N)/2;ff(i,c)=f((i+0.5)/N,c)-f((i-0.5)/N,c);l=4eps();
d(c)=((mod1(i,N),ff(i,c)) for i∈(-7:7).+Int(round(c*N)));F=copy(E);
ρ(x,y=x)=(r.*=0;for l∈d.((x.+y)./2);for k∈l;r[k[1]]+=k[2]*w;end;end;r);
@time @gif for t ∈1:T;X.=x;V.=v;F.*=NaN;for _∈0:9;≈(F,E,rtol=l,atol=0)&&break;F.=E;
 x.=X.+(v.+V)/2*dt;E.=real.(ifft((ξ.=fft(ρ(x,X))./ik;ξ[1]*=0;ξ)));
 for j∈1:P;v[j]=V[j]+sum(k->E[k[1]]*k[2],d((x[j]+X[j])/2))*dt;end;end;x.=mod.(x,1);
 D[t,1:2].=(sum(E.^2)/N,sum(v.^2)*W/P)./2;D[t,3:4].=sum.((D[t,1:2],v/P));
 scatter(x[1:P÷2],v[1:P÷2],c=1);scatter!(x[P÷2+1:P],v[P÷2+1:P],ylims=(-3,3),c=2);
 D[t,1:3].*=2/W;plot!((1:t)./T,D[1:t,:],legend=0==1,xlims=(0,1)); @show t/T
 plot!((0.5:N)/N,ρ(x[1:P÷2])./W,c=1);plot!((0.5:N)/N,ρ(x[P÷2+1:P])./W,c=2)
end every 8;return (D,N,P,dt,T,W,w););(D,N,P,dt,T,W,w)=pic()
la(x)=log10(abs(x));t=(1:T).*dt;plot(t,la.(D[:,1]),label="electric field energy")
plot!(t,la.(1 .-D[:,3]),label="total energy change");yticks!(-35:1)
plot!(t,la.(D[:,4]),label="total momentum",legend=:bottomright)
γ(x)=imag(sqrt(Complex(x^2+1-sqrt(4*x^2+1))))*sqrt(W/2)/log(10);ylims!(-35,1)
plot!(t,2*γ(2π/sqrt(W/2)).*(t.-T÷8*dt).+la.(D[T÷8,1]),label="predicted")
xlabel!("Time");ylabel!("log 10");savefig("GaussianFixedPointQuiet.jpg")
