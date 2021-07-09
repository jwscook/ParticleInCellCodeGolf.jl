using LinearAlgebra,Plots;N=128;P=64N;dt=1/4N;T=1024;W=200;w=W/P*N;
R=rand;x=R(P);v=R([-1.0,1.0],P);m1=mod1;u()=(x.=m1.(x.+v/2*dt,1));K=zeros(T,2);
g(x)=Int.(m1.(round.(x.*N),N));
D2=zeros(N,N);for i∈1:N;D2[i,m1.(i.+[-1,0,1],N)].=[1,-2,1]*N^2;end
D1=zeros(N,N);for i∈1:N;D1[i,m1.(i.+[0,1],N)].=[-1,1]*N;end
@gif for t in 1:T; 
  u();ϕ=D2\(r=sum(i->i.==1:N,g(x))*w;r.-=sum(r)/N);u();
  E=D1*ϕ;v+=E[g(x.-1/2N)]*dt;scatter(x,v,ylims=(-3,3))
  K[t,:].=(sum(E.^2)/2N,sum(v.^2/2)*W/P);plot!((1:t)/T,K[1:t,:].*2/W);
  plot!((1:t)/T,sum(K[1:t,:].*2/W,dims=2),legend=0==1);
end every 8
