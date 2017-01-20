load('Y.mat');
%load('X.mat');
c=5;
N=500;% no. of training data
wts = ones(N,1);% weight vector
alpha = zeros(4,N);
imp = zeros(4,1);
err = zeros(4,1);
b=zeros(4,1);
for boost=1:4
C=zeros(N,1);


% upper bound for alpha
for i=1:N
    C(i)=c*i*2;
end
trs=2500;% index after which training data starts
C= C/(N);
G= zeros(N,N) ;
% calulating G
for i=1:N
    for j= 1:N
        G(i,j)=Y(trs+i)*Y(trs+j)*(rbf(wts(j)*X(trs+i,:),wts(i)*X(trs+j,:)));
    end
end 
one=ones(N,1);
% equation to be optimized
fun = @(a)-a*(one)+(a*G*(transpose(a)))/2; % a is alpha matrix
a0 = zeros(1,N); %initializing alpha
for i=1:N
    a0(i) = rand()*i;
end
a0=a0*2/(N*10);
Aeq = zeros(N,1);
for i=1:N
    Aeq(i,1)=Y(trs+i,1);
end
Aeq=transpose(Aeq);
options = optimset('Algorithm','sqp','MaxIter',50,'MaxFunEvals',20000,'Display','iter');
a = fmincon(fun,a0,[],[],Aeq,0,zeros(1,N),C',[],options);
% calculations for b
a1=1000;
a2=-1000;
Sum = zeros(N,1);% w(t)x for train data
% calculations for b
for i=1:N
    S = 0;
    for j=1:N
        S= S + a(j)*Y(trs+j)*(rbf(wts(j)*X(trs+j,:),wts(i)*X(trs+i,:))); 
    end  
    Sum(i)=S;
    if Y(trs+i)== 1
        if a1 > Sum(i)
            a1 = Sum(i);
        end
    else
        if a2 < Sum(i)
            a2 = Sum(i);
        end
    end
end 
b(boost)= -(a1+a2)/2;


 % predicting labels on train data

Sum = zeros(N,1); %w(t)x for train data
pred = ones(N,1);
accuracy=0;
for i=1:N
    s = 0;
    for j=1:N
        s= s + a(j)*Y(trs+j)*(rbf(wts(j)*X(trs+j,:),wts(i)*X(trs+i,:))); 
    end  
    Sum(i)=s;
    if Sum + b(boost) < 0
        pred(i)= -1;
    end    
    if pred(i)==Y(trs+i)
        accuracy = accuracy + 1;
    else
        err(boost)=err(boost)+wts(i);
    end
end  
accuracy = accuracy*100/N;
alpha(boost,:)= a;
err(boost)= err(boost)/N;
imp(boost)= log((1-err(boost))/err(boost))/2;
% updating weights
for u=1:N
    if pred(u)== Y(trs+u)
        wts(u)=wts(u)*exp(imp(boost));
    else
        wts(u)= wts(u)*exp(-imp(boost));
    end    
end 
wet = 0;
for p=1:N
    wet = wet + wts(p);
end    
wts = wts*N/wet;


end  

% predicting on test data
m=100;
sum = zeros(m,4); %w(t)x for train data
pred = ones(m,1);
acc=0;
for i=1:m
    s = zeros(4,1);
    % calculating h1 - h4
    for j=1:N
        for k=1:4
            s(k)= s(k) + alpha(k,j)*Y(trs+j)*(rbf(X(trs+j,:),X(trs+N+i,:))); 
        end    
    end 
    for k=1:4
        sum(i,k)=s(k);
    end 
    temp=0;
    % predicting labels using H(x)
    for k=1:2
        temp = temp + imp(k)*(sum(i,k)+b(k));
    end    
    pred(i)= sign(temp) ;
    if pred(i)==Y(trs+N+i)
        acc = acc + 1;
    end
    
end    
acc = acc*100/m;
   