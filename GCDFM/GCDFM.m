function F_new= GCDFM(X,F_input,Settings)

% Measurement equation is:     x(t) =    f(t)   *   B'    +  Et where Cov(E(t))=R
%                             1 x N       1 x k     k x N
% State Equation is:           f(t)=     f(t-1)  *   A'   +  Ut where Cov(U(t))=Q           
%                             1 x k       1 x k     k x k

[T,~]=size(X);

%initialise new factors with pc estimates
F=F_input; 

%options for solver
options = optimoptions('fminunc','display',Settings.MethodNewDisplay,'MaxIter',Settings.FminsearchMaxIter,...
    'MaxFunEvals',10000,'TolFun',Settings.Tol,'TolX',Settings.Tol);  

LLseries(1)=GetLL(F_input,X);      %LL of input data
disp(['Log-likeliood of PCs input  :',num2str(LLseries(1))]);


for t=1:T               % this is the loop which iterates through t=1..T
    ft=F(t,:);          % obtain the factors for time period t from the matrix F, used as initial values in optimisaion
    ftHat=fminunc(@(ft)GetObjective(F,X,ft,t),ft,options);   % perform the minimisation 
    F(t,:)=ftHat;       %store the optimised factors for time period t
    LogL= GetLL(F,X);
    LLseries(t+1)=LogL;
    disp(['t  :',num2str(t),'   Log-likeliood     :',num2str(LogL),'   %diff     :',num2str((LogL/LLseries(t)-1)*100)]);
end
disp(['Final Log-likeliood  :',num2str(LogL),'   %diff from PCs input     :',num2str((LogL/LLseries(1)-1)*100)]);

F_new=F; 

function objective=GetObjective(F,X,ft,t)
% this function is a wrapper for the main GetLL function below
F2=F;                  % copy supplied factors
F2(t,:)=ft;            % insert the ft vectorr which we want to to find the argmin over
 
objective = GetLL(F2,X);

function [objective, F, A, B, R, Q, E, U] = GetLL(F,X)
%this function is the negative LL

[T,N]=size(X);

% Define the current X and F and the lagged F
Xcurrent=X(2:T,:);    %current X
Flag=F(1:T-1,:);      %lagged F
Fcurrent= F(2:T,:);   %current F

%demean the factors
M=ones(T-1,1)*mean(Fcurrent);
Fcurrent=Fcurrent-M;
Flag=Flag-M;

%normalise F so F'F/T=Identity
Adjustment=inv(cov(Fcurrent))^0.5;
Fcurrent=Fcurrent*Adjustment;        
Flag=Flag*Adjustment;                

%concentrate out the parameters via OLS and caculate residuals
A=(Flag'*Flag)^-1*Flag'*Fcurrent;     % standard OLS formula for state equation  
U=Fcurrent-Flag*A;                    % state eqn errors
B=((Xcurrent'*Fcurrent)*inv(U'*U+Fcurrent'*Fcurrent))';  % OLS plus interaction U'U term 
E=Xcurrent-Fcurrent*B;                %measurement eqn errors

%concentrate out the covariances given the reulting residuals
R=E'*E/(T-2);      
Q=U'*U/(T-2);

%Calculate the Log-likelihood
Omega=R+B'*Q*B;  
objective=0.5*(T-1)*(N+log(det(Omega)));     % this is the LL output 
  
