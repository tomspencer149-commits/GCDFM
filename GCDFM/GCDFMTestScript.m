disp('START GCDFM test script-----------------------------------------------------------')

clear;           %clear existing variables

Settings.FminsearchMaxIter=1000;   % fminsearch MaxIter
Settings.Tol=0.0001;               % fminsearch tolerance
Settings.MethodNewDisplay='off';  % display option for fminsearch

w = warning ('off','all');

k=3; % number of dynamic factors

disp(['k=',num2str(k)])

%simulate random data
X=randn(100,10);

% get PCs
F_input=pca(X','NumComponents', 3);  

% call the new method to get the new factors
disp('starting GCDFM')
F_new = GCDFM(X,F_input,Settings);

disp('-------------------------------FINISHED-------------------------------------------')
