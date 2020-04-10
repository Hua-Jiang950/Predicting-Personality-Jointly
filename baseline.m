clc;clear all;
data=xlsread('E:\paper_data\all_net.xlsx');
label=xlsread('E:\paper_data\all_factor.xlsx');
[M,N]=size(data);
fin_err=zeros(5,5);
error3=zeros(1,5);
pr1=cell(5,5);
error4=cell(5,5);
indices=crossvalind('Kfold',data(1:M,N),5);
for ki=1:5
test =(indices == ki);
train=~test;
train_data=data(train,:);
train_label=label(train,:);
%train_target=target(:,train);
 yan_data=data(test,:);
 yan_label=label(test,:);
 data=train_data;
 label=train_label;
 [p,q]=size(yan_label);

% data=xlsread('E:\paper_data\female_net.xlsx');
% label=xlsread('E:\paper_data\female_factor.xlsx');
%load 'femalet.mat'
[M,N]=size(data);
indices=crossvalind('Kfold',data(1:M,N),5);
ex=0:11;
    lambda=2.^ex
    nPar=length(lambda);
error=cell(1,5);
weight=cell(nPar,5);
error2=cell(1,5);
for k=1:5
test =(indices == k);
train=~test;
train_data=data(train,:);
train_label=label(train,:);
%train_target=target(:,train);
 test_data=data(test,:);
 test_label=label(test,:);
 [Num_test,Num_factor]=size(test_label);
% test_target=target(:,test);
[nSubj,feature]=size(train_data);

    disp('Press any key:');
    
    brainNetSet=zeros(nPar,Num_factor);
    error1=zeros(nPar,1);
    opts=[];
    opts.init=2;% Starting point: starting from a zero point here
    opts.tFlag=0;% termination criterion
    % abs( funVal(i)- funVal(i-1) ) ¡Ü .tol=10e?4 (default)
    %For the tFlag parameter which has 6 different termination criterion.
    % 0 ? abs( funVal(i)- funVal(i-1) ) ¡Ü .tol.
    % 1 ? abs( funVal(i)- funVal(i-1) ) ¡Ü .tol ¡Á max(funVal(i),1).
    % 2 ? funVal(i) ¡Ü .tol.
    % 3 ? kxi ? xi?1k2 ¡Ü .tol.
    % 4 ? kxi ? xi?1k2 ¡Ü .tol ¡Á max(||xi||_2, 1).
    % 5 ? Run the code for .maxIter iterations.
    opts.nFlag=0;% normalization option: 0-without normalization
    opts.rFlag=0;% regularization % the input parameter 'rho' is a ratio in (0, 1)
    opts.rsL2=0; % the squared two norm term in min  1/2 || A x - y||^2 + 1/2 rsL2 * ||x||_2^2 + z * ||x||_1
    fprintf('\n mFlag=0, lFlag=0 \n');
    opts.mFlag=0;% treating it as compositive function
    opts.lFlag=0;% Nemirovski's line search
    Y=train_label;
    M=size(Y,2);
    for L=1:nPar
        brainNet=zeros(feature,5);
        for i=1:nSubj
            tmp=train_data(1,:);
            for m=1:M
            %tmp=tmp-repmat(mean(tmp),T,1);% data centralization
            currentNet=zeros(feature,5);
            %for j=1:nROI
               y=Y(i,m);
                A=tmp;
                [x, funVal1, ValueL1]= LeastR(A, y, lambda(L), opts);
                currentNet(:,m) = x;
            %end
            brainNet(:,m)=currentNet(:,m);
            end
        end
        pred_label=test_data*brainNet;
        err=sum(abs(pred_label-test_label))./Num_test;
        err1=sum(err);
        brainNetSet(L,:)=err;
        error{k}=brainNetSet;
        error1(L,:)=err1;
         weight{L,k}=brainNet;
        fprintf('Done %d/%d in %d networks!\n',L,nPar,k );
    end
    
   
    error2{k}=error1;
    save('brainNetSet_SR.mat','error','weight','error2');
    er=cell2mat(error2(1,k));
 index=find(er==min(min(er)));
 pred_label=yan_data*weight{index,k};
    
    %pred_label=yan_data*weight{1,k};
    err3=sum(sum(abs(pred_label-yan_label)))./(p*q);
    error3(ki,k)=err3;
    err4=sum(abs(pred_label-yan_label))./p;
    error4{ki,k}=err4;
    for e=1:5
        Pear=corrcoef(pred_label(:,e),yan_label(:,e));
        pr(1,e)=Pear(1,2);
        
    end
    pr1{ki,k}=pr;
    
end
%fin_err(ki,:)=error3;
end
 save('linearacc.mat','error3','pr1','error4');