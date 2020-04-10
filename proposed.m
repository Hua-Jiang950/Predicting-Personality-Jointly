clc;clear all;
data=xlsread('E:\paper_data\all_net.xlsx');
label=xlsread('E:\paper_data\all_factor.xlsx');
[M,N]=size(data);
fin_err=zeros(5,5);
error3=zeros(5,5);
error4=cell(5,5);
pr1=cell(5,5);
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
 [P,Q]=size(yan_label);
 [M,N]=size(data);
 pr=zeros(1,5);
 indices=crossvalind('Kfold',data(1:M,N),5);

 ex=0:11;
    lamda=2.^ex
   z=2.^ex
%    lamda=0;
%    z=0;
    disp('Press any key:');
    nL=length(lamda);
    nZ=length(z);

    WeightSet=cell(nL,nZ,5);%权重矩阵d*l
    SimilarSet=cell(nL,nZ,5);%相似度矩阵l*l
    error=cell(nL,nZ,5);
    error1=cell(nL,nZ,5);
for Km=1:5
test =(indices == Km);
train=~test;
train_data=data(train,:);
train_label=label(train,:);
%train_target=target(:,train);
 test_data=data(test,:);
 test_label=label(test,:);
 [Num_test,Num_factor]=size(test_label);
 
     
    
    opt.epsilon = 10^-5;
    opt.max_itr = 1000;
    X=train_data;
    Y=train_label;
    L=size(Y,2);
   % S=eye(size(lab_train,2))/size(lab_train,2);
    for iL=1:nL
        for iZ=1:nZ
            opt.lamda=lamda(iL);
            opt.z=z(iZ);
            Weight=zeros(size(X,2),size(Y,2));
            Simliar=zeros(size(Y,2));
            W=ones(size(X,2),size(Y,2));
            S=eye(size(Y,2))/size(Y,2);
            M=S.^2;
            D=diag(sum(M,2));
            F(1)=0.5*norm(X*W-Y,'fro')^2+lamda(iL)*trace(W*(D-M)*W')+z(iZ)*sum(sum(abs(W)));
 for k= 1:100
                [W,fval_vec, itr_counter] = accel_grad_mlr_jiang1028(X,Y,S,opt);
                %W=W{k+1};
                sW=0;
                for p=1:size(W,2)
                    for q=1:size(W,2)
                        if p~=q
                        sumW=1/sum((W(:,p)-W(:,q)).^2);
                        else sumW=0;
                        end
                        sW=sW+sumW;
                    end
                end
                %sW=1/sW;
for i=1:size(W,2)
                    for j=1:size(W,2)
                        if i~=j
                        %S(i,j)=2*trace(W*(L*eye(5)-ones(5))*W')/sum((W(:,i)-W(:,j)).^2);%%wei wan cheng
                        %S(i,j)=sum((W(:,i)-W(:,j)).^2)/2*trace(W*(L*eye(5)-ones(5))*W');
                        S(i,j)=(1/sum((W(:,i)-W(:,j)).^2))/sW;
                        else S(i,j)=0;
                        end
                   end
end
                F(k+1)=0.5*norm(X*W-Y,'fro')^2+lamda(iL)*trace(W*(diag(sum(S.^2,2))-S.^2)*W')+z(iZ)*sum(sum(abs(W)));
                if abs(F(k+1)-F(k))<10^-5
                    break;
                end
                Weight=W;
                Simliar=S;
 end
        pred_label=test_data*Weight;
        err=sum(abs(pred_label-test_label))./Num_test;
        err1=sum(err);
            WeightSet{iL,iZ,Km}=Weight;
           SimilarSet{iL,iZ,Km}= Simliar;
            error{iL,iZ,Km}= err;
            error1{iL,iZ,Km}= err1;
            fprintf('Done lamda=%d,z=%d networks!\n',iL,iZ); 
        end
    end
 
 save('weight.mat','WeightSet','SimilarSet','error','error1');
 er=cell2mat(error1(:,:,Km));
 [index1,index2]=find(er==min(min(er)));
 pred_label=yan_data*WeightSet{index1,index2,Km};
 %pred_label=yan_data*WeightSet{1,1,Km};
 
 
    err3=sum(sum(abs(pred_label-yan_label)))./(P*Q);
    err4=sum(abs(pred_label-yan_label))./P;
    error3(ki,Km)=err3;
     error4{ki,Km}=err4;
    for e=1:5
        Pear=corrcoef(pred_label(:,e),yan_label(:,e));
        pr(1,e)=Pear(1,2);
        
    end
    pr1{ki,Km}=pr;
    
    
end
 %fin_err(ki,:)=error3;
 
 
 
end
 save('linearacc1.mat','error3','pr1','error4');