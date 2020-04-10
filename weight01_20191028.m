clc;clear all;

root=cd; addpath(genpath([root '/DATA'])); addpath(genpath([root '/FUN']));
load jianmo.mat; data=feature; clear feature;

nSubj=size(data,1);
feature=size(data,2);
% T=size(data,1);
 method=input('SR[1],SLR[2]:');
  %%
 if method==1
    %Parameter setting for SLEP
    ex=-5:5;
   % lambda=2.^ex
    lambda=0;
    disp('Press any key:'); pause;
    nPar=length(lambda);
    WeightSet=cell(1,nPar);
    
    opts=[];
    opts.init=2;% Starting point: starting from a zero point here
    opts.tFlag=0;% termination criterion
    % abs( funVal(i)- funVal(i-1) ) ≤ .tol=10e?4 (default)
    %For the tFlag parameter which has 6 different termination criterion.
    % 0 ? abs( funVal(i)- funVal(i-1) ) ≤ .tol.
    % 1 ? abs( funVal(i)- funVal(i-1) ) ≤ .tol × max(funVal(i),1).
    % 2 ? funVal(i) ≤ .tol.
    % 3 ? kxi ? xi?1k2 ≤ .tol.
    % 4 ? kxi ? xi?1k2 ≤ .tol × max(||xi||_2, 1).
    % 5 ? Run the code for .maxIter iterations.
    opts.nFlag=0;% normalization option: 0-without normalization
    opts.rFlag=0;% regularization % the input parameter 'rho' is a ratio in (0, 1)
    opts.rsL2=0; % the squared two norm term in min  1/2 || A x - y||^2 + 1/2 rsL2 * ||x||_2^2 + z * ||x||_1
    fprintf('\n mFlag=0, lFlag=0 \n');
    opts.mFlag=0;% treating it as compositive function
    opts.lFlag=0;% Nemirovski's line search
    Y=Y;
    M=size(Y,2);
    for L=1:nPar
        brainNet=zeros(feature,5);
        for i=1:nSubj
            tmp=data(1,:);
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
        WeightSet{L}=brainNet;
        fprintf('Done %d/%d networks!\n',L,nPar);
    end
    save('WeightSet_S1.mat','WeightSet','Y');
end
 
 if method==2
     ex=5:15;
    %lamda=2.^ex
   % z=2.^ex
   lamda=0;
   z=0;
    disp('Press any key:'); pause;
    nL=length(lamda);
    nZ=length(z);
    WeightSet=cell(nL,nZ);%权重矩阵d*l
    SimilarSet=cell(nL,nZ);%相似度矩阵l*l
    opt.epsilon = 10^-5;
    opt.max_itr = 1000;
    X=data;
    Y=Y;
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
            WeightSet{iL,iZ}=Weight;
           SimilarSet{iL,iZ}= Simliar;
            
            
            fprintf('Done lamda=%d,z=%d networks!\n',iL,iZ); 
        end
    end
     save('WeightSet_S.mat','WeightSet','SimilarSet','Y');
     
     
     
     
     
 end