clc;clear all
%load('E:\新建文件夹\WeChat Files\wx879324221\FileStorage\File\paper\finaly\baseline\all_acc.mat')
%load('E:\新建文件夹\WeChat Files\wx879324221\FileStorage\File\paper\finaly\baseline\female_acc.mat')
load('E:\新建文件夹\WeChat Files\wx879324221\FileStorage\File\paper\finaly\baseline\male_acc.mat')
[m,n]=size(error4);
sumerro=zeros(1,5);
pr2=zeros(1,5);
for i=1:m
    for j=1:n
       sumerro=sumerro+error4{i,j} ;
        pr2=pr2+pr1{i,j};
    end
end
sumerro=sumerro./25;
pr3=pr2./25;


