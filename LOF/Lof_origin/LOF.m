%LOF�㷨
%distΪm*m�ľ������ÿһ�д���һ���������������ݾ���������������Ըþ���Ϊ
%�Խ���Ϊ0�ģ����ڶԽ��߶ԳƵľ���,KΪk-����
% function lof = LOF(dist,K)

clear;
clc;

A=importdata('data.mat');%��Ҫ������Ⱥ�����㷨��������ݼ�
numData=size(A,1);
KD=[];
for i=1:1:numData
[~,dist]=knnsearch(A(i,:),A(:,:));
KD=[KD;dist'];
end

m=size(dist,1);                 %mΪ��������distΪ����֮��ľ���
distance = zeros(m,m);
num = zeros(m,m);               %distance ��num������¼������˳�򣬺Ͷ�����˳��
kdistance = zeros(m,1);         %����ÿ�������kdistance
count  = zeros(m,1);            %k����Ķ�����
reachdist = zeros(m,m);         %��������֮���reachable-distance
lrd = zeros(m,1);
lof = zeros(m,1);
%����k-distance
for i=1:m 
    [distance(i,:),num(i,:)]=sort(dist(i,:),'ascend');%distance���������dist��������num��¼����ǰ����dist����λ����Ϣ
    kdistance(i)=distance(i,K+1);%���k���ھ��룬��Ϊ������һ��ֵΪ��������ľ���Ϊ0������k+1���ǵ�k���������
    count(i) = -1;               %�Լ��ľ���Ϊ0��Ҫȥ���Լ�
    for j = 1:m                  %�ų��ж�����ݶԸõ�����ͬ��k���ھ��룬���û����count(i)=k��
        if dist(i,j)<=kdistance(i)
            count(i) = count(i)+1;%����k�������������ݵ����
        end
    end
end 
%����ɴ����
for i = 1:m
    for j=1:i-1                  %�������һ���������εľ���
        reachdist(i,j) = max(dist(i,j),kdistance(j));
        reachdist(j,i) = reachdist(i,j);
    end
end
%����ֲ��ɴ��ܶ�
for i = 1:m
    sum_reachdist=0;
    for j=1:count(i)
        sum_reachdist=sum_reachdist+reachdist(i,num(i,j+1));
    end
    %����ÿ�����lrd
    lrd(i)=count(i)/sum_reachdist;
end
% �õ��ֲ��쳣����lofֵ
for i=1:m
    sumlrd=0;
    for j=1:count(i)
        sumlrd=sumlrd+lrd(num(i,j+1))/lrd(i);
    end
    lof(i)=sumlrd/count(i);
end
%����ʱ�������ݵ�
% for i=1:1:numData
%     if lof(i)<1.5
%         plot(A(i,1),A(i,2),'b.');
%         hold on;
%     else
%         plot(A(i,1),A(i,2),'ro');
%         hold on
%     end
% end
% 
