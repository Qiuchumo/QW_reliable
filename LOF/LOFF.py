#coding=utf-8
#本质是基于密度的检测 缺点：计算量巨大
#优化 重复点计算
import math
print(sorted([1,3,2])[:1],[1,3,2][1:])

class LOF:
    def __init__(self,data,k,threshold):
        self.data=data
        self.k=k
        self.threshold=threshold
        self.outliners=[]

    def __calDistance(self,a,b):
        sum1=0
        for k in range(len(a)):
            sum1+=((a[k]-b[k])**2)
        sum1=math.sqrt(sum1)
        if sum1==0:
            sum1=0.00000001
        return sum1

    def __calNk(self,point):
        Nk=[]
        disList=[]
        for j in range(len(self.data)):
            dis=self.__calDistance(point,self.data[j])
            disList.append([dis,data[j]])
        distList=sorted(distList)
        distance=distList[k-1][0]
        distList=distList[0:k-1]
        disList2=[]
        for di in distList[k-1:]:
            if di[0]==distList[k-1][0]:
                disList2.append(di)
        Nk=distList+disList2
        Nk=[nk[1] for nk in Nk]
        return Nk,distance

    def __getReachDis(self,point,Nk):
        reachDis=[]
        for nk in Nk:
            Nk1,dis1=self.__calNk(point)
            dis2=self.__calNk(point,nk)
            reachDis.append(max(distance,distance2))
        return reachDis

    def __getLrd(self,point):
        Nk,distance=self.__calNk(point)
        reachDis=self.__getReachDis(point,Nk)
        lrdPoint=1.0*sum(reachDis)/len(reachDis)
        return lrdPoint,Nk

    def __getLrdList(self,num):
        lrdList=[]
        lrdPoint,Nk=self.__getLrd(self.data[num])
        for l in range(len(Nk)):
            lr,=self.__getLrd(Nk(l))
            lrdList.append(lr)
        return lrdPoint,lrdList

    def __getLOF(self,num):
        lrdPoint,lrdList=self.__getLrdList(num)
        lofValue=0
        for lrd in lrdList:
            lofValue+=1.0*(lrd/lrdPoint)
        lofValue=lofValue/len(lrdList)
        return self.data[i],lofValue

    def run(self):
        for i in range(len(data)):
            lofP,lofV=self.__getLOF(i)
            if lofV>self.threshold:
                self.outliners.append(lofP)
        self.outliners=list[(self.outliners)]
        return self.outliners