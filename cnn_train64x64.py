import random
import os
import sys
import math
import numpy as np
import copy



trainPath = "D://python2.7.6//MachineLearning//plankton-CNN//trainbyYR"
dataName="D://python2.7.6//MachineLearning//plankton-CNN//dataVec.txt"
labelfile= "D://python2.7.6//MachineLearning//plankton-CNN//dataLabel.txt"
 
     

global classDic,labelList,dataList #11 2000
 
global epoch;epoch=3
global alpha;alpha=0.2
xd=64
xdd=xd*xd
nfilter=20
filterd=21
filterdd=filterd**2
pwind=4
convd=xd-filterd+1
convdd=convd**2
poold=convd/pwind
pooldd=poold**2

nh=(convd)**2
nhh=(poold)**2


######################

def loadData():
    global dataMat,yMat,classList,labelList,dataList
    classDic={};labelList=[];dataList=[]
    ########## all label  list
    content=open(labelfile,'r')
    line=content.readline().strip(' ')
    line=line.split(' ')
    for label in line:
        labelList.append(int(label))
    print '1',len(labelList)
    
    ##########
    obs=[]
    content=open(dataName,'r')
    line=content.readline().strip('\n').strip(' ')
    line=line.split(' ')
    #print line,len(line)
    while len(line)>1:
        obs=[float(n) for n in line if len(n)>1]
        #print 'o',obs,len(obs)
        
        line=content.readline().strip('\n').strip(' ');line=line.split(' ')
         
        dataList.append(obs);#print 'datalist',len(dataList)
    ##########
    print '%d obs loaded'%len(dataList),len(set(labelList)),'kinds of labels',len(dataList[0]),'dim'
    #print labelList,classDic
    ####
     
    #####
    dataMat=np.mat(dataList)
     
    ########
    num,dd=np.shape(dataMat)
     
    yMat=np.zeros((num,11))
    for n in range(num):
        truey=labelList[n]
        yMat[n,truey]=1.0


def initialH():
    global dataMat,yMat
    global hMat,hhMat,outputMat
    
    num,dd=np.shape(dataMat)
    hMat=np.mat(np.zeros((nfilter,nh)))#nfilter,nh
    outputMat=np.mat(np.zeros((num,11))) #nclass
    hhMat=np.mat(np.zeros((nfilter,nhh)))#nhh
    
    
     

def initialPara():
    global Cmat,Wmat,bmat,bbmat #initial from random eps
    num,dd=np.shape(dataMat)
      
    Cmat=np.mat(np.zeros((11,nhh*nfilter)))
    Wmat=np.mat(np.zeros((nfilter,filterdd)))
    bmat=np.mat(np.zeros((1,nfilter)))
    bbmat=np.mat(np.zeros((1,11)))
    for i in range(11):
        for j in range(nhh*nfilter):
            Cmat[i,j]=random.uniform(0,0.1)
    for i in range(nfilter):
        for j in range(filterdd):
            Wmat[i,j]=random.uniform(0,0.1)
    
    #######
    for j in range(nfilter):
        bmat[0,j]=random.uniform(0,0.1)
     
    for j in range(11):
        bbmat[0,j]=random.uniform(0,0.1)
     
    

            
def initialErr():#transfer err sensitive
    global errW,errC,up1,up2
    global dataMat
     
    n,d=np.shape(dataMat)
    errW=np.mat(np.zeros((1,nh)))
    errC=np.mat(np.zeros((1,11)))
    up1=np.mat(np.zeros((poold,poold)))
    up2=np.mat(np.zeros((convd,convd)))
     
    
def initialGrad():
      
    global gradc,gradw,gradb,gradbb
    gradc=np.mat(np.zeros((11,nfilter*nhh)))
    gradw=np.mat(np.zeros((nfilter,filterdd)))
    gradb=np.mat(np.zeros((1,nfilter)))
    gradbb=np.mat(np.zeros((1,11)))

def forward(x): #xi index not xvector
    global hMat,hhMat,outputMat
    global Cmat,Wmat,bmat,bbmat
    global dataMat
    
    xvec=dataMat[x,:]
    ######1x256->16x16->64x81   9x9filter
    x16=vec2mat(xvec,xd,xd)
    #print x16
    #### ->64x81    (16-9+1)x(16-9+1) 8x8
    x64=np.mat(np.zeros((nh,filterdd)))##64 pieces of dim81 patch
    i=0
    for hang in range(convd):
        for lie in range(convd):
            patch=x16[hang:hang+filterd,lie:lie+filterd]  #[0:9]==0,1,2,,,8 no 9
            #print patch
            pVec=patch.flatten() ;#print np.shape(pVec)#matric 1x81
            x64[i,:]=pVec
            i+=1

    #####conv
    for patch in range(nh):
        for kernel in range(nfilter):
            con=Wmat[kernel,:]*x64[patch,:].T#1x81  x  81x1
            con=con[0,0]+bmat[0,kernel]
            con=1.0/(1.0+math.exp((-1.0)*con))
            hMat[kernel,patch]=con
    #####pool
    for k in range(nfilter): #each kernel
        ####1x64->8x8 featmap  , 4x4 poolmap
        feaMap=vec2mat(hMat[k,:],convd,convd) 
        ####pool with 2x2 window mean pooling
        poolMap=np.mat(np.zeros((poold,poold)))
        for hang in range(poold):
            for lie in range(poold):
                patch=feaMap[hang*pwind:hang*pwind+pwind,lie*pwind:lie*pwind+pwind]#pool window 2x2
                v=patch.flatten().mean()
                poolMap[hang,lie]=v
        #####4x4->1x16 poolmap
        hhMat[k,:]=poolMap.flatten()
    #######full connect
    hhvec=hhMat.flatten()#5x16 -> 1x80
    fvec=hhvec*Cmat.T+bbmat#1x80  x  80x10==1x10
    outputMat[x,:]=softmax(fvec)
    ######
     
    return x16
                
def calcGrad(x,x16):#x index not vec
    global hMat,hhMat,outputMat,yMat
    global Cmat,Wmat,bmat,bbmat
    global dataMat
    global gradc,gradw,gradb,gradbb
    global errW,errC,up1,up2
    
    ####err c floor
    fy=outputMat[x,:]-yMat[x,:] #matric 1x10
    sgm=outputMat[x,:].A*(1.0-outputMat[x,:].A)
    errC=np.mat(fy.A*sgm)#matric 1x10
    ######grad c
    hhflat=hhMat.flatten()#1x80 matric
    gradc=errC.T*hhflat  #10x1  x  1x80==10x80
    gradbb=copy.copy(errC)#1x10 ##cannot change  at the same time
    #####5 kernel
    for k in range(nfilter):
        ####calc up1
        vec=errC*Cmat[:,k*nhh:k*nhh+nhh] #1x10  x  10x16==1x16
        up1=vec2mat(vec,poold,poold) #1x16->4x4
        ######calc up2 :upsample: expand and divide 2x2 pooling windon
        for hang in range(poold):
            for lie in range(poold):
                m=up1[hang,lie]/float(pwind**2)
                mat2x2=np.mat(np.zeros((pwind,pwind)))+m#2x2 window filed with mean/4
                up2[pwind*hang:pwind*hang+pwind,pwind*lie:pwind*lie+pwind]=mat2x2
        #####8x8->1x64
        vecUp2=up2.flatten()#matric 1x64
        ####err for w floor
        sgm=hMat[k,:].A*(1.0-hMat[k,:].A)#1x64 array
        errW=np.mat(sgm*vecUp2.A)#1x64 matric
        ######calc w grad :conv2(x,errw,valid) 16x16 conv with 8x8==9x9
        ####x 16x16->(8x8)x81  conv with patch/filter 8x8
        x81=np.mat(np.zeros((filterdd,nh)))##81 pieces of dim64 patch8x8
        i=0
        for hang in range(filterd):
            for lie in range(filterd):
                patch=x16[hang:hang+convd,lie:lie+convd]  #[0:9]==0,1,2,,,8 no 9
                #print patch
                pVec=patch.flatten() ;#print np.shape(pVec)#matric 1x81
                x81[i,:]=pVec; 
                i+=1
         
        ###conv x with filter 8x8
        gradw[k,:]=errW*x81.T#1x64  x  64x81==1x81
        ########
        gradb[0,k]=errW.sum(1)[0,0]
    ##############gradient normalize
    for k in range(11):
        gradc[k,:]=normalize(gradc[k,:],'vector')
    for k in range(nfilter):
        gradw[k,:]=normalize(gradw[k,:],'vector')
    
def updatePara():
    global Cmat,Wmat,bmat,bbmat
    global gradc,gradw,gradb,gradbb
    Cmat=Cmat+alpha*(-1.0)*gradc
    Wmat=Wmat+alpha*(-1.0)*gradw
    bmat=bmat+alpha*(-1.0)*gradb
    bbmat=bbmat+(-1.0)*alpha*gradbb
    
def calcLoss():
    global Cmat,Wmat,bmat,bbmat
    global outputMat,yMat,dataMat # fk is calculated with old para
    num,dim=np.shape(dataMat)
    loss=0.0
    for n in range(num)[:100]:
        diff=outputMat[n,:]-yMat[n,:]#1x10 mat
        ss=diff*diff.T;ss=ss[0,0]
        loss+=ss
    #print 'least square loss',loss
    return loss
        
        
    
    
    
    
        
        
        
        
    
    
        
    
    

##########################################
def vec2mat(vec,nhang,nlie): #input vec 1x16 ouput matric 4x4
    if nhang!=nlie:print 'num hang must = num lie'
    n=nhang#for example :1x16->4x4
    Mat=np.mat(np.zeros((n,n)))
    for hang in range(n):
        for lie in range(n):
            pos=n*hang+lie
            Mat[hang,lie]=vec[0,pos]
    return Mat
    
    
def shuffleObs():
    global dataMat
    num,dim=np.shape(dataMat) #1394 piece of obs
    order=range(num)[:]  #0-100  for loss calc,101...for train obs by obs ///not work. must use whole set to train
    random.shuffle(order)
    return order
    
    

def softmax(outputMat): #1x10 vec
    vec=np.exp(outputMat)  #1x10  #wh+b
    ss=vec.sum(1);ss=ss[0,0]
    outputMat=vec/(ss+0.000001)
    return outputMat
    
def normalize(vec,opt):
    if opt=='prob': #in order to sum prob=1
        ss=vec.sum(1)[0,0]
        vec=vec/(ss+0.000001)
    if opt=='vector': #in order to mode or length ||vec||=1
        mode=vec*vec.T
        mode=math.sqrt(mode[0,0])
        vec=vec/(mode+0.000001)
    if opt not in ['vector','prob']:
        print 'only vector or prob'
    return vec 

        

                

        
        
            




###################main
loadData()
initialH()
initialPara()
initialErr()
initialGrad()

#####
for ep in range(epoch):
    obsList=shuffleObs()
    alpha/=2.0
    for obs in obsList[:]: #obs=x index not vec
        #obs=random.sample(range(10),1)[0]
        x16=forward(obs)
        loss=calcLoss()#loss calc with old para
        #print loss
        calcGrad(obs,x16)
        updatePara()
    print  'epoch %d loss %f'%(ep,loss)


###output

#####output para w m n c ,b
global Cmat,Wmat,bmat,bbmat
outPath="D://python2.7.6//MachineLearning//plankton-CNN//para"
 
outfile1 = "C.txt"
outfile2 = "W.txt"

outfile3 = "B.txt"
outfile4 = "BB.txt"
 

outPutfile=open(outPath+'/'+outfile1,'w')
n,m=np.shape(Cmat)
for i in range(n):
    for j in range(m):
        outPutfile.write(str(Cmat[i,j]))
        outPutfile.write(' ')
    outPutfile.write('\n')
    
outPutfile.close()
##
outPutfile=open(outPath+'/'+outfile2,'w')
n,m=np.shape(Wmat)
for i in range(n):
    for j in range(m):
        outPutfile.write(str(Wmat[i,j]))
        outPutfile.write(' ')
    outPutfile.write('\n')
    
outPutfile.close()
###
outPutfile=open(outPath+'/'+outfile3,'w')
n,m=np.shape(bmat)

for j in range(m):
    outPutfile.write(str(bmat[0,j]))
    outPutfile.write(' ')
outPutfile.write('\n')
    
outPutfile.close()
## 
outPutfile=open(outPath+'/'+outfile4,'w')
n,m=np.shape(bbmat)

for j in range(m):
    outPutfile.write(str(bbmat[0,j]))
    outPutfile.write(' ')
outPutfile.write('\n')
    
outPutfile.close()


 
 

 
    
         
    
    
    







    
    
