from keras.datasets import cifar10
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
(x_train, y_Train), (x_test, y_Test) = cifar10.load_data()
cost_f=[]
acc=[]
G=[]
acctr=[]
def Relu(Z):
    Relu=np.where(Z<=0, 0, Z)
    return Relu
def dReLU(x):
    Dr = np.where(x<0, 0, 1)
    return Dr
def sigmoid(x):
    X=np.exp(x)
    S=sum(X)
    score=X/S
    return score
     #Convolution Layer
def Conv_layer(image,fil1,fil2,fil3,fil4,fil5,fil6,fil7,fil8,fil9,fil10):
    image_d,image_h,image_w=np.shape(image)
    fil_dep,fil_h,fil_w=np.shape(fil1)
    fil_2=fil2
    h=0
    l=np.empty((0,1))
    while(h<=image_h-fil_h):
        w=0
        while(w<=image_w-fil_w):
            dumy=image[:,h:h+fil_h,w:w+fil_w].flatten()
            dumy=np.array(dumy.reshape(fil_h*fil_w*image_d,1))
            l=np.append(l,[dumy])
            w=w+1
        h=h+1 #this+1 is stride
    i=np.reshape(l,[(image_h-fil_h+1)*(image_w-fil_w+1),fil_h*fil_w*image_d])
    i=i.T
    fil_1=np.reshape(fil1.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_2=np.reshape(fil2.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_3=np.reshape(fil3.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_4=np.reshape(fil4.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_5=np.reshape(fil5.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_6=np.reshape(fil6.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_7=np.reshape(fil7.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_8=np.reshape(fil8.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_9=np.reshape(fil9.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_10=np.reshape(fil10.flatten(),[1,fil_dep*fil_h*fil_w])
    c=np.concatenate((fil_1,fil_2,fil_3,fil_4,fil_5,fil_6,fil_7,fil_8,fil_9,fil_10), axis=0)
    after_conv=np.dot(c,i)
    after_conv=np.reshape(after_conv,[10,image_h-fil_h+1,image_h-fil_h+1])
    return after_conv
def Conv_layer1(image,fil1,fil2,fil3,fil4,fil5,fil6,fil7,fil8,fil9,fil10,fil11,fil12,fil13,fil14,fil15):
    image_d,image_h,image_w=np.shape(image)
    fil_dep,fil_h,fil_w=np.shape(fil1)
    h=0
    l=np.empty((0,1))
    while(h<=image_h-fil_h):
        w=0
        while(w<=image_w-fil_w):
            dumy=image[:,h:h+fil_h,w:w+fil_w].flatten()
            dumy=np.array(dumy.reshape(fil_h*fil_w*image_d,1))
            l=np.append(l,[dumy])
            w=w+1
        h=h+1 #this+1 is stride
    i=np.reshape(l,[(image_h-fil_h+1)*(image_w-fil_w+1),fil_h*fil_w*image_d])
    i=i.T
    fil_1=np.reshape(fil1.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_2=np.reshape(fil2.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_3=np.reshape(fil3.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_4=np.reshape(fil4.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_5=np.reshape(fil5.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_6=np.reshape(fil6.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_7=np.reshape(fil7.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_8=np.reshape(fil8.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_9=np.reshape(fil9.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_10=np.reshape(fil10.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_11=np.reshape(fil11.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_12=np.reshape(fil12.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_13=np.reshape(fil13.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_14=np.reshape(fil14.flatten(),[1,fil_dep*fil_h*fil_w])
    fil_15=np.reshape(fil15.flatten(),[1,fil_dep*fil_h*fil_w])
    c=np.concatenate((fil_1,fil_2,fil_3,fil_4,fil_5,fil_6,fil_7,fil_8,fil_9,fil_10,fil_11,fil_12,fil_13,fil_14,fil_15), axis=0)
    after_conv=np.dot(c,i)
    after_conv=np.reshape(after_conv,[15,image_h-fil_h+1,image_h-fil_h+1])
    return after_conv
def Conv_layerB(image,fil1):
    image_h,image_w=np.shape(image)
    fil_h,fil_w=np.shape(fil1)
    h=0
    l=np.empty((0,1))
    while(h<=image_h-fil_h):
        w=0
        while(w<=image_w-fil_w):
            dumy=image[h:h+fil_h,w:w+fil_w].flatten()
            dumy=np.array(dumy.reshape(fil_h*fil_w,1))
            l=np.append(l,[dumy])
            w=w+1
        h=h+1 #this+1 is stride
    i=np.reshape(l,[(image_h-fil_h+1)*(image_w-fil_w+1),fil_h*fil_w])
    i=i.T
    fil_1=fil1.flatten()
    fil_1=np.reshape(fil_1,[1,fil_h*fil_w])
    after_conv=np.dot(fil_1,i)
    after_conv=np.reshape(after_conv,[image_h-fil_h+1,image_h-fil_h+1])
    return after_conv
def Pooling_Layer(image):  
    image_d,image_h,image_w=np.shape(image)
    fil_h,fil_w=2,2
    h=fil_h//2
    w=fil_w//2
    param=image_h//2-1
    After_Pooling=np.zeros([image_d,image_h//2,image_w//2])
    back_pol=np.zeros([image_d,image_h,image_w])
    for d in range(image_d):
         o=0
         for i in range(h,image_h-param):
            l=0
            for j in range(w,image_w-param):
                temp1=np.zeros([2,2])
                for m in range(fil_h):
                    for n in range(fil_w):
                        temp1[m][n]=image[d][i-h+m+o][j-w+n+l]
                        After_Pooling[d][i-1][j-1]=np.amax(temp1)
                        a1=np.unravel_index(np.argmax(temp1),temp1.shape)
                        z1=a1[0]
                        z2=a1[1]
                back_pol[d][i-h+z1+o][j-w+z2+l]=1
                l=l+1
            o=o+1
    return After_Pooling,back_pol
def Model():
    epoch=1
    LR=0.1
    Gamma=0.9
    s=0.1
    F1_1=np.random.normal(0,s,[3,5,5])
    F1_2=np.random.normal(0,s,[3,5,5])
    F1_3=np.random.normal(0,s,[3,5,5])
    F1_4=np.random.normal(0,s,[3,5,5])
    F1_5=np.random.normal(0,s,[3,5,5])
    F1_6=np.random.normal(0,s,[3,5,5])
    F1_7=np.random.normal(0,s,[3,5,5])
    F1_8=np.random.normal(0,s,[3,5,5])
    F1_9=np.random.normal(0,s,[3,5,5])
    F1_10=np.random.normal(0,s,[3,5,5])
    
    F2_1=np.random.normal(0,s,[10,3,3])
    F2_2=np.random.normal(0,s,[10,3,3])
    F2_3=np.random.normal(0,s,[10,3,3])
    F2_4=np.random.normal(0,s,[10,3,3])
    F2_5=np.random.normal(0,s,[10,3,3])
    F2_6=np.random.normal(0,s,[10,3,3])
    F2_7=np.random.normal(0,s,[10,3,3])
    F2_8=np.random.normal(0,s,[10,3,3])
    F2_9=np.random.normal(0,s,[10,3,3])
    F2_10=np.random.normal(0,s,[10,3,3])
    F2_11=np.random.normal(0,s,[10,3,3])
    F2_12=np.random.normal(0,s,[10,3,3])
    F2_13=np.random.normal(0,s,[10,3,3])
    F2_14=np.random.normal(0,s,[10,3,3])
    F2_15=np.random.normal(0,s,[10,3,3])

    W1=np.random.normal(0,s,[200,540])
    W2=np.random.normal(0,s,[10,200])
    
    Update1W=np.zeros([200,540])
    Update2W=np.zeros([10,200])
    
    Update1f=np.zeros([3,3])
    Update2f=np.zeros([3,3])
    Update3f=np.zeros([3,3])
    Update4f=np.zeros([3,3])
    Update5f=np.zeros([3,3])
    Update6f=np.zeros([3,3])
    Update7f=np.zeros([3,3])
    Update8f=np.zeros([3,3])
    Update9f=np.zeros([3,3])
    Update10f=np.zeros([3,3])
    Update11f=np.zeros([3,3])
    Update12f=np.zeros([3,3])
    Update13f=np.zeros([3,3])
    Update14f=np.zeros([3,3])
    Update15f=np.zeros([3,3])
    
    Update1ff=np.zeros([5,5])
    Update2ff=np.zeros([5,5])
    Update3ff=np.zeros([5,5])
    Update4ff=np.zeros([5,5])
    Update5ff=np.zeros([5,5])
    Update6ff=np.zeros([5,5])
    Update7ff=np.zeros([5,5])
    Update8ff=np.zeros([5,5])
    Update9ff=np.zeros([5,5])
    Update10ff=np.zeros([5,5])
    while(epoch<=10):
        correct=0
        for i in range(500):    
            image=x_train[i,:,:,:]/255
            R=image[:,:,0]
            B=image[:,:,1]
            G=image[:,:,2]
            c=np.concatenate((R,B,G))
            d_img=np.reshape(c,[3,32,32])
            Y=y_Train[i]
            target = np.zeros([10,1]) + 0.01
            target[Y]= 0.99
            #Input Ready
            #Forward Propagation
            First_conv=Conv_layer(d_img,F1_1,F1_2,F1_3,F1_4,F1_5,F1_6,F1_7,F1_8,F1_9,F1_10)           
            First_convS=Relu(First_conv)
            pool_1,mask1=Pooling_Layer(First_convS)
            
            Second_conv=Conv_layer1(pool_1,F2_1,F2_2,F2_3,F2_4,F2_5,F2_6,F2_7,F2_8,F2_9,F2_10,F2_11,F2_12,F2_13,F2_14,F2_15)           
            Second_convS=Relu(Second_conv)
            pool_2,mask2=Pooling_Layer(Second_convS)
            flat_layer=pool_2.flatten() # 16 nodes
            flat_layer=np.reshape(flat_layer,(540,1))
            
            Z1=np.dot(W1,flat_layer)
            A1=Relu(Z1)
            Z2=np.dot(W2,A1)
            A2=sigmoid(Z2)
            output=np.reshape(A2,(10,1))
            
             #Cost FUnction Calculation
            Error=target-output
            cost=0.5*sum((Error)**2)/10
            #Backward Propagation
             #dw2 ????
            sigmoido_grad=np.multiply(output,1-output)
            er_and_sigmd=np.multiply(Error,sigmoido_grad)
            A1_T=A1.T
            dW2=np.dot(er_and_sigmd,A1_T)
            #dW1 ??
            W2_T=W2.T
            er_flat=np.dot(W2_T,er_and_sigmd)
            sigmd_der_flat=dReLU(Z1)
            this=np.multiply(er_flat,sigmd_der_flat)
            flat_layer_T=flat_layer.T
            dW1=np.dot(this,flat_layer_T)
             #error propagated at flatten and pooling layer
            W1_T=W1.T
            err_flatten=np.dot(W1_T,this)
            FD=np.reshape(err_flatten,(15,6,6))
            repeat_2=FD.repeat(2,axis=1).repeat(2,axis=2)
            maxpol2_eror=repeat_2*mask2
            Sig_der_fil2=dReLU(Second_conv)
            err_Conv2=np.multiply(Sig_der_fil2,maxpol2_eror)
            F=[]
            cc=[]
            for n in range(15):
                F1=err_Conv2[n,:,:]
                dF2_1=np.zeros([10,3,3])
                for m in range(10):
                    b=Conv_layerB(pool_1[m,:,:],F1)
                    dF2_1[m,:,:]=b
                F=np.append(F,[dF2_1])
            dF2=np.reshape(F,(15,10,3,3))
            df1=dF2[0,:,:,:]
            df2=dF2[1,:,:,:]
            df3=dF2[2,:,:,:]
            df4=dF2[3,:,:,:]
            df5=dF2[4,:,:,:]
            df6=dF2[5,:,:,:]
            df7=dF2[6,:,:,:]
            df8=dF2[7,:,:,:]
            df9=dF2[8,:,:,:]
            df10=dF2[9,:,:,:]
            df11=dF2[10,:,:,:]
            df12=dF2[11,:,:,:]
            df13=dF2[12,:,:,:]
            df14=dF2[13,:,:,:]
            df15=dF2[14,:,:,:]
            F2_1R=np.rot90(F2_1,2, axes=(2,1))
            F2_2R=np.rot90(F2_2,2, axes=(2,1))
            F2_3R=np.rot90(F2_3,2, axes=(2,1))
            F2_4R=np.rot90(F2_4,2, axes=(2,1))
            F2_5R=np.rot90(F2_5,2, axes=(2,1))
            F2_6R=np.rot90(F2_6,2, axes=(2,1))
            F2_7R=np.rot90(F2_7,2, axes=(2,1))
            F2_8R=np.rot90(F2_8,2, axes=(2,1))
            F2_9R=np.rot90(F2_9,2, axes=(2,1))
            F2_10R=np.rot90(F2_10,2, axes=(2,1))
            F2_11R=np.rot90(F2_11,2, axes=(2,1))
            F2_12R=np.rot90(F2_12,2, axes=(2,1))
            F2_13R=np.rot90(F2_13,2, axes=(2,1))
            F2_14R=np.rot90(F2_14,2, axes=(2,1))
            F2_15R=np.rot90(F2_15,2, axes=(2,1))
            cc=np.append(cc,[F2_1R,F2_2R,F2_3R,F2_4R,F2_5R,F2_6R,F2_7R,F2_8R,F2_9R,F2_10R,F2_11R,F2_12R,F2_13R,F2_14R,F2_15R])
            A_rotated=np.reshape(cc,(15,10,3,3))
            #padding
            f=np.shape(err_Conv2)
            padded_err_Conv2=np.zeros([15,16,16])
            for t in range(f[0]):
                padded_err_Conv2[t,:,:]=np.pad(err_Conv2[t], (2, 2), 'constant', constant_values=(0,0))
            Accumulated_Error=np.zeros([10,14,14])
            for k in range(15):
                F_P=padded_err_Conv2[k,:,:]
                for n in range(10):
                    F_map=A_rotated[k,n,:,:]
                    ER=Conv_layerB(F_P,F_map)
                    Accumulated_Error[n,:,:]=Accumulated_Error[n,:,:]+ER
                
            repeat_1=Accumulated_Error.repeat(2,axis=1).repeat(2,axis=2)
            maxpol1_eror=repeat_1*mask1 
            
            Sig_der_fil1=dReLU(First_conv)
            err_Conv1=np.multiply(Sig_der_fil1,maxpol1_eror)
            #df1 calculation
            FF=[]
            for n in range(10):
                F1=err_Conv1[n,:,:]
                dF2_1=np.zeros([3,5,5])
                for m in range(3):
                    b=Conv_layerB(d_img[m,:,:],F1)
                    dF2_1[m,:,:]=b
                FF=np.append(FF,[dF2_1])
            dF1=np.reshape(FF,(10,3,5,5))
            
            dff1=dF1[0,:,:,:]
            dff2=dF1[1,:,:,:]
            dff3=dF1[2,:,:,:]
            dff4=dF1[3,:,:,:]
            dff5=dF1[4,:,:,:]
            dff6=dF1[5,:,:,:]
            dff7=dF1[6,:,:,:]
            dff8=dF1[7,:,:,:]
            dff9=dF1[8,:,:,:]
            dff10=dF1[9,:,:,:]
            
            Update1W=Gamma*Update1W+(1-Gamma)*dW1
            Update2W=Gamma*Update2W+(1-Gamma)*dW2
    
            Update1f=Gamma*Update1f+(1-Gamma)*df1
            Update2f=Gamma*Update2f+(1-Gamma)*df2
            Update3f=Gamma*Update3f+(1-Gamma)*df3
            Update4f=Gamma*Update4f+(1-Gamma)*df4
            Update5f=Gamma*Update5f+(1-Gamma)*df5
            Update6f=Gamma*Update6f+(1-Gamma)*df6
            Update7f=Gamma*Update7f+(1-Gamma)*df7
            Update8f=Gamma*Update8f+(1-Gamma)*df8
            Update9f=Gamma*Update9f+(1-Gamma)*df9
            Update10f=Gamma*Update10f+(1-Gamma)*df10
            Update11f=Gamma*Update11f+(1-Gamma)*df11
            Update12f=Gamma*Update12f+(1-Gamma)*df12
            Update13f=Gamma*Update13f+(1-Gamma)*df13
            Update14f=Gamma*Update14f+(1-Gamma)*df14
            Update15f=Gamma*Update15f+(1-Gamma)*df15
            
            
            Update1ff=Gamma*Update1ff+(1-Gamma)*dff1
            Update2ff=Gamma*Update2ff+(1-Gamma)*dff2
            Update3ff=Gamma*Update3ff+(1-Gamma)*dff3
            Update4ff=Gamma*Update4ff+(1-Gamma)*dff4
            Update5ff=Gamma*Update5ff+(1-Gamma)*dff5
            Update6ff=Gamma*Update6ff+(1-Gamma)*dff6
            Update7ff=Gamma*Update7ff+(1-Gamma)*dff7
            Update8ff=Gamma*Update8ff+(1-Gamma)*dff8
            Update9ff=Gamma*Update9ff+(1-Gamma)*dff9
            Update10ff=Gamma*Update10ff+(1-Gamma)*dff10

           
            
            W2=W2+LR*Update2W
            W1=W1+LR*Update1W
            
            F1_1=F1_1+LR*Update1ff
            F1_2=F1_2+LR*Update2ff
            F1_3=F1_3+LR*Update3ff
            F1_4=F1_4+LR*Update4ff
            F1_5=F1_5+LR*Update5ff
            F1_6=F1_6+LR*Update6ff
            F1_7=F1_7+LR*Update7ff
            F1_8=F1_8+LR*Update8ff
            F1_9=F1_9+LR*Update9ff
            F1_10=F1_10+LR*Update10ff
            
            F2_1=F2_1+LR*Update1f
            F2_2=F2_2+LR*Update2f
            F2_3=F2_3+LR*Update3f
            F2_4=F2_4+LR*Update4f
            F2_5=F2_5+LR*Update5f
            F2_6=F2_6+LR*Update6f
            F2_7=F2_7+LR*Update7f
            F2_8=F2_8+LR*Update8f
            F2_9=F2_9+LR*Update9f
            F2_10=F2_10+LR*Update10f
            F2_11=F2_11+LR*Update11f
            F2_12=F2_12+LR*Update12f
            F2_13=F2_13+LR*Update13f
            F2_14=F2_14+LR*Update14f
            F2_15=F2_15+LR*Update15f
            #print(Z2)
        cost_f.append(cost)
        print("Cost Function After ", epoch , " Epoch : ",cost)     
        #accuracy calculation
        for j in range(200):
            image1=x_test[j,:,:,:]/255
            R=image1[:,:,0]
            B=image1[:,:,1]
            G=image1[:,:,2]
            c=np.concatenate((R,B,G))
            d_img=np.reshape(c,[3,32,32])
            y=y_Test[j]
            First_conv=Conv_layer(d_img,F1_1,F1_2,F1_3,F1_4,F1_5,F1_6,F1_7,F1_8,F1_9,F1_10)           
            First_convS=Relu(First_conv)
            pool_1,mask1=Pooling_Layer(First_convS)
            
            Second_conv=Conv_layer1(pool_1,F2_1,F2_2,F2_3,F2_4,F2_5,F2_6,F2_7,F2_8,F2_9,F2_10,F2_11,F2_12,F2_13,F2_14,F2_15)           
            Second_convS=Relu(Second_conv)
            pool_2,mask2=Pooling_Layer(Second_convS)
            flat_layer=pool_2.flatten() # 16 nodes
            flat_layer=np.reshape(flat_layer,(540,1))
            Z1=np.dot(W1,flat_layer)
            A1=Relu(Z1)
            Z2=np.dot(W2,A1)
            A2=sigmoid(Z2)
            output=np.reshape(A2,(10,1))
            index = np.argmax(output,axis=0)
            #print(index,y)
            #print(y,index)
            if (index==y):
                correct+=1
            accuracy=(correct/200)
        acc.append(accuracy)
        print("Test Accuracy After ", epoch , " Epoch : ",accuracy)
        for j in range(500):
            image1=x_train[j,:,:,:]/255
            R=image1[:,:,0]
            B=image1[:,:,1]
            G=image1[:,:,2]
            c=np.concatenate((R,B,G))
            d_img=np.reshape(c,[3,32,32])
            y=y_Train[j]
            First_conv=Conv_layer(d_img,F1_1,F1_2,F1_3,F1_4,F1_5,F1_6,F1_7,F1_8,F1_9,F1_10)           
            First_convS=Relu(First_conv)
            pool_1,mask1=Pooling_Layer(First_convS)
            
            Second_conv=Conv_layer1(pool_1,F2_1,F2_2,F2_3,F2_4,F2_5,F2_6,F2_7,F2_8,F2_9,F2_10,F2_11,F2_12,F2_13,F2_14,F2_15)           
            Second_convS=Relu(Second_conv)
            pool_2,mask2=Pooling_Layer(Second_convS)
            flat_layer=pool_2.flatten() # 16 nodes
            flat_layer=np.reshape(flat_layer,(540,1))
            Z1=np.dot(W1,flat_layer)
            A1=Relu(Z1)
            Z2=np.dot(W2,A1)
            A2=sigmoid(Z2)
            output=np.reshape(A2,(10,1))
            index = np.argmax(output,axis=0)
            #print(index,y)
            #print(y,index)
            if (index==y):
                correct+=1
            accuracy1=(correct/500)
        acctr.append(accuracy1)
        print("Train  Accuracy After ", epoch , " Epoch : ",accuracy1)
        epoch =epoch + 1
    return output
mo=Model()
plot=10
p=np.linspace(1,plot,plot)
fig = plt.figure(figsize=(8,8))
plt.plot(p,cost_f,'k')
plt.xlabel('No. of Epochs')
plt.ylabel('Cost Function')
plt.show()
p=np.linspace(1,plot,plot)
fig = plt.figure(figsize=(8,8))
plt.plot(p,acc,'r-',p,acctr,'g-')
plt.xlabel('No. of Epochs')
plt.ylabel(' Accuracy')
plt.show()
