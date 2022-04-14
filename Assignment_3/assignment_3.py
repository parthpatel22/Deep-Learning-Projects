from matplotlib.pyplot import axis
import numpy as np
from sklearn.model_selection import train_test_split
X_tr = np.load("fashion_mnist_train_images.npy")
ytr = np.load("fashion_mnist_train_labels.npy")
X_te = np.load("fashion_mnist_test_images.npy")
yte = np.load("fashion_mnist_test_labels.npy")

n_values = np.max(ytr) + 1
ytr = np.eye(n_values)[ytr]
n_values = np.max(yte) + 1
yte1 = np.eye(n_values)[yte]

X_train, X_test, y_train, y_test = train_test_split(X_tr, ytr, test_size=0.2, random_state=42)

def softmax_hyperparameters(X_train,y_train,X_test,y_test,classes, ep, n, alpha, l_rate):
    w = np.full((np.shape(X_train)[1],classes),0.00001) #initialise w
    b = np.ones(shape = classes)

    for e in range(ep): #epoch 
        random_indices = np.random.permutation(len(X_train)) #randomize training set
        X_new = X_train[random_indices]
        y_new = y_train[random_indices]
        for i in range(int((np.shape(X_train)[0]/n))): #create batches and train
            temp_X = X_new[(i*n):((i+1)*n),:] #batch creation of X
            temp_Y = y_new[(i*n):((i+1)*n),:] #batch creation of Y
            

            
            y_pred = np.exp(np.dot(temp_X,w)+b)
            total = np.sum(y_pred,axis=1)
            y_pred = np.divide(y_pred.T,total).T
        
            
            der_w = -((np.dot(np.transpose(temp_X),(temp_Y-y_pred)))/len(temp_X)) + (alpha)*w #derivate of fmse with respect to w
            der_b = -(np.sum(temp_Y-y_pred,axis=0)/ len(temp_X)) #derivate of fmse with respect to b

            
            w = w - (l_rate * der_w) #update w value
            b = b - (l_rate * der_b) #update b value
            
            
    
    y_pred = np.exp(np.dot(X_test,w)+b)
    total = np.sum(y_pred,axis=1)
    y_pred = np.divide(y_pred.T,total).T # predict on validation set 
    ly_pred = np.log(y_pred)
    print(y_test.shape)
    print(ly_pred.shape)
    fce =  - (np.sum(np.multiply(y_test,ly_pred))/len(y_test))
    return fce,w,b

#Create grid
epoch = [40,80,120,160]
batch_size = [100,200,300,400]
alpha_str = [ 1, 5, 15, 50]
learning_rate = [0.0001, 0.005,0.00001, 0.0005]
#Variables to store the best hyperparamter value
ep1 = 20
b1 = 100
a1 = 5
l1 = 0.00001

lowest_fce = 99999999999999999

#run grid search
for i in epoch:
    for j in batch_size:
        for k in alpha_str:
            for t in learning_rate:
                temp = softmax_hyperparameters(X_train,y_train,X_test,y_test,10, i, j, k, t)[0]#get fmse value for each hyperaparamter
                if temp < lowest_fce:
                    ep1 = i
                    b1 = j
                    a1 = k
                    l1 = t 
                    lowest_fce = temp

print("Epoch:",ep1)
print("Batch_size:",b1)
print("Regularization:",a1)
print("Learning Rate:",l1)

fce_train,w,b= (softmax_hyperparameters(X_tr,ytr,X_tr,ytr,10, ep1, b1, a1, l1))
print("FCE train cost:",fce_train)
y_pred = np.exp(np.dot(X_te,w)+b)
total = np.sum(y_pred,axis=1)
y_pred = np.divide(y_pred.T,total).T # predict on validation set 
ly_pred = np.log(y_pred)
fce_test =  - (np.sum(np.multiply(yte1,ly_pred))/len(yte))
maxInRows = np.argmax(ly_pred, axis=1)
temp = 0
for i in range(len(yte)):
    if maxInRows[i] == yte[i]:
        temp+=1

print("FCE test cost:",fce_test)
print("Correctly classified:",temp/len(yte)*100,"%")
