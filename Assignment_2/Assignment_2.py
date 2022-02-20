import numpy as np
from sklearn.model_selection import train_test_split
X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
ytr = np.load("age_regression_ytr.npy")
X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
yte = np.load("age_regression_yte.npy")
print(np.shape(X_tr))
#hyperparamter tuning function which returns fmse value
def SGD_hyperparameter(X_train,y_train,X_test,y_test, ep, n, alpha, l_rate): 
    w = np.ones(shape = np.shape(X_train)[1]) #initialise w
    b = 0 #initialise b
    
    for e in range(ep): #epoch 
        random_indices = np.random.permutation(len(X_train)) #randomize training set
        X_new = X_train[random_indices]
        y_new = y_train[random_indices]
        for i in range(int((np.shape(X_train)[0]/n))): #create batches and train
            temp_X = X_new[(i*n):((i+1)*n),:] #batch creation of X
            temp_Y = y_new[(i*n):((i+1)*n)] #batch creation of Y

            y_pred = np.dot(temp_X,w) + b #calculate y_pred
            der_w = ((np.dot(np.transpose(temp_X),(y_pred - temp_Y)))/len(temp_X)) + (alpha/len(temp_X))*w #derivate of fmse with respect to w
            der_b = np.sum(y_pred - temp_Y)/ len(temp_X) #derivate of fmse with respect to b

            
            w = w - (l_rate * der_w) #update w value
            b = b - (l_rate * der_b) #update b value

    y_pred = np.dot(X_test,w) + b # predict on validation set 
    fmse = (np.sum(np.square((y_test - y_pred))))/(2*len(y_test)) #calculate fmse for that particular hyperparameter
    return fmse

#function which calculates w and b on complete training dataset and returns them
def SGD(X_tr, ytr, ep, n, alpha, l_rate): 
    w = np.ones(shape = np.shape(X_tr)[1])
    b = 0
    for e in range(ep):
        random_indices = np.random.permutation(len(X_tr))
        X_new = X_tr[random_indices]
        y_new = ytr[random_indices]
        for i in range(int((np.shape(X_tr)[0]/n))):
            temp_X = X_new[(i*n):((i+1)*n),:]
            temp_Y = y_new[(i*n):((i+1)*n)]
            y_pred = np.dot(temp_X,w) + b
            der_w = ((np.dot(np.transpose(temp_X),(y_pred - temp_Y)))/len(temp_X)) + (alpha/len(temp_X))*w
            der_b = np.sum(y_pred -temp_Y )/ len(temp_X)

            w = w - (l_rate * der_w)
            b = b - (l_rate * der_b)
 
    return w,b

#Create grid
epoch = list(range(50,250,50))
batch_size = list(range(100, 250, 50))
alpha_str = [ 1, 5, 25, 50]
learning_rate = [0.001, 0.005,0.0001, 0.0005]
#Variables to store the best hyperparamter value
ep1 = 0
b1 = 0
a1 = 0
l1 = 0

lowest_mse = 99999999999999999

#split X_tr into two sets, training and validation
X_train, X_val, y_train, y_val = train_test_split(X_tr, ytr, test_size=0.2, random_state=42)

#run grid search
for i in epoch:
    for j in batch_size:
        for k in alpha_str:
            for t in learning_rate:
                temp = SGD_hyperparameter(X_train,y_train,X_val,y_val, i, j, k, t) #get fmse value for each hyperaparamter
                if temp < lowest_mse:
                    ep1 = i
                    b1 = j
                    a1 = k
                    l1 = t 
                    lowest_mse = temp
#Best Values of the hyperparameter 
print(ep1,b1,a1,l1)

#Get w and b with best hyperparameters by training on complete training dataset
w,b = SGD(X_tr,ytr, ep1, b1, a1, l1)

#Calculate y_pred on X_te which gives the test fmse value
y_pred = np.dot(X_te,w) + b
fmse = (np.sum(np.square((yte - y_pred))))/(2*len(yte))

#Calculate y_pred on X_tr which gives the training fmse value
y_pred_tr = np.dot(X_tr,w) + b
fmse_tr = (np.sum(np.square((ytr - y_pred_tr))))/(2*len(ytr))

print('Train: ',fmse_tr)
print('Test: ',fmse)