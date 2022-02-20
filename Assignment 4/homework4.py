import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
# For this assignment, assume that every hidden layer has the same number of neurons.
NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 30
NUM_OUTPUT = 10
epoch = 100
b1 = 256
a1 = 5
lr1 = 0.001
# Unpack a list of weights and biases into their individual np.arrays.
def unpack (weightsAndBiases):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)

    return Ws, bs

def forward_prop (x, y, weightsAndBiases):
    Ws, bs = unpack(weightsAndBiases)
    zs = []
    hs = []
   
    x=x.T
    y=y.T
    zs.append((np.dot(Ws[0],(x)).T + bs[0]).T)
    z = zs[0]
    z[z<0]=0
    hs.append(z)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        
        zs.append((np.dot(Ws[i+1],(hs[i])).T + bs[i+1]).T)
        z = zs[i+1]
        z[z<0]=0
        hs.append(z)
   
    y_pred = np.exp((np.dot(Ws[-1],(hs[-1])).T + bs[-1]).T)
    y_pred = y_pred.T
    total = np.sum(y_pred,axis=1)
    yhat = np.divide(y_pred.T,total).T
    yhat = yhat.T
    ly_pred = np.log(yhat)

    loss =  - (np.sum(np.multiply(y,ly_pred))/y.shape[1])
    return loss, zs, hs, yhat
   
def back_prop (x, y, weightsAndBiases):
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases)
    Ws, bs = unpack(weightsAndBiases)
    dJdWs = []  # Gradients w.r.t. weights
    dJdbs = []  # Gradients w.r.t. biases
    x=x.T
    y=y.T
    
    # TODO
    g = (yhat - y)/y.shape[1]
    for i in range(NUM_HIDDEN_LAYERS, -1, -1):
        if i!=0:
            dJdWs.append(np.dot(g,hs[i-1].T))
            dJdbs.append((np.sum(g,axis=1)))
    
            relu = zs[i-1]
            relu[relu>0]=1
            relu[relu<=0] = 0
            g = np.multiply(relu,np.dot(Ws[i].T,g))
            
        else:
            dJdWs.append(np.dot(g,x.T))
            dJdbs.append(np.sum(g,axis=1))
    
        # TODO
    dJdWs = dJdWs[::-1]
    dJdbs = dJdbs[::-1]
    # Concatenate gradients
    
    return np.hstack([ dJdW.flatten() for dJdW in dJdWs ] + [ dJdb.flatten() for dJdb in dJdbs ]) 

def train (trainX, trainY, weightsAndBiases, testX, testY):
    trajectory = []
    for e in range(epoch):
        random_indices = np.random.permutation(len(trainX)) #randomize training set
        X_new = trainX[random_indices]
        y_new = trainY[random_indices]
        for p in range(int((np.shape(trainX)[0]/b1))): #create batches and train
            temp_X = X_new[(p*b1):((p+1)*b1),:] #batch creation of X
            temp_Y = y_new[(p*b1):((p+1)*b1),:]
            derweightsAndBiases = back_prop(temp_X,temp_Y, weightsAndBiases)
            dWs, dbs = unpack(derweightsAndBiases)
            ws,bs = unpack(weightsAndBiases)
            for t in range(len(ws)):
                ws[t] = ws[t] - (lr1*(dWs[t] + (a1*ws[t]/ws[t].shape[0])))
                bs[t] = bs[t] - (lr1*(dbs[t]))
                # print(t)
            weightsAndBiases = np.hstack([ W.flatten() for W in ws ] + [ b.flatten() for b in bs ])
        trajectory.append(weightsAndBiases)
        if e>=epoch-20:
            loss = forward_prop(trainX,trainY,weightsAndBiases)[0]
            print("Epoch "+str(e)+":",loss)
        
        # TODO: implement SGD.
        # TODO: save the current set of weights and biases into trajectory; this is
        # useful for visualizing the SGD trajectory.
        
    return weightsAndBiases, trajectory

# Performs a standard form of random initialization of weights and biases
def initWeightsAndBiases ():
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)
    
    
    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    # print(np.shape(Ws))
    # print(np.shape(bs))
    return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])

def plotSGDPath (trainX, trainY, trajectory):
    # TODO: change this toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed 
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    pca = PCA(n_components=2)
    trajectory = np.array(trajectory)
    PCA_WB = pca.fit_transform(trajectory)

    def toyFunction (x1, x2):
        invWB = pca.inverse_transform([x1, x2])
        loss = forward_prop(trainX, trainY, invWB)[0]
        return loss

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-20, 20, 2)
    axis2 = np.arange(-20, 20, 2)
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # # # Now superimpose a scatter plot showing the weights during SGD.
    ax1 =[]
    ax2 = []
    for i in range(len(PCA_WB)):
        ax1.append(PCA_WB[i][0])
        ax2.append(PCA_WB[i][1])
    Xaxis=np.array(ax1)
    Yaxis=np.array(ax2)
    Zaxis = []
    for i in range(len(Xaxis)):
        Zaxis.append(toyFunction(Xaxis[i], Yaxis[i]))
    Zaxis = np.array(Zaxis)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')
    
  

    plt.show()

def findBestHyperparameters(train_X,train_Y,ValX,ValY,weightsAndBiases):
    nhl = [3,4,5]
    nh = [30,40,50]
    epoch = 250
    batch_size = [16,32,64,128]
    alpha = [1,5,10]
    learning_rate = [0.001,0.0001,0.0005,0.005]
    lowest_fce = 99999999999999999
    global NUM_HIDDEN_LAYERS
    global NUM_HIDDEN
    for i in nhl:
        for j in nh:
            for b2 in batch_size:
                for a in alpha:
                    for lr in learning_rate:
                        
                        NUM_HIDDEN_LAYERS = i
                        NUM_HIDDEN = j
                        weightsAndBiases = initWeightsAndBiases()
                        for e in range(epoch):
                            random_indices = np.random.permutation(len(train_X)) #randomize training set
                            X_new = train_X[random_indices]
                            y_new = train_Y[random_indices]
                            for p in range(int((np.shape(train_X)[0]/b2))): #create batches and train
                                temp_X = X_new[(p*b2):((p+1)*b2),:] #batch creation of X
                                temp_Y = y_new[(p*b2):((p+1)*b2),:]
                                derweightsAndBiases = back_prop(temp_X,temp_Y, weightsAndBiases)
                                dWs, dbs = unpack(derweightsAndBiases)
                                ws,bs = unpack(weightsAndBiases)

                                for t in range(len(ws)):
                                    ws[t] = ws[t] - (lr*(dWs[t] + (a*ws[t]/ws[t].shape[0])))
                                    bs[t] = bs[t] - (lr*(dbs[t]))
                                
                                weightsAndBiases = np.hstack([ W.flatten() for W in ws ] + [ b.flatten() for b in bs ])
                        loss = forward_prop(ValX, ValY, weightsAndBiases)[0]
                        if loss<lowest_fce:
                            lowest_fce = loss
                            n1 = i
                            n2= j
                            b1=b2
                            a1=a
                            lr1=lr

    return n1,n2,epoch,b1,a1,lr1
                        

if __name__ == "__main__":
    # TODO: Load data and split into train, validation, test sets
    # trainX = ...
    # trainY = ...
    # ...
    trainX = np.load("fashion_mnist_train_images.npy")
    trainY = np.load("fashion_mnist_train_labels.npy")
    testX= np.load("fashion_mnist_test_images.npy")
    testY= np.load("fashion_mnist_test_labels.npy")
    n_values = np.max(trainY) + 1
    trainY = np.eye(n_values)[trainY]
    n_values = np.max(testY) + 1
    yte1 = np.eye(n_values)[testY]
    train_X,ValX,train_Y,ValY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)
    weightsAndBiases = initWeightsAndBiases()
    # Perform gradient check on 5 training examples
    
    print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[0:5,:]), np.atleast_2d(trainY[0:5,:]), wab)[0], \
                                    lambda wab: back_prop(np.atleast_2d(trainX[0:5,:]), np.atleast_2d(trainY[0:5,:]), wab), \
                                    weightsAndBiases))
    

    NUM_HIDDEN_LAYERS,NUM_HIDDEN,epoch,b1,a1,lr1 =findBestHyperparameters(train_X,train_Y,ValX,ValY,weightsAndBiases)
    print("Best HyperParametrs:")
    print("Number of Hidden Layers:",NUM_HIDDEN_LAYERS)
    print("Number of Units in Each hidden layer:",NUM_HIDDEN)
    print("Epoch:", epoch)
    print("Batch size:",b1)
    print("Regularization term:",a1)
    print("Learning Rate:", lr1)
    weightsAndBiases = initWeightsAndBiases()
    weightsAndBiases, trajectory = train(trainX, trainY, weightsAndBiases, testX, testY)
    test_loss,z,h,yhat = forward_prop(testX, yte1, weightsAndBiases)
    train_loss = forward_prop(trainX,trainY,weightsAndBiases)[0]
    ly_pred = np.log(yhat)
    maxInRows = np.argmax(ly_pred, axis=0)
    temp = 0
    for i in range(len(testY)):
        if maxInRows[i] == testY[i]:
            temp+=1
    print("Train_FCE_Loss:", train_loss)
    print("Test_FCE_Loss:", test_loss)
    print("Correctly classified:",temp/len(testY)*100,"%")
    
    # Plot the SGD trajectory
    plotSGDPath(trainX[:2500,:], trainY[:2500,:], trajectory)
