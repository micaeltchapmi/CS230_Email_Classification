#source: https://houxianxu.github.io/implementation/SVM.html
#source: https://www.kaggle.com/code/zaiyankhan/zaiyan-svm-with-cifar-10/script

import numpy as np
import time
import argparse
import _init_paths
import matplotlib.pyplot as plt
from data_utils import get_CIFAR10_data, preprocessing_CIFAR10_data, load_CIFAR10, visualize_sample

# Load Cifar 10 dataset
parser = argparse.ArgumentParser(description='Train classification network')
args = parser.parse_args()
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = get_CIFAR10_data()
#visualize_sample(X_train_raw, y_train_raw, classes)
X_train, y_train, X_val, y_val, X_test, y_test = preprocessing_CIFAR10_data(args, X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw)

W, H, C = 32, 32, 3

# add bias dimension and reshape data into rows (N, W*H*C)
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])


class Svm (object):
    """" Svm classifier """

    def __init__ (self, inputDim, outputDim):
        self.W = None

        sigma =0.01
        self.W = sigma * np.random.randn(inputDim,outputDim)


    def calLoss (self, x, y, reg):

        loss = 0.0
        dW = np.zeros_like(self.W)

        s = x.dot(self.W)
        #Score with yi
        s_yi = s[np.arange(x.shape[0]),y]
        #finding the delta
        delta = s- s_yi[:,np.newaxis]+1
        #loss for samples
        loss_i = np.maximum(0,delta)
        loss_i[np.arange(x.shape[0]),y]=0
        loss = np.sum(loss_i)/x.shape[0]
        #Loss with regularization
        loss += reg*np.sum(self.W*self.W)
        #Calculating ds
        ds = np.zeros_like(delta)
        ds[delta > 0] = 1
        ds[np.arange(x.shape[0]),y] = 0
        ds[np.arange(x.shape[0]),y] = -np.sum(ds, axis=1)

        dW = (1/x.shape[0]) * (x.T).dot(ds)
        dW = dW + (2* reg* self.W)
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW

    def train (self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iter):
            xBatch = None
            yBatch = None

            num_train = np.random.choice(x.shape[0], batchSize)
            xBatch = x[num_train]
            yBatch = y[num_train]
            loss, dW = self.calLoss(xBatch,yBatch,reg)
            self.W= self.W - lr * dW
            lossHistory.append(loss)

            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x):

        yPred = np.zeros(x.shape[0])

        s = x.dot(self.W)
        yPred = np.argmax(s, axis=1)

        return yPred


    def calAccuracy (self, x, y):
        acc = 0

        yPred = self.predict(x)
        acc = np.mean(y == yPred)*100

        return acc


def main():
    numClasses = len(classes)

    print ('Start training Svm classifier')

    classifier = Svm(X_train.shape[1], numClasses)

    # Show weight for each class before training
    if classifier.W is not None:
        tmpW = classifier.W[:-1, :]
        tmpW = tmpW.reshape(32, 32, 3, 10)
        tmpWMin, tmpWMax = np.min(tmpW), np.max(tmpW)
        for i in range(numClasses):
            plt.subplot(2, 5, i+1)
            plt.title(classes[i])
            wPlot = 255.0 * (tmpW[:, :, :, i].squeeze() - tmpWMin) / (tmpWMax - tmpWMin)
            plt.imshow(wPlot.astype('uint8'))
        plt.clf()

    # Training classifier
    startTime = time.time()
    classifier.train(X_train, y_train, lr=1e-7, reg=5e4, iter=1500 ,verbose=True)
    print ('Training time: {0}'.format(time.time() - startTime))
    print ('Training acc:   {0}%'.format(classifier.calAccuracy(X_train, y_train)))
    print ('Validating acc: {0}%'.format(classifier.calAccuracy(X_val, y_val)))
    print ('Testing acc:    {0}%'.format(classifier.calAccuracy(X_test, y_test)))


if __name__=="__main__":
    main()