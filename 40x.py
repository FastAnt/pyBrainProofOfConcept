import pybrain as pb
import numpy as np
import math
import random
import pickle

velocity_idx     = 0
acceleration_idx = 1
time_idx         = 2
indexArray = np.zeros((100))

def generateOneX(x_0,v,t,a):
    trainX = np.zeros((40,3))
    indexArray = x_0
    trainX[indexArray] = [1,1,1]
    return trainX

def generateTrainX():
    trainX = np.zeros((100,40,3))
    for i  in range(len(trainX)) :
        indexArray[i] = random.randint(0,10)
        trainX[i][indexArray[i]] = [random.uniform(0,3),random.uniform(0,3),random.uniform(0,3)]
    return trainX

def generateTrainY(train_X):
    trainY = np.zeros((100,40,3))
    for i in range(len(trainX)) :
        newX = int(indexArray[i] + \
               train_X[i][indexArray[i]][velocity_idx] * train_X[i][indexArray[i]][time_idx] + \
               train_X[i][indexArray[i]][acceleration_idx] * pow(train_X[i][indexArray[i]][time_idx],2)/2)
        print "=============================================="
        print "newX"
        print newX
        print "indexArray[i]"
        print indexArray[i]
        print "=============================================="
        print "=============================================="
        print "train_X[i][indexArray[i]]"
        print train_X[i][indexArray[i]]
        print "=============================================="
        print "=============================================="
        print "trainY[i][newX]"
        print trainY[i][newX]
        print "=============================================="

        trainY[i][newX] = train_X[i][indexArray[i]]
        print "=============================================="
        print "trainY[i][newX] - new"
        print trainY[i][newX]
        print "=============================================="
    return trainY
def saveTrainData(TrainDataX, TrainDataY):
    fileX = open("trainX" , 'w')
    pickle.dump(TrainDataX, fileX)
    fileY = open("trainY",'w')
    pickle.dump(TrainDataY, fileY)
    fileX.close()
    fileY.close()

def openTrainData():
    fileX = open("trainX" , 'r')
    trainX = pickle.load(fileX)
    fileX.close()
    fileY = open("trainY",'r')
    TrainY = pickle.load(fileY)
    fileY.close()
    return  trainX, TrainY

#trainX = generateTrainX()
#trainY = generateTrainY(trainX)
#saveTrainData(trainX,trainY)

trainX , trainY = openTrainData()

print trainX
print trainY
#def generateTrainY():

