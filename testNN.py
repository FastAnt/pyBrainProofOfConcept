from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection , IdentityConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
import math
import os.path
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

def openTrainData():
    fileX = open("trainX" , 'r')
    trainX = pickle.load(fileX)
    fileX.close()
    fileY = open("trainY",'r')
    TrainY = pickle.load(fileY)
    fileY.close()
    return  trainX, TrainY

def prepareDataSet():
    dataSet =  SupervisedDataSet(120, 120)
    trainX , trainY = openTrainData()
    for i in range(100):
        xSample = np.reshape(trainX[i],120)
        ySample = np.reshape(trainY[i],120)
        dataSet.addSample(xSample, ySample)
    #print dataSet
    return dataSet


def neuroNetworkAlgorithm():
    if (os.path.isfile('nnWeights2')) :
        fileObject = open('nnWeights2','r')
        net = pickle.load(fileObject)
        n = net
        n.sorted = False
        n.sortModules()
        fileObject.close()
        print "download nn weights"
    else:
        n = FeedForwardNetwork()
        inLayer = LinearLayer(120)
        hiddenLayer  = LinearLayer(200*3)
        hiddenLayer2 = LinearLayer(200*3)
        hiddenLayer3 = LinearLayer(200*3)
        hiddenLayer4 = SigmoidLayer(200*3)
        outLayer = LinearLayer(120)
        n.addInputModule(inLayer)

        n.addModule(hiddenLayer)
        n.addModule(hiddenLayer2)
        n.addModule(hiddenLayer3)
        n.addModule(hiddenLayer4)

        n.addOutputModule(outLayer)
        in_to_hidden            = FullConnection(inLayer, hiddenLayer)
        hidden_to_hidden2       = FullConnection(hiddenLayer, hiddenLayer2)
        hidden2_to_hidden3      = FullConnection(hiddenLayer2, hiddenLayer3)
        hidden3_to_hidden4      = IdentityConnection(hiddenLayer3, hiddenLayer4)
        hidden4_to_out          = FullConnection(hiddenLayer4, outLayer)

        n.addConnection(hidden4_to_out)
        n.addConnection(hidden3_to_hidden4)
        n.addConnection(in_to_hidden)
        n.addConnection(hidden_to_hidden2)
        n.addConnection(hidden2_to_hidden3)
        n.sortModules()
    dataSet = prepareDataSet()
    trainer = BackpropTrainer(n, dataSet,0.001)
    for i in range(15):
        trainer.trainEpochs(2)
        print("ERROR")
        print(trainer.train())
    dataSet = prepareDataSet()
    trainer = BackpropTrainer(n, dataSet,0.0005)
    for i in range(15):
        trainer.trainEpochs(1)
        print("ERROR")
        print(trainer.train())
    dataSet = prepareDataSet()
    trainer = BackpropTrainer(n, dataSet, 0.0001)
    for i in range(15):
        trainer.trainEpochs(1)
        print("ERROR")
        print(trainer.train())

    fileObject = open('nnWeights2', 'w')
    pickle.dump(n, fileObject)
    fileObject.close()

def generateOneX(x_0,v,t,a):
    trainX = np.zeros((40,3))
    indexArray = x_0
    trainX[indexArray] = [1,1,1]
    return trainX

def generateOneY(x_0,v,a,t):
    trainY = np.zeros((40, 3))
    index = int(x_0 + v*t + a * t*t / 2)
    trainY[index] = [v,a,t]
    return trainY

def testRes():
    print generateOneX(4,1,1,1)
    testData = generateOneX(1,1,1,1)
    if (os.path.isfile('nnWeights2')) :
        fileObject = open('nnWeights2','r')
        net = pickle.load(fileObject)
        n = net
        n.sorted = False
        n.sortModules()
        fileObject.close()
        print "download nn weights"
        testRes = n.activate(testData.reshape(120))
        print testRes.reshape(40,3)
        testDataRes =  generateOneY(1,1,1,1)
        print "TRUE RES"
        print (testDataRes)
        print "DIFF===================================="
        dif =  testRes.reshape(40,3) - testDataRes.reshape(40,3)
        print (dif)




testRes()
#neuroNetworkAlgorithm()