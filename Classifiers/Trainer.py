import torch
from torch import nn
import collections
from torch import optim

#model interface

class ModelTrainer(object):


    def __init__(self,net):
        self.net=net
        # optmizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)

        self.crossEntropy = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()

    def getLoss(self,X,Y):
        #print("numpy original",Y)

        #generate loss
        inputVars1=torch.from_numpy(X[0])
        inputVars2=torch.from_numpy(X[1])

        targetVars=torch.from_numpy(Y)

        predicts=self.net(inputVars1,inputVars2)
        targetVars=targetVars.long()
        #print(predicts.size(),targetVars.size())
        #print("torch targets",targetVars)
        #print("torch predicts",predicts)
        #print("predicts",torch.argmax(predicts,dim=1))
        #print("correct{}".format(torch.sum(targetVars==torch.argmax(predicts,dim=1))))
        loss=self.crossEntropy(predicts,targetVars)

        return loss

    def optimizeModel(self):
        #update parameters
        self.optimizer.step()

    def zeroGrad(self):
        self.optimizer.zero_grad()

