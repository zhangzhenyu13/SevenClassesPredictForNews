import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):

    def __init__(self,dim_size=50):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(dim_size, 30)
        self.fc2 = nn.Linear(30, 7)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def testTrain():
    from DataCleaning.DataLoader import NewsDataset
    from DataCleaning.mediaEncoder import MediaEncoder

    net = Net()
    net.train()

    epochs=1000
    batch_size=50
    onehot=MediaEncoder()
    nwdata=NewsDataset("data/trainex.xls")
    cols=[6]

    for _ in range(epochs):

        batch,labels=nwdata.fetchBatch(cols=cols,batch_size=batch_size)
        x=[]
        for i in range(len(cols)):
            x1=batch[:,i]
            x1=onehot.encodeMedia(x1)
            x.append(x1)
        if len(x)<2:
            x=x[0]
        else:
            x=np.concatenate(x,axis=1)


        x=torch.from_numpy(x)
        y=torch.from_numpy(labels)

        input = x.float()
        target = y.long()

        #print(input.size())
        #print(target.size())

        criterion=nn.CrossEntropyLoss()

        # create your optimizer
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update

        output=torch.argmax(output,1)
        acc=torch.sum(output==target)
        print("\nloss",loss,"acc",acc,"batch size",batch_size)
        #print(output.size(),output)
        #print(target.size(),target)

if __name__ == '__main__':
    testTrain()