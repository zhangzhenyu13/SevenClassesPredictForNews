from torch import nn
import torch
from torch import optim
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self,max_len, embedding_size, classes, num_words):
        super(TextCNN,self).__init__()
        self.max_len=max_len
        self.embedding_size=embedding_size
        self.classes=classes
        self.num_words=num_words

        self.embeding = nn.Embedding(num_words,embedding_size)

        V = self.num_words
        D = self.embedding_size
        C = self.classes
        Ci = 1
        Co = 32
        Ks_short = [2,3,5,7,9]
        Ks_long=[3,5,8,10,15]
        self.embed = nn.Embedding(V, D)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks_short])
        self.convs2 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks_long])


        self.dropout = nn.Dropout(0.5)
        self.all_C=(len(Ks_short)+len(Ks_long)) * Co

        self.fc1 = nn.Linear(self.all_C, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def shortCNN(self,x):
        x = self.embed(x)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        #x = nn.Conv1d(x.size(1), 40, 1)(x)
        x = x.squeeze(2)

        return x

    def longCNN(self, x):
        x = self.embed(x)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs2]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        #x=nn.Conv1d(x.size(1),60,1)(x)
        x=x.squeeze(2)


        return x

    def forward(self, x1,x2=None):
        x1=x1.long()
        if x2 is not None:
            x2=x2.long()

        x1=self.shortCNN(x1)
        if x2 is not None:
            x2=self.longCNN(x2)

        if x2 is not None:
            x=torch.cat([x1,x2],1)
        else:
            x=x1

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = F.softmax(self.fc1(x),dim=1)  # (N, C)
        return logit

    def trainMode(self):
        self.train()

    def evalMode(self):
        self.eval()