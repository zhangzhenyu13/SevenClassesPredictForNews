from torch import nn
import torch
from torch import optim
import torch.nn.functional as F

#stacking grus

class StackingRNN(nn.Module):

    def __init__(self, embedding_size, hidden_size,classes, num_words, n_layers=1, dropout=0.0):

        super(StackingRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_words,embedding_size)

        self.lstm=nn.LSTM(input_size=embedding_size,hidden_size=hidden_size,
                          num_layers=n_layers,batch_first=True,dropout=dropout,
                          bidirectional=True)

        self.out=nn.Linear(hidden_size,classes)


        print("init stacking rnn")



    def forward(self, input_seq):
        # Convert word indexes to embeddings
        input_seq=input_seq.long()
        #print(input_seq.size(),self.embedding)
        embedded = self.embedding(input_seq)
        # Forward pass through GRU
        outputs, (h_n,c_n) = self.lstm(embedded)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, -1, :self.hidden_size] + outputs[:, -1 ,self.hidden_size:]
        outputs=outputs.squeeze(0)
        out=F.softmax(self.out(outputs),dim=1)
        # Return output and final hidden state
        return out

    def trainMode(self):
        self.train()

    def evalMode(self):
        self.eval()
