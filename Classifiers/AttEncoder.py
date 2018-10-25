# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import collections
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os

from Classifiers.Trainer import ModelTrainer

#encoder model
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, num_words, n_layers=1, dropout=0.0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_words,hidden_size)

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, hidden=None):
        # Convert word indexes to embeddings
        input_seq=input_seq.long()
        #print(input_seq.size(),self.embedding)
        embedded = self.embedding(input_seq)
        # Forward pass through GRU
        outputs, hidden = self.gru(embedded, hidden)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

# Luong attention model
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        #attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

#combine encoder and attention models
class AttnEncoderNet(nn.Module):
    def __init__(self, encoder_model:EncoderRNN, attn_model :str, hidden_size, output_size=7, n_layers=1, dropout=0.1):
        super(AttnEncoderNet, self).__init__()

        # model vars
        self.encoder_model=encoder_model

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.hidden_layer = nn.Linear(hidden_size , hidden_size//2)
        self.out = nn.Linear(hidden_size//2, output_size)

        self.attn = Attn(attn_model, hidden_size)

        #define optimzer
        self.optimizerEncoder=optim.Adam(self.encoder_model.parameters(),lr=0.001)
        self.optimizerSelf=optim.Adam(self.parameters(),lr=0.0001)

        #metrics
        self.crossEntropy = torch.nn.CrossEntropyLoss()


    def forward(self, input_seq):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        encoder_outputs, encoder_hidden = self.encoder_model(input_seq)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden_init = encoder_hidden[:self.n_layers]

        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(encoder_outputs, decoder_hidden_init)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs)
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        hiddenLayer=self.hidden_layer(context)
        # Predict next word using Luong eq. 6
        output = self.out(hiddenLayer)
        output=output.squeeze(1)
        # Return output
        return output

    def trainMode(self):
        self.train()
        self.encoder_model.train()

    def evalMode(self):
        self.eval()
        self.encoder_model.eval()

    def optimizeModel(self):
        #update parameters

        self.optimizerEncoder.step()
        self.optimizerSelf.step()

    def zeroGrad(self):
        self.optimizerSelf.zero_grad()
        self.optimizerEncoder.zero_grad()


    def saveModel(self,voc,embedding):

        file_path = "models/en-attn-net"
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        torch.save({
                'en': self.encoder_model.state_dict(),
                'net':self.state_dict(),
                'en_opt': self.optimizerEncoder.state_dict(),
                'net_opt': self.optimizerSelf.state_dict(),
                'voc_dict': voc.__dict__,
                'embedding': self.encoder_model.embedding.embedding.state_dict()
            }, file_path)

    def loadModel(self):

        file_path = os.path.join("../models/en-attn-net({}-{}_{}).torchmodel")
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        saved_data=torch.load(file_path)

        self.encoder_model.load_state_dict(saved_data["en"])
        self.load_state_dict(saved_data["net"])
        self.optimizerEncoder.load_state_dict(saved_data["en_opt"])
        self.optimizerSelf.load_state_dict(saved_data["net_opt"])
        self.encoder_model.embedding.load_state_dict(saved_data["embedding"])

