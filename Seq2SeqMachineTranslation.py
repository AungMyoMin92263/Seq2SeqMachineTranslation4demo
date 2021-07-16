# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 22:33:34 2021

@author: aMM
"""

import spacy
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import torch.optim as optim
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter

from utils import translate_sentence,bleu, save_checkpoint, load_checkpoint


spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


english = Field(sequential=True, 
                use_vocab=True, tokenize=tokenize_eng, lower=True)
german = Field(sequential=True, 
                use_vocab=True, tokenize=tokenize_ger, lower=True)



train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields= (german, english))
# exts = extension tells us that source language and target language
                                                         
english.build_vocab(train_data, max_size=10000,min_freq=2)
german.build_vocab(train_data, max_size=10000,min_freq=2)



    
    
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout)
        
        
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        
    def forward(self, x):
        # shape of x = (seq_length, batch)
        embedding = self.dropout(self.embedding(x))
        # shape of embedding = (seq_length, batch, embedding_size)
        
        outputs, (hidden, cell) = self.rnn(embedding)
        
        return hidden, cell
        
        


class DecoderRNN(nn.Module):
    def __init__(self, input_size,embedding_size, hidden_size, num_layers, output_size, p):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        self.embedding_size(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        
        # shape of x: (batch) but we want (1, batch) each word of the first prediction 
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        # embedding shape : (1,Batch,embedding_size)
        
        outputs, (hidden, cell) = self.rnn(embedding, hidden, cell )
        # shape of outputs: (1, batch, hidden_size)
        
        predictions = self.fc(outputs)
        # shape of prediction (1,batch, length_of_eng_vocab)
        predictions = preditions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, source, target, teacher_force_ratio= 0.5): # take prediction value 50% 
        batch_size = source.shape[1]   # To build the startind output
        target_length = target.shape[0] # for loop through the target_len
        target_vacab_size = len(english.vocab)
        hidden, cell = self.encoder(source)
        outputs = torch.zeros(target_length, batch_size, target_vacab_size).to(device)
        
        for t in range(1, target_length):
            output, hidden, cell = self.DecoderRNN(x, hidden, cell)
            
            outputs[t] = output
            
            # best guess 
            best_guess = output.argmax(1)
            
            x = target[t] if random.random() < teacher_force_ratio else best_guess
            
        return outputs



# Training Hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)

output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensor board
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator,  valiation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key= lambda x:len(x.src), device = device)


encoder_net = EncoderRNN(input_size_encoder, encoder_embedding_size,
                      hidden_size, num_layers, enc_dropout)

decoder_net = DecoderRNN(input_size_decoder,decoder_embedding_size, hidden_size,
                      num_layers, output_size,p)
                      

model = Seq2Seq(encoder_net, decoder_net)
optimizer = optmin.Adam(model.parameters(), lr = learning_ratee)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)


for epoch in range(num_epochs):
    print(f'Epoch [{epoch} / {num_epochs}')
    
    checkpoint = {'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()}
    
    save_chechpoints(checkpoint)
    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        
        output = model(inp_data, target)
        
        # output shape: ()
        
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        
        optimizer.zero_grad()
        loss = criterion(output, target)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm =1)
        optimizer.step()
        
        writer.add_scalars('Training Loss', loss, global_step =step)
        step+=1










        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    