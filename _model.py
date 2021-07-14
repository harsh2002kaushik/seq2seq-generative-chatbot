#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 04:15:52 2021

@author: harsh
"""

import tensorflow as tf
import time
import os
import numpy
import pickle

punc = ['.',',',':',';','?','/','[','[',"}","{","<",">","|","-","_","!","@","#","$","%","^","&","*","(",")","`","~","'",'"',"+","="]
alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
nums = ['0','1','2','3','4','5','6','7','8','9']
alphanums = alphabets + nums
contraction_mapping = {"ain't": "is not","it's": "it is","i'm": "i am","i've": "i have", "aren't": "are not","can't": "cannot", "cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
keep_punc = ["!","#","&","*","(",")",":","/",","]

def spaced(sentence):
  k = ''
  for i in str(sentence.lower()).split():
    try:
      if str(i[0]) == '@' and str(i[1]) in alphanums :
        i = 'person'        
    except:
        i = str(i)
    if len(i) >= 2:
      if str(i[-1]) not in alphanums and str(i[-2])  in alphanums:
        i=str(i[:-1])+' '+str(i[-1])
      elif str(i[-1]) not in alphanums and str(i[-2]) not in alphanums:
        for j in range (len(i)):
          if str(i[-1]) not in alphanums:
            i=str(i[:-1])
          else:
            break

    if len(i)>=2:
      if str(i[0]) not in alphanums and str(i[1]) in alphanums:
        i=str(i[0])+' '+str(i[1:])
      elif str(i[0]) not in alphanums and str(i[1]) not in alphanums:
        for j in range (len(i)):
          if str(i[0]) not in alphanums:
            i=str(i[1:])
          else:
            break
            
    if i in contraction_mapping:
      i = contraction_mapping[i]
    k =str(k)+' '+str(i)   
  return k


def preprocess_data(data):
    b= []
    for i in range(len(data)):
        a = '<start> ' + spaced(data[i]) + ' <end>'
        b.append(a)
        
    return b

def preprocess_sequence(sequence):
    b = '<start>' + spaced(sequence) + ' <end>'
    return b
num_words = 23500
embedding_dim = 128        
encoder_units = 512
decoder_units = 1024    
EPOCHS = 100
vocab_size  = num_words + 1
BATCH_SIZE = 64
max_length_inp = 12
max_length_out = 12
tokenizer = pickle.load(open('tokenizer.pkl','rb'))

embedding =  tf.keras.layers.Embedding(vocab_size, embedding_dim)

class Encoder(tf.keras.Model):
    
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
    super(Encoder, self).__init__()
    self.embedding_dim = embedding_dim
    self.enc_units = enc_units
    self.batch_size = batch_size
    lstm_1 = tf.keras.layers.LSTM(self.enc_units, return_sequences=True, return_state=True)
    self.bilstm_1 = tf.keras.layers.Bidirectional(lstm_1, merge_mode='concat')
    lstm = tf.keras.layers.LSTM(self.enc_units, return_sequences=True, return_state=True)           
    self.bilstm = tf.keras.layers.Bidirectional(lstm, merge_mode='concat')

  def call(self, x, hidden):

    x = embedding(x)
    x = self.bilstm_1(x, initial_state = hidden)
    output, forward_h, forward_c, backward_h, backward_c = self.bilstm(x)
    
    
    state_h = tf.concat((forward_h, backward_h), axis=-1)
    state_c = tf.concat((forward_h ,backward_h), axis=-1)
    state = [state_h, state_c]
    state = tf.squeeze(state)
    return output, state_h, state

  def initialize_state(self):
    return [tf.zeros((self.batch_size,self.enc_units )) for i in range(4)]

class BahdanauAttention(tf.keras.layers.Layer):
    
  def __init__(self, units):
    super(BahdanauAttention,self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query,1)

    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis)+self.W2(values)))

    attention_weights = tf.nn.softmax(score, axis=1) 

    context_vector = attention_weights*values
    context_vector = tf.reduce_sum(context_vector, axis=1) 

    return context_vector,attention_weights

class Decoder(tf.keras.Model):
    
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm = tf.keras.layers.LSTM(self.dec_units, return_sequences=True, return_state=True)       
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)

    x = embedding(x)  
    x = tf.concat([tf.expand_dims(context_vector,1), x], axis=-1)  
    output, state_h,state_c =  self.lstm(x)        
    output = tf.reshape(output, (-1, output.shape[2]))   
    x = self.fc(output)
    state = [state_h,state_c]
    state = tf.squeeze(state)

    return x,state_h, attention_weights


#max_length_inp = max_length_inp
#max_length_out = max_length_out
class Model(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, encoder_units, BATCH_SIZE)
        self.decoder = Decoder(vocab_size, embedding_dim, decoder_units, BATCH_SIZE)              
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')          
        self.checkpoint_dir = './training_checkpoints'  
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer = self.optimizer,
                                 embedding = embedding,
                                 encoder = self.encoder,
                                 decoder = self.decoder) 
        self.checkpointmanager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep = 3)
        
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)   

        mask = tf.cast(mask, dtype = loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)   

    def train_step(self, inp, targ, enc_state):
        loss = 0

        with tf.GradientTape() as tape:    
            enc_output, enc_hidden, enc_state = self.encoder(inp, enc_state)         
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)  
            
            # Teacher forcing - feeding the target as the next input
            for t in range (1, targ.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                loss += self.loss_function(targ[:,t],predictions)

                dec_input = tf.expand_dims(targ[:,t],1)

            batch_loss = loss/targ.shape[1]

            variables  = self.encoder.trainable_variables + self.decoder.trainable_variables + embedding.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(list(zip(gradients, variables)))
 
        return batch_loss
    
    def train(self):
        wandb.init(project = "seq2seq model")

        for epoch in range(EPOCHS):
            start = time.time()
            
            enc_state = self.encoder.initialize_state()
            total_loss = 0

            for (batch, (inp, targ)) in list(dataset.enumerate()):
                batch_loss = self.train_step(inp, targ, enc_state)
                total_loss += batch_loss

                if batch%75 == 0:       
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
                    
                log  = {"epoch":epoch,
                        "batch loss":batch_loss,
                        "total loss":total_loss}
                wandb.log(log)  
            if (epoch + 1) % 1 == 0:
                self.checkpointmanager.save()   
      
    
            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
 
                
    def save_embeddings(self):
        weights = embedding.get_weights()[0]
        
        with open(os.path.join(log_dir,'metadata.tsv'),'w') as f:
            for index in vocab:
                if index <= num_words and index != 0:
                    f.write('{}\n'.format(vocab[index]))
                
        with open(os.path.join(log_dir,'vectors.tsv'),'w') as f:
            for index in vocab:
                if index <= num_words and index != 0:
                    vec = weights[index]
                    f.write('{}\n'.format('\t'.join([str(x) for x in vec])))
        
        
    def tokenize_sentence(self,sentence):
        sequence  = []
        for word in sentence.split():
            try:
                if tokenizer.word_index[word]<=num_words:
                    sequence.append(tokenizer.word_index[word])
                else:
                    sequence.append(1)
            except:
                sequence.append(1)

        return sequence
    
    def evaluate(self,sentence):
        
        sentence = preprocess_sequence(sentence)
        inputs  = self.tokenize_sentence(sentence)
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
        inputs = tf.convert_to_tensor(inputs)
        result = ''

        enc_state = [tf.zeros((1,encoder_units )) for i in range(4)]

        enc_out,  enc_hidden, enc_state = self.encoder(inputs, enc_state)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']],0)     
        for t in range(max_length_out):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)
            prediction_id = tf.argmax(predictions[0]).numpy()

            result += tokenizer.index_word[prediction_id] + ' '

            if tokenizer.index_word[prediction_id] == '<end>' :
                return result, sentence

            dec_input = tf.expand_dims([prediction_id],0)
            
        return result, sentence           
                
    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
