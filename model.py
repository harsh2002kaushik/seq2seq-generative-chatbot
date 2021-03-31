#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 10:44:02 2021

@author: harsh
"""


import tensorflow as tf
import time
import sklearn
import os
import sklearn.model_selection
import numpy as np
from preprocessor import *

que = questions
ans = responses


data = que+ans

num_words = 23500

tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', num_words = num_words, oov_token='OOV')
tokenizer.fit_on_texts(data)

input_tensor = tokenizer.texts_to_sequences(ans)
input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, padding='post',maxlen = 12)

output_tensor = tokenizer.texts_to_sequences(que)
output_tensor = tf.keras.preprocessing.sequence.pad_sequences(output_tensor, padding='post', maxlen =12)

max_length_inp = input_tensor.shape[1]
max_length_out = output_tensor.shape[1]

input_tensor_train, input_tensor_val, output_tensor_train, output_tensor_val = sklearn.model_selection.train_test_split(input_tensor, output_tensor, test_size=0.01)

vocab = tokenizer.index_word
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 128        
encoder_units = 512
decoder_units = 1024    
EPOCHS = 100
vocab_size  = num_words + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, output_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


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

