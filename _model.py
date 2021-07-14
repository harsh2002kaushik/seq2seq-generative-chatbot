#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 04:15:52 2021

@author: harsh
"""

import tensorflow as tf
import time
import os
#from model import Encoder, Decoder,embedding, max_length_inp, max_length_out
from preprocessor import preprocess_sequence
import pickle
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
