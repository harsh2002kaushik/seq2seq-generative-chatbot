#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 10:44:30 2021

@author: harsh
"""

import tensorflow as tf
import time
import os
from model import *
import wandb

log_dir = 'training_checkpoints/'     

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

        
if __name__ == "__main__":
    model = Model()
    model.restore()
    #model.train()
    
def save_weight():
    model.save_weights('model_weights')
    
#model.restore()
#embedding.get_weights()
#model.evaluate('i know')
#model.save_weights('model_weights')
