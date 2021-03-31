#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 10:43:16 2021

@author: harsh
"""

import numpy as np
import os

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

###################################################################################################################################################################
 
qna_lines = open(os.path.join('data','dialogs.txt'), encoding='utf-8', errors='ignore').read().split('\n')

qna_que = []
qna_ans = []

for line in qna_lines:
    line = line.split('\t')
    qna_que.append(line[0])
    qna_ans.append(line[1])    

qna_que = preprocess_data(qna_que)
qna_ans = preprocess_data(qna_ans)

# sorting out those sequences whose length is less than that of specified
index = []      
for i in range(len(qna_que)):
    if len(qna_que[i].split())<=12 and len(qna_ans[i].split())<=12:
        index.append(i)
        
qna_questions = []
qna_responses = []

for i in index:
    qna_questions.append(qna_que[i])
    qna_responses.append(qna_ans[i])
    
len(qna_questions)
###################################################################################################################################################################


lines = open(os.path.join('data','movie_lines.txt'), encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open(os.path.join('data','movie_conversations.txt'), encoding='utf-8', errors='ignore').read().split('\n')

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        

convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))
    
cornell_que = []
cornell_ans = []

for conv in convs:
    for i in range(len(conv)-1):
        cornell_que.append(id2line[conv[i]])
        cornell_ans.append(id2line[conv[i+1]])
        
        
    
cornell_que = preprocess_data(cornell_que)
cornell_ans = preprocess_data(cornell_ans)

# sorting out those sequences whose length is less than that of specified
index = []      

for i in range(len(cornell_que)):
    if len(cornell_que[i].split())<=12 and len(cornell_ans[i].split())<=12:
        index.append(i)
        
cornell_questions = []
cornell_responses = []

for i in index:
    cornell_questions.append(cornell_que[i])
    cornell_responses.append(cornell_ans[i])


###################################################################################################################################################################
# merging the datasets

questions = qna_questions + cornell_questions
responses = qna_responses + cornell_responses
    
with open(os.path.join('data','questions.txt'), 'w') as output:
    for row in questions:
        output.write(str(row) + '\n')
        
        
with open(os.path.join('data','responses.txt'), 'w') as output:
    for row in responses:
        output.write(str(row) + '\n')
