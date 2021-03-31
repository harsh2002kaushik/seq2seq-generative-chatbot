# seq2seq-generative-chatbot
## Description
A deep learning chatbot implimented usng seq2seq attention based model usinf bilstm layers and was trained on combination of dataset like Cornell-Movie-Dialogs-Corpus for sequences having length less than 10 words.


## Prerequisties
 * unzipping the cornell-movie-dialogs-corpus
 * Python(>3.5)
 * Tensorflow(2.4.1)
 * sklearn
 * wandb
 * dash 

## Running 
### to run the trained model on local host:
 * install the requirements : `pip install -r requirements.txt`
 * run : `python3 dashapp.py`

### to train the model from python console:
 * install the requirements : `pip install -r requirements.txt`
 * make a wandb account 
 * run : `python3 train.py`
