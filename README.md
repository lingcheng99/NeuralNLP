# NeuralNLP

This repo contains my understanding and implementation of famous NLP algorithms from scratch, using PyTorch  

Inspired by [CMU NeuralNLP class](https://github.com/neubig/nn4nlp-code) and [Ben Trevett's Seq2Seq](https://github.com/bentrevett/pytorch-seq2seq/)

Most files are written as a standalone python file, including data preprocessing, model architecture, model training and evaluation. 

All use publicly available data. Performance was reported when available.

Feedbacks are welcome!



### Sentiment Classification (CNN and LSTM)

03-cnn-class.py on classes dataset
- CNN model: embedding -> conv1d -> max_pooling -> ReLU -> FC as final layer
- CNNClassification build dataset and train
- SentimentDataset read data and map to index; TorchDataset handles batching and padding
- Sentiment classification data: data/classes/train.txt, dev.txt, and test.txt, with 0-4 as sentiment labels
- test acc=0.3710

05-lstm-sentiment.py on IMDB dataset
- RNN1: one-directional GRU -> FC
- RNN2: bidirectional GRU -> FC
- IMDBDataset read and preprocess data, with Glove embeddings
- get_batch() generate minibatch
- train_rnn() train ane evalaute model
- test acc=0.84

### Word2vec (skipgram and cbow)

06-wordemb-skipgram.py
- data from http://mattmahoney.net/dc/text8.zip
- Text8Dataset read data and performs subsampling and negative sampling
- Skipgram take u_pos (positive center), v_pos (positive context), and v_neg (negative context) to compute loss
- Word2Vec train and evaluate model

07-wordemb-cbow.py
- data from data/ptb/train.txt, valid.txt, and test.txt
- WordEmbCbow: embedding -> FC
- CBowDataset read and preprocess data
- train_model() train model and save embeddings

### Seq2seq (LSTM and attention)

08-seq2seq-LSTM.py
- Seq2seq with encoder and decoder; both encoder and decoder use LSTM
- Multi30k dataset from torchtext
- After 10 epoch, Test Loss: 2.995 | Test PPL:  19.979

09-seq2seq-attention.py
- Seq2seq with encoder and decoder, implemented as the transformer paper
- Multi30k dataset from torchtext
- After 10 epoch, Test Loss: 1.689 | Test PPL:   5.416


### Basic Language Model (feedforward and LSTM)

02-lm-pytorch.py
- FNN_LM: embedding -> FC -> Tanh -> FC
- train_model() train on sentence one-by-one
- data: Penn Treebank data, data/ptb/train.txt, valid.txt, and test.txt

04-lm-lstm.py
- LM_LSTM: embedding -> LSTM -> FC
- ptb_iterator() geenrate minibatch from Penn Treebank data
- run_epoch() train and evaluate model
- Test perplexity: 125.90
