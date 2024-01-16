# NeuralNLP

Pytorch implementation of CMU Neural NLP class 11-747 

Reference: https://github.com/neubig/nn4nlp-code

**Lecture 2. Language Modeling**

02-lm-pytorch.py
- feed-forward language model to predict next word
- Treebank data: data/ptb/train.txt, valid.txt, and test.txt

**Lecture 3. Sentiment classification with CNN**

03-cnn-class.py
- CNN model: embedding -> conv1d -> max_pooling -> ReLU -> FC as final layer
- CNNClassification build dataset and train; SentimentDataset read data and map to index; TorchDataset handles batching
- Sentiment classification data: data/classes/train.txt, dev.txt, and test.txt, with 0-4 as sentiment labels
