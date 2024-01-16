# NeuralNLP

Pytorch implementation of CMU Neural NLP class 11-747 

Reference: https://github.com/neubig/nn4nlp-code

### Language Model (feedforward and LSTM)

Treebank data: data/ptb/train.txt, valid.txt, and test.txt

02-lm-pytorch.py
- FNN_LM: embedding -> FC -> Tanh -> FC
- train_model() train on sentence one-by-one

04-lm-lstm.py
- LM_LSTM: embedding -> LSTM -> FC
- ptb_iterator() geenrate minibatch
- run_epoch() train and evaluate model
- Test perplexity: 125.90


### Sentiment Classification (CNN and LSTM)

Sentiment classification data: data/classes/train.txt, dev.txt, and test.txt, with 0-4 as sentiment labels

03-cnn-class.py
- CNN model: embedding -> conv1d -> max_pooling -> ReLU -> FC as final layer
- CNNClassification build dataset and train
- SentimentDataset read data and map to index; TorchDataset handles batching and padding

