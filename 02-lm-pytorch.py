from collections import defaultdict
import math
import time
import random
import torch
import torch.nn as nn
from torch.autograd import Variable


N = 2  # the length of ngram
EMB_SIZE = 128  # the size of the embedding
HID_SIZE = 128  # the size of the hidden layer
USE_CUDA = torch.cuda.is_available()
MAX_LEN = 100  # maximal length for generated sentence


class FNN_LM(nn.Module):
    """feed-forward neural language model
    embedding -> linear -> tanh -> linear
    """
    def __init__(self, nwords, emb_size, hid_size, num_hist, dropout):
        super(FNN_LM, self).__init__()
        self.embedding = nn.Embedding(nwords, emb_size)
        self.fnn = nn.Sequential(
            nn.Linear(num_hist * emb_size, hid_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_size, nwords)
        )

    def forward(self, words):
        # 3D tensor of shape (batch_size, num_hist, emb_size)
        emb = self.embedding(words)
        # 2D tensor of shape (batch_size, num_hist * emb_size)
        feat = emb.view(emb.size(0), -1)
        # 2D tensor of shape (batch_size, nwords)
        logit = self.fnn(feat)
        return logit


def read_data():
    # w2i map word to index
    w2i = defaultdict(lambda: len(w2i))
    # S is end of sentence symbol,  w2i['<s>']=0
    S = w2i["<s>"]
    # unknown token, w2i['<unk>']=1
    UNK = w2i["<unk>"]

    def read_dataset(filename):
        with open(filename, 'r') as f:
            for line in f:
                yield [w2i[x] for x in line.strip().split(" ")]

    # train_data is nested list, each list map words to index in w2i
    train_data = list(read_dataset("../data/ptb/train.txt"))
    # any new words (non-existent key in w2i) will be mapped to UNK
    w2i = defaultdict(lambda: UNK, w2i)
    dev_data = list(read_dataset("../data/ptb/valid.txt"))
    i2w = {v: k for k, v in w2i.items()}
    nwords = len(w2i)
    return train_data, dev_data, w2i, i2w, nwords


def convert_to_variable(words):
    """convert a list of int into a PyTorch variable (earlier version PyTorch requires this)
    Input words: list(int)
    Return torch.autograd.Variable instance for gradient computation"""
    var = Variable(torch.LongTensor(words))
    if USE_CUDA:
        var = var.cuda()
    return var


def calc_score_of_histories(words, model):
    """calculate scores for one value
    :param words: list[list[int]], mini-batch of input, each item is a window
    :param model: nn.module class
    return prediction, tensor of size (batch_size, vocab_size)
    """
    words_var = convert_to_variable(words)
    logits = model(words_var)
    return logits


def calc_sent_loss(sent, model, w2i):
    """calculate loss value for the entire sentence"""
    # initial history is equal to end of sentence symbol
    S = w2i["<s>"]
    hist = [S] * N  # window size N
    all_histories = list()  # nested list, each item is current window
    all_targets = list()  # next word after current window

    # next_word is first word in sent
    for next_word in sent + [S]:
        # all_histories is nested list, each list is current window
        all_histories.append(list(hist))
        # all_targets grow word by word
        all_targets.append(next_word)
        # move hist by one word
        hist = hist[1:] + [next_word]
    # use all_histories to get logits, one for each item in all_histories
    logits = calc_score_of_histories(all_histories, model)
    # compare logits to all_targets to get loss
    loss = nn.functional.cross_entropy(logits, convert_to_variable(all_targets), size_average=False)
    return loss


def train_model(train_data, dev_data, w2i, i2w, nwords):
    model = FNN_LM(nwords=nwords, emb_size=EMB_SIZE, hid_size=HID_SIZE, num_hist=N, dropout=0.2)
    if USE_CUDA:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_iter = 5  # number of iterations
    last_dev = 1e20
    best_dev = 1e20

    for ITER in range(num_iter):
        random.shuffle(train_data)
        # set model in training mode
        model.train()
        # for each iteration, reset train_loss and train_words
        train_words, train_loss = 0, 0.0
        start = time.time()
        for sent_id, sent in enumerate(train_data):
            my_loss = calc_sent_loss(sent, model, w2i)
            train_loss += my_loss.item()
            train_words += len(sent)
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()

            if (sent_id + 1) % 5000 == 0:
                print("--finished %r sentences (word/sec=%.2f)" %
                      (sent_id+1, train_words/(time.time()-start)))
        print("iter %r: train loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" %
              (ITER, train_loss / train_words, math.exp(train_loss / train_words), train_words / (time.time() - start)))

        # evaluate on dev set
        model.eval()  # set model in evaluation model
        dev_words, dev_loss = 0, 0.0
        start = time.time()
        for sent_id, sent in enumerate(dev_data):
            my_loss = calc_sent_loss(sent, model, w2i)
            dev_loss += my_loss.item()
            dev_words += len(sent)

        # track best_dev, save model only if it is the best
        if best_dev > dev_loss:
            torch.save(model, "02-lm-model.pt")
            best_dev = dev_loss

        # Save the model
        print("iter %r: dev loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" %
              (ITER, dev_loss / dev_words, math.exp(dev_loss / dev_words), dev_words / (time.time() - start)))

    return model


def eval_model(i2w):
    """load saved model and generate sentence"""
    model = torch.load("02-lm-model.pt")
    model.eval_pred()
    S = 0
    hist = [S] * N
    sent = []
    while True:
        logits = calc_score_of_histories([hist], model)
        prob = nn.functional.softmax(logits, dim=0)
        # use item() to get single number
        next_word = torch.multinomial(prob, 1).item()
        print(prob.size(), next_word)
        if next_word == S or len(sent) == MAX_LEN:
            break
        sent.append(next_word)
        hist = hist[1:] + [next_word]
    print(" ".join([i2w[x] for x in sent]))
    return sent


if __name__ == '__main__':
    train_data, dev_data, w2i, i2w, nwords = read_data()
    fnn_model = train_model(train_data, dev_data, w2i, i2w, nwords)
    eval_model(i2w)







