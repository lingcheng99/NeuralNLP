import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import time


class LM_LSTM(nn.Module):
    def __init__(self, num_steps, vocab_size, batch_size, embedding_dim, hidden_dim, num_layers, drop_out):
        super(LM_LSTM, self).__init__()
        self.num_steps = num_steps  # number of inputs
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.drop_out = drop_out
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout_layer = nn.Dropout(drop_out)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=drop_out)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0.0)
        self.fc.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, self.batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, inputs, hidden):
        # embeds size (num_steps, batch_size, embedding_dim)
        embeds = self.dropout_layer(self.embedding(inputs))
        # lstm_out size (num_steps, batch_size, hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = self.dropout_layer(lstm_out)
        # logits size (num_steps * batch_size, vocab_size)
        logits = self.fc(lstm_out.view(-1, self.hidden_dim))
        # reshape logits to (num_steps, batch_size, vocab_size)
        return logits.view(self.num_steps, self.batch_size, self.vocab_size), hidden


def read_ptb_data():
    """"read data/ptb/*.txt files, map words to index, return List[int]
    """
    # w2i map word to index
    w2i = defaultdict(lambda: len(w2i))
    # S is end of sentence symbol,  w2i['<s>']=0
    S = w2i["<s>"]
    # unknown token, w2i['<unk>']=1
    UNK = w2i["<unk>"]

    def read_dataset(filename):
        with open(filename, 'r') as f:
            # read entire file as one long string, not line by line
            words = f.read().replace("\n", "<eos>").split()
            for w in words:
                yield w2i[w]

    # train_data is nested list, each list map words to index in w2i
    train_data = list(read_dataset("../data/ptb/train.txt"))
    # any new words (non-existent key in w2i) will be mapped to UNK
    w2i = defaultdict(lambda: UNK, w2i)
    test_data = list(read_dataset("../data/ptb/test.txt"))
    i2w = {v: k for k, v in w2i.items()}
    vocab_size = len(w2i)
    return train_data, test_data, w2i, i2w, vocab_size


def ptb_iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw PTB data, generate mini-batch
    @param raw_data: List[int], train_data or dev_data from read_ptb_data
    @param batch_size: int, batch_size
    @param num_steps: int, the number of unrolls
    Return pairs of batched data, each of shape (batch_size, num_steps)
    The second element of the tuple is the same data time-shifted to the right by one
    """
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_num = data_len // batch_size
    # batch_size number of rows, batch_num number of cols
    data = np.zeros([batch_size, batch_num], dtype=np.int32)
    for i in range(batch_size):
        # i-th row in data has #batch_num samples, continuously from raw_data
        data[i] = raw_data[batch_num * i: batch_num * (i+1)]

    epoch_size = (batch_num - 1) // num_steps
    for i in range(epoch_size):
        # x is from all row, num_steps col, so x has #atch_size rows, and it is continuous in each row
        x = data[:, i * num_steps: (i+1) * num_steps]
        # y is x time-shifted to the right by one
        y = data[:, i * num_steps + 1: (i+1) * num_steps + 1]
        yield (x, y)


def run_epoch(model, criterion, data, is_train=False, lr=1.0):
    if is_train:
        model.train()
    else:
        model.eval_pred()
    # compute epoch_size using model parameters
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    hidden = model.init_hidden()
    costs, iters = 0.0, 0
    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.num_steps)):
        # need to be LongTensor, size (num_steps, batch_size) for both inputs and targets
        inputs = torch.from_numpy(x).type(torch.LongTensor).transpose(0, 1).contiguous()
        targets = torch.from_numpy(y).type(torch.LongTensor).transpose(0, 1).contiguous()
        model.zero_grad()
        hidden[0].detach_()  # hidden is tuple from LSTM (hidden, cell), detach individually
        hidden[1].detach_()

        outputs, hidden = model(inputs, hidden)
        tt = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))
        loss = criterion(outputs.view(-1, model.vocab_size), tt)
        costs += loss.item() * model.num_steps
        iters += model.num_steps

        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            for p in model.parameters():
                p.data.add_(p.grad.data, alpha=-lr)
            if step % (epoch_size // 10) == 10:
                print("{} perplexity: {:8.2f}".format(step * 1.0 / epoch_size, np.exp(costs / iters)))
                print("time_elapsed: ", (time.time() - start_time))
    return np.exp(costs / iters)


if __name__ == '__main__':
    train_data, test_data, w2i, i2w, vocab_size = read_ptb_data()
    print("vocab size: ", vocab_size, len(train_data), len(test_data))

    num_steps = 35
    embedding_dim = 100
    hidden_dim = 256
    num_layers = 2
    batch_size = 64
    num_epochs = 10
    drop_out = 0.5
    model = LM_LSTM(num_steps, vocab_size, batch_size, embedding_dim, hidden_dim, num_layers, drop_out)
    criterion = nn.CrossEntropyLoss()

    # adjust learning_rate
    lr = 20.0
    lr_decay_base = 1 / 1.15
    m_flat_lr = 14.0

    for epoch in range(num_epochs):
        lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
        lr = lr * lr_decay
        train_p = run_epoch(model, criterion, train_data, True, lr)
        print('Train perplexity at epoch {}: {:8.2f}'.format(epoch, train_p))
    # run all test data at once
    model.batch_size = 1
    # test_data with is_train=False and lr=1.0
    test_p = run_epoch(model, criterion, test_data)
    print('Test Perplexity: {:8.2f}'.format(test_p))
