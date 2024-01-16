from collections import defaultdict
import math
import time
import random
import torch
import torch.nn as nn


CONTEXT_WINDOW = 2
EMBEDDING_DIM = 128
MAX_LEN = 100
BATCH_SIZE = 64
NUM_EPOCH = 100


class WordEmbCbow(nn.Module):
    """from context_window, predict target word
    """
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbCbow, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        # from (batch_size, inputs_length, emb_dim), sum over inputs_length to give (batch_size, emb_dim)
        embeds = torch.sum(self.embedding(inputs), dim=1)
        # from (batch_size, emb_dim) to (batch_size, vocab_size)
        out = self.linear(embeds)
        # log_softmax return same size, just softmax over dim=1, then log, return (batch_size, vocab_size)
        return nn.functional.log_softmax(out, dim=1)


class CBowDataset(torch.utils.data.Dataset):
    """read in data from filepath, and return an instance of dataset
    for each sentence, map word to index, create context_vec and target_vec from each padded sentence
    For torch Dataset, with __len__ and __getitem__, return a pair of tensors as X and y
    """
    def __init__(self, filepath):
        """read in file, process into context_vec and target_vec"""
        super(CBowDataset, self).__init__()

        # w2i dictionary to map word to index
        self.w2i = defaultdict(lambda: len(self.w2i))
        # S is end of sentence symbol,  w2i['<s>']=0
        S = self.w2i["<s>"]
        # unknown token, w2i['<unk>']=1
        UNK = self.w2i["<unk>"]

        # read in sentences and map word to index
        self.sentences = list()
        with open(filepath, 'r') as f:
            for line in f:
                self.sentences.append([self.w2i[x] for x in line.strip().split(" ")])

        self.iw2 = {v: k for k, v in self.w2i.items()}
        self.vocab_size = len(self.w2i)
        self.window_size = CONTEXT_WINDOW
        self.context_target = list()  # store tuple(context_vec, target_vec)

        for sent in self.sentences:
            padded_sent = [S] * CONTEXT_WINDOW + sent + [S] * CONTEXT_WINDOW
            for i in range(CONTEXT_WINDOW, len(sent) - CONTEXT_WINDOW):
                context_vec = torch.tensor(padded_sent[i - CONTEXT_WINDOW: i]
                                           + padded_sent[i + 1: i + CONTEXT_WINDOW + 1], dtype=torch.long)
                target_vec = torch.tensor([padded_sent[i]], dtype=torch.long)
                self.context_target.append((context_vec, target_vec))

    def __len__(self):
        return len(self.context_target)

    def __getitem__(self, idx):
        context, target = self.context_target[idx]
        return context, target


def train_model():
    train_filepath = "../data/ptb/train.txt"
    train_dataset = CBowDataset(train_filepath)
    vocab_size = train_dataset.vocab_size

    model = WordEmbCbow(vocab_size, EMBEDDING_DIM)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.NLLLoss()

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    for epoch in range(NUM_EPOCH):
        print("started epoch %r" % epoch)
        train_loss = 0.0
        start = time.time()
        # enumerate(dataloader) to get index, and (batch_X, batch_y)
        for idx, (batch_X, batch_y) in enumerate(train_dataloader):
            model.zero_grad()
            # after log_softmax, size (batch_size, vocab_size)
            log_probs = model(batch_X)
            # need batch_y.squeeze(), from (batch_size, 1) to (batch_size)
            loss = loss_func(log_probs, batch_y.squeeze())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (idx + 1) % 10000 == 0:
                print("--finished %r batches in time %.2fs" % (idx+1, time.time()-start))
        print("epoch %r: train loss=%.4f, time=%.2fs" %
              (epoch, train_loss, time.time()-start))

    embedding_location = "./data/wordemb/embeddings_trained.txt"

    print("saving embedding files")
    with open(embedding_location, 'w') as embeddings_file:
        W_w_np = model.embedding.weight.data.cpu().numpy()
        for i in range(vocab_size):
            ith_embedding = '\t'.join(map(str, W_w_np[i]))
            embeddings_file.write(ith_embedding + '\n')


if __name__ == '__main__':
    train_model()

