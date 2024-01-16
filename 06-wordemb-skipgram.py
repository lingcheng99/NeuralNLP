import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import math
import random
import time
import csv
from scipy.stats import spearmanr


class Text8Dataset(torch.utils.data.Dataset):
    """read in text8.txt from filepath, subsample and create dataset
    """
    def __init__(self, filepath, vocab_size, window_size, neg_sample_num):
        self.vocab_size = vocab_size
        self.process = False  # generate_batch(), True to start generating batch
        self.data_index = 0  # generate_batch(), reset for each epoch

        self.words = self.read_data(filepath)

        self.words_idx, self.vocab_counter, self.w2i, self.i2w = self.build_dataset()

        self.train_data = self.subsampling()

        self.neg_sample_table = self.init_neg_sample_table()

        self.dataset = self.create_dataset(window_size, neg_sample_num)

    def read_data(self, filepath):
        with open(filepath) as f:
            # this file has just one line, no need for `for line in f`
            data = [x for x in f.read().split() if x != 'eoood']
        return data

    def build_dataset(self):
        """from self.words (original list of tokens), keep top-n words, n=self.vocab_size-1
        :return words_idx: List[Int], map self.words in top-n words to their index, unknown words to 0
        :return vocab_counter: List[(String, Int),...], each tuple is word and its count, for top-n words
        :return w2i: dict{String: Int}, map top-n words to index
        :return i2w: dict{Int: String}, map index to top-n words
        """
        # UNK for unknown words, initialize count of UNK at -1
        vocab_counter = [['UNK', -1]]
        # add top-n words by counter, n=self.vocab_size-1
        vocab_counter.extend(Counter(self.words).most_common(self.vocab_size - 1))

        # w2i map top-n words to index
        w2i = dict()
        for w, _ in vocab_counter:
            w2i[w] = len(w2i)
        # i2w map index to top-n words
        i2w = {v: k for k, v in w2i.items()}

        # words_idx: List[Int], is to map each word in self.words to index in top-n, other words map to 0
        words_idx = list()
        unk_cnt = 0
        for w in self.words:
            if w in w2i:
                words_idx.append(w2i[w])
            else:
                words_idx.append(0)
                unk_cnt += 1
        vocab_counter[0][1] = unk_cnt
        return words_idx, vocab_counter, w2i, i2w

    def subsampling(self):
        """subsample original self.words_idx to balance rare words and frequent words
        each word wi in the training set is kept with probability, f(wi) is fraction of total words in corpus for wi
        (sqrt(f(wi)/0.001) + 1) * 0.001/f(wi)
        :return List[Int], from self.words_idx, subsample and return a shortened list with same pools of index
        """
        # get count of top-n words
        counts = [tup[1] for tup in self.vocab_counter]
        # normalize frequency
        freq = np.array(counts) / np.sum(counts)

        prob = dict()
        for idx, x in enumerate(freq):
            y = (math.sqrt(x/0.001) + 1) * 0.001 / x
            prob[idx] = y
        subsampled_data = list()
        for w in self.words_idx:
            if random.random() < prob[w]:
                subsampled_data.append(w)
        return subsampled_data

    def init_neg_sample_table(self):
        """create self.neg_sample_table as template for negative sampling
        the author suggests the best way is to raise the word count to 3/4 power
        P(wi) = f(wi) ** (3/4) / sum-of-f(wj) ** (3/4)
        :return List[Int], each Int is index for top-n words in self.i2w
        """
        counts = [tup[1] for tup in self.vocab_counter]
        prob_freq = np.array(counts) ** 0.75
        prob_sum = np.sum(prob_freq)
        # after normalization, this is the probability of keeping word wi
        prob_ratio = prob_freq / prob_sum
        table_size = 1e8
        # prob_int convert probability to integer
        prob_table = np.round(prob_ratio * table_size)

        sample_table = list()
        for idx, x in enumerate(prob_table):
            # occurrence of idx in sample_table is proportional to its probability in prob_ratio
            sample_table += [idx] * int(x)

        return np.array(sample_table)

    def create_dataset(self, window_size, neg_sample_num):
        """create list of pos_u, pos_v, and neg_v"""
        span = 2 * window_size + 1
        dataset = list()
        for i in range(len(self.train_data) - span):
            pos_u = torch.tensor([self.train_data[i + window_size]] * 2 * window_size, dtype=torch.long)
            pos_v = torch.tensor(self.train_data[i: i + window_size] + self.train_data[i + window_size + 1: i + span]
                                 , dtype=torch.long)
            neg_v = torch.tensor(np.random.choice(self.neg_sample_table, size=(2 * window_size, neg_sample_num))
                                 , dtype=torch.long)
            dataset.append((pos_u, pos_v, neg_v))

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        # u is the center vector, and v is the context vector
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.init_emb()

    def init_emb(self):
        init_range = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-init_range, init_range)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, u_pos, v_pos, v_neg, batch_size):
        # first compute score from positive center u_pos, and positive context v_pos
        # with dataset, after embedding size (batch_size, 2*window_size, embed_dim), need to sum for dim=1
        embed_u = self.u_embeddings(u_pos.view(-1))
        embed_v = self.v_embeddings(v_pos.view(-1))
        # torch.mul is element-wise multiplication, same size as embed_u or embed_v (batch_size, embedding_dim)
        pos_score = torch.mul(embed_u, embed_v)
        # dim=0 is batch_size, dim=1 is embedding_dim, size=(batch_size)
        pos_score = torch.sum(pos_score, dim=1)
        # take logsigmoid and squeeze() to keep size=(batch_size)
        log_target = F.logsigmoid(pos_score).squeeze()

        # second compute score from positive center u_pos, and negative context v_neg
        # v_neg size=(batch_size, 2*window_size, negative_sample_num)
        # need sum(dim=1) to output size (batch_size, negative_sample_num, embedding_dim)
        neg_embed_v = self.v_embeddings(torch.flatten(v_neg, start_dim=0, end_dim=1))
        # torch.bmm performs batch matrix_matrix product; embed_u.unsqueeze(2) add dimension at dim=2
        # output (batch_size, negative_num, 1), squeeze() to remove 1 to (batch_size, embedding_dim)
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        # sum over dim=1 for embedding_dim, size=(batch_size)
        neg_score = torch.sum(neg_score, dim=1)
        # take logsigmoid, squeeze()  to keep size=(batch_size)
        sum_log_sampled = F.logsigmoid(neg_score).squeeze()

        loss = log_target + sum_log_sampled
        return -1 * loss.sum() / batch_size

    def input_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()


class Word2Vec:
    def __init__(self, filepath, vocab_size=100000, embedding_dim=100, epoch_num=10, batch_size=64,
                 window_size=5, neg_sample_num=10):
        self.dataset = Text8Dataset(filepath, vocab_size, window_size, neg_sample_num)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.window_size = window_size
        self.neg_sample_num = neg_sample_num

    def train(self):
        model = SkipGram(self.vocab_size, self.embedding_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        dataLoader = torch.utils.data.DataLoader(self.dataset.dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epoch_num):
            start = time.time()
            print("epoch %d starts" % epoch)
            # each epoch reset batch_num, and train_loss
            train_loss = 0.0

            for idx, (batch_pos_u, batch_pos_v, batch_neg_v) in enumerate(dataLoader):
                if idx == 0:
                    print("First batch: ", batch_pos_u.size(), batch_pos_v.size(), batch_neg_v.size())
                optimizer.zero_grad()
                loss = model(batch_pos_u, batch_pos_v, batch_neg_v, self.batch_size)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                if idx > 0 and idx % 10000 == 0:
                    end = time.time()
                    word_embeddings = model.input_embeddings()
                    sp1, sp2 = scoring_func(self.dataset.w2i, word_embeddings)
                    print('batch=%d sp=%1.3f %1.3f  time_elapsed = %4.2f loss = %4.3f\r'
                          % (idx, sp1, sp2, (end - start), loss.item()))


def scoring_func(w2i, embed):
    """given w2i, embed (embedding vectors), compare score in files combined.csv and rw.txt"""

    # use combined.csv to get one score
    with open("../data/wordemb/combined.csv") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip header row
        human_sim = list()
        cosine_sim = list()
        for row in csv_reader:
            # ensure both words in w2i
            if (row[0] not in w2i) or (row[1] not in w2i):
                continue
            word1 = int(w2i[row[0]])  # get index of first word
            word2 = int(w2i[row[1]])
            human_sim.append(float(row[2]))

            value1 = embed[word1]
            value2 = embed[word2]
            score = np.dot(value1, value2) / np.linalg.norm(value1) * np.linalg.norm(value2)
            cosine_sim.append(score)

    cor1, _ = spearmanr(human_sim, cosine_sim)

    with open("../data/wordemb/rw.txt") as f:
        next(f)
        human_sim = list()
        cosine_sim = list()
        for line in f:
            lst = line.strip().split()
            if (lst[0] not in w2i) or (lst[1] not in w2i):
                continue
            word1 = int(w2i[lst[0]])  # get index of first word
            word2 = int(w2i[lst[1]])
            human_sim.append(float(lst[2]))

            value1 = embed[word1]
            value2 = embed[word2]
            score = np.dot(value1, value2) / np.linalg.norm(value1) * np.linalg.norm(value2)
            cosine_sim.append(score)
    cor2, _ = spearmanr(human_sim, cosine_sim)
    return cor1, cor2


if __name__ == '__main__':
    filepath = ""
    wc = Word2Vec(filepath)
    wc.train()

