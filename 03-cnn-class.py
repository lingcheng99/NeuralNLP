from collections import defaultdict
import time
import random
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, window_size, num_tags):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        self.conv_1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=window_size,
                                 stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu = nn.ReLU()
        self.linear_layer = nn.Linear(in_features=num_filters, out_features=num_tags, bias=True)
        nn.init.xavier_uniform_(self.linear_layer.weight)

    def forward(self, inputs):
        # inputs (batch_size, padded_length), embedding to (batch_size, padded_length, embedding_dim)
        emb = self.embedding(inputs)
        # from (batch_size, padded_length, embedding_dim) to (batch_size, embedding_dim, padded_length)
        emb = emb.permute(0, 2, 1)
        # size (batch_size, num_filters, padded_length)
        conv = self.conv_1d(emb)
        # max pooling from (batch_size, num_filters, padded_length) to (batch_size, num_filters)
        conv = conv.max(dim=2)[0]
        conv = self.relu(conv)
        # from (batch_size, num_filters) to (batch_size, num_classes)
        out = self.linear_layer(conv)
        return out


class SentimentDataset:
    """"read in train_file and dev_file, each List[(words, tag)], one tuple per sentence
    train_data and dev_data share w2i and t2i
    """
    def __init__(self, train_file, dev_file):
        self.train_data, self.dev_data, self.w2i, self.t2i = self.build_dataset(train_file, dev_file)
        self.vocab_size = len(self.w2i)
        self.num_tags = len(self.t2i)

    def build_dataset(self, train_file, dev_file):
        w2i = defaultdict(lambda: len(w2i))
        t2i = defaultdict(lambda: len(t2i))
        UNK = w2i["<unk>"]

        def read_dataset(filename):
            with open(filename, "r") as f:
                for line in f:
                    # each line start with tag ||| sentences
                    tag, words = line.lower().strip().split(" ||| ")
                    yield [w2i[x] for x in words.split(" ")], t2i[tag]

        # tuple(List[int], int), first item is token mapped to index, second is tag mapped to index
        train_data = list(read_dataset(train_file))
        w2i = defaultdict(lambda: UNK, w2i)
        dev_data = list(read_dataset(dev_file))
        return train_data, dev_data, w2i, t2i


class TorchDataset(torch.utils.data.Dataset):
    """"from train_data or dev_data, implement dataset with __len__ and __getitem__
    train_data/dev_data: List[Tuple(List[int], int)], first is list of tokens mapped to index, second is tag to index
    Need padding: torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    """
    def __init__(self, data, window_size):
        self.dataset = self.create_dataset(data, window_size)

    @staticmethod
    def create_dataset(data, window_size):
        dataset = list()
        for words, tag in data:
            # if less than window_size, add padding to window_size
            if len(words) < window_size:
                words += [0] * (window_size - len(words))
            words_tensor = torch.tensor(words, dtype=torch.long)
            tag_tensor = torch.tensor(tag, dtype=torch.long)
            dataset.append((words_tensor, tag_tensor))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def collate_fn(batch):
        # transpose batch, List[Tuple] to List[List[Tuple[0]], List[Tuple[1]]]
        batch_elements = list(zip(*batch))
        # first element after transpose is List[List[Int]] for each text
        X = batch_elements[0]
        # pad_sequence to same length
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        # second element after transpose is List[Int] for tag
        y = batch_elements[1]
        y = torch.tensor(y)
        return X, y


class CNNClassification:
    def __init__(self, train_file, test_file, embedding_dim=64, window_size=3, num_filters=64,
                 epoch_num=100, batch_size=32):
        self.data = SentimentDataset(train_file, test_file)
        self.train_dataset = TorchDataset(self.data.train_data, window_size)
        self.dev_dataset = TorchDataset(self.data.dev_data, window_size)
        self.vocab_size = self.data.vocab_size
        self.num_tags = self.data.num_tags
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_filters = num_filters
        self.epoch_num = epoch_num
        self.batch_size = batch_size

    def train(self):
        model = CNN(self.vocab_size, self.embedding_dim, self.num_filters, self.window_size, self.num_tags)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                                   collate_fn=self.train_dataset.collate_fn)
        dev_loader = torch.utils.data.DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=True,
                                                 collate_fn=self.dev_dataset.collate_fn)

        for epoch in range(self.epoch_num):
            start = time.time()
            print("epoch %d starts" % epoch)

            train_loss, train_correct = 0.0, 0.0
            train_length = 0

            for idx, (batch_X, batch_y) in enumerate(train_loader):
                # batch_X size(batch_size, padded_length), batch_y size (batch_size)
                # print("batch: ", idx, batch_X.size(), batch_y.size())
                train_length += batch_X.size()[0]
                model.zero_grad()
                scores = model(batch_X)
                loss = criterion(scores, batch_y)
                train_loss += loss.item()

                probs = torch.softmax(scores, dim=1)
                pred = probs.argmax(dim=1)
                corrects = pred == batch_y
                train_correct += corrects.sum()
                # print("probs: ", probs.size(), pred.size(), corrects.size(), corrects.sum())

                loss.backward()
                optimizer.step()

            print("train loss/sent=%.4f, acc=%.4f, time=%.2fs"
                  % (train_loss/train_length, train_correct/train_length, time.time() - start))

            test_correct = 0.0
            test_length = 0
            for idx, (batch_X, batch_y) in enumerate(dev_loader):
                test_length += batch_X.size()[0]
                scores = model(batch_X)
                probs = torch.softmax(scores, dim=1)
                pred = probs.argmax(dim=1)
                corrects = pred == batch_y
                test_correct += corrects.sum()
            print("test acc=%.4f" % (test_correct / test_length))


if __name__ == '__main__':
    train_file, dev_file = "../data/classes/train.txt", "../data/classes/test.txt"
    cnn = CNNClassification(train_file, dev_file)
    cnn.train()


