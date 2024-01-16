from collections import Counter
import os
import nltk
from nltk.tokenize import RegexpTokenizer
import random
import time
import numpy as np
import torchtext
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def flatten(l):
    """"flatten nested list l, return flat list"""
    return [item for sublist in l for item in sublist]


class RNN1(nn.Module):
    """GRU (one-direction) + FC"""
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(RNN1, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, bias=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, input, hidden):
        # use given hidden for initial hidden states
        all_h, last_h = self.gru(input, hidden)
        # one layer and one direction
        output = self.linear(last_h[0])
        # return last_h together with output to re-feed for next batch
        return output, last_h

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.context_dim, requires_grad=True)


class RNN2(nn.Module):
    """Bidirectional GRU, embedding already created from glove-vector
    GRU(input_dim, context_dim), then gru(input, hidden)
    input of shape (seq_len, batch_size, input_dim)
    initial h_0 of shape (num_layers * num_directions, batch_size, hidden_dim)
    output shape (seq_len, batch_size, num_directions * hidden_dim)
    hidden h_n of shape (num_layers * num_directions, batch_size, hidden_dim)

    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(RNN2, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, bias=True, batch_first=True,
                          dropout=0, bidirectional=True)
        # 2 * hidden_dim, because bidirectional=True
        self.linear = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, input, hidden):
        # last_h shape (2, batch_size, hidden_dim)
        all_h, last_h = self.gru(input, hidden)
        # after concat, shape (batch_size, 2 * hidden_dim)
        concated_h = torch.cat([last_h[0], last_h[1]], 1)
        # after linear, output shape (batch_size, num_classes)
        output = self.linear(concated_h)
        return output, last_h

    def init_hidden(self, batch_size):
        # initialize hidden with zeros, (num_layers * num_directions, batch_size, hidden_dim)
        return torch.zeros(2, batch_size, self.hidden_dim, requires_grad=True)


class IMDBDataset:
    """read IMDB dataset, create training_set, testing_set, and torchtext.vocab.Vocab instance
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.train_dir = os.path.join(filepath, 'train')
        self.test_dir = os.path.join(filepath, 'test')
        # read in texts and labels
        self.train_texts, self.train_labels = self.read_data(self.train_dir)
        self.test_texts, self.test_labels = self.read_data(self.test_dir)
        
        # specify tokenizer
        self.sent_tokenize = nltk.sent_tokenize
        self.word_tokenize = RegexpTokenizer(r'\w+').tokenize
        # create vocab_counter
        self.vocab_counter = Counter(flatten([self.paragraph_to_words(text) for text in self.train_texts]))
        # create torchtext.vocab.Vocab instance
        self.w2v = self.create_w2v()

        # maximal length of words in a sentence or paragraph, chosen by average length in dataset
        self.max_seq_len = 250
        
        # zip text and labels
        self.training_set = list(zip(self.train_texts, self.train_labels))
        self.testing_set = list(zip(self.test_texts, self.test_labels))

    @staticmethod
    def read_data(path):
        """given path for IMDB dataset, train or test, load text into one list, and labels into another list
        @return texts: List[String]
        @return labels: List[Int]
        """
        texts, labels = list(), list()
        # both train and test have two directories, one neg and one pos
        for label_type in ['neg', 'pos']:
            dir_name = os.path.join(path, label_type)
            for fname in os.listdir(dir_name):
                if fname[-4:] == '.txt':
                    f = open(os.path.join(dir_name, fname))
                    # f.read() return one long string of entire review
                    texts.append(f.read())
                    f.close()
                label = 1 if label_type == 'pos' else 0
                labels.append(label)
        return texts, labels

    def paragraph_to_words(self, text):
        """given a paragraph (multiple sentences), tokenize into a list of words
        :param text: String
        :return List[String]
        """
        # first sent_tokenize, then word_tokensize, create nested list
        words = [self.word_tokenize(s) for s in self.sent_tokenize(text)]
        # list comprehension to flatten nested list
        return flatten(words)

    def create_w2v(self):
        """create an instance of torchtext.vocab.Vocab using vocab_counter and glove vectors"""
        glove_path = ""
        vectors = torchtext.vocab.Vectors(name=glove_path, cache='./glove-vectors')
        w2v = torchtext.vocab.Vocab(self.vocab_counter, max_size=20000, min_freq=3, vectors=vectors)
        return w2v

    def text_to_vec(self, text):
        """convert text, a paragraph, to word vectors using w2v, with concatenated vector for each word
        :return tensor size(1, seq_len, embedding_dim)
        """
        # convert word on index using self.w2v.stoi, and slice to self.max_seq_len
        indexes = [self.w2v.stoi[word] for word in self.paragraph_to_words(text)][0: self.max_seq_len]
        # use self.w2v.vectors, size (len(indexes), 1, embedding_dim)
        sent_word_vectors = torch.cat([self.w2v.vectors[i].view(1, -1) for i in indexes]).view(len(indexes), 1, -1)
        # resize to (1, len(indexes), embedding_dim)
        sent_word_vectors = sent_word_vectors.view(1, len(indexes), -1)
        return sent_word_vectors
    

def get_batch(dataset, mode, start_idx, end_idx):
    """given IMDBdataset instance, generate batch with start_idx and end_idx
    :param dataset: IMDBDataset instance
    :param mode: String, train or test
    :param start_idx and end_idx: Int, index for start and end
    :return batch_inputs: PackedSequence, size (batch_size, max_seq_len, embedding_dim),
    :return  labels: tensor, size (batch_size)
    """
    if mode == 'train':
        input_dataset = dataset.training_set
    elif mode == 'test':
        input_dataset = dataset.testing_set
    else:
        raise ValueError("mode must be train or test")

    # slice input_dataset to get batch
    batch = input_dataset[start_idx: end_idx]
    # two lists, one for texts and one for labels
    input_texts, labels = zip(*batch)
    # convert texts to vectors, each one (1, seq_len, embedding_dim)
    input_vectors = [dataset.text_to_vec(text) for text in input_texts]
    # convert labels to tensors
    labels = torch.tensor(labels, dtype=torch.long)

    # get seq_len for each sequence in the input_texts
    seq_lens = torch.tensor([vec.shape[1] for vec in input_vectors], dtype=torch.long)
    # print("seq_lens: ", seq_lens.max(), seq_lens)
    embedding_dim = input_vectors[0].shape[2]
    # initialize with zeros, size (batch_size, max_seq_len, embedding_dim); seq_lens often chop to 250
    batch_inputs = torch.zeros(len(seq_lens), seq_lens.max(), embedding_dim)

    # fill in batch_inputs for each sentence in input_text, sliced to max_seq_len
    for idx, (seq, seq_len) in enumerate(zip(input_vectors, seq_lens)):
        batch_inputs[idx, :seq_len] = seq

    # sort tensor by seq_lens, and rearrange batch_inputs and labels
    seq_lens, perm_idx = seq_lens.sort(dim=0, descending=True)
    batch_inputs = batch_inputs[perm_idx]
    # pack_padded_sequence create a PackedSequence object
    batch_inputs = pack_padded_sequence(batch_inputs, seq_lens.numpy(), batch_first=True)
    labels = labels[perm_idx]

    return batch_inputs, labels


def train_rnn(dataset):
    """given an instance of IMDBDataset, train and evaluate RNN model"""
    learning_rate = 0.001
    batch_size = 50
    num_passes = 25000 // batch_size
    num_epochs = 5
    input_dim = 100
    hidden_dim = 50
    num_classes = 2

    criterion = nn.CrossEntropyLoss()
    model = RNN2(input_dim, hidden_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        start_time = time.time()
        # shuffle training_set and init hidden
        random.shuffle(dataset.training_set)
        # reinitialize hidden to zero after each epoch
        hidden = model.init_hidden(batch_size)

        model.train()

        for i in range(num_passes):
            # use i and batch_size to get batch by index
            start_idx = i * batch_size
            end_idx = (i+1) * batch_size
            # specify train mode to get_batch
            inputs, labels = get_batch(dataset, 'train', start_idx, end_idx)

            # starting each batch, detach hidden state
            hidden.detach_()

            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('pass [%d/%d], in epoch [%d/%d] Loss: %.4f' %
                      (i+1, num_passes, epoch, num_epochs, loss.item()))

        end_time = time.time()
        print(f'Epoch: {epoch + 1:02} | Time: {np.round(end_time - start_time, 0)}s')

        # evaluate after each epoch
        model.eval()
        # use entire testing_set in one batch
        len_test = len(dataset.testing_set)
        testing_inputs, testing_labels = get_batch(dataset, 'test', 0, len_test)
        # need to specifically initialize hidden for this batch size
        hidden = model.init_hidden(len_test)
        outputs, hidden = model(testing_inputs, hidden)
        _, predicted = torch.max(outputs.data, dim=1)
        total = testing_labels.size(0)  # this is int
        correct = (predicted == testing_labels.data).sum().item() # sum is still tensor, take out number
        print('Accuracy of the network on the  test data: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    imdb_dir = '../data/imdb/aclImdb'
    imdb_dataset = IMDBDataset(imdb_dir)
    train_rnn(imdb_dataset)

