import torch
import torch.nn as nn
import spacy
import random
import time
import numpy as np
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k


BATCH_SIZE = 128
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROP_OUT = 0.5


def get_data():
    spacy_german = spacy.load('de_core_news_sm')
    spacy_english = spacy.load('en_core_web_sm')

    def tokenize_english(text):
        return [token.text for token in spacy_english.tokenizer(text)]

    def tokenize_german(text):
        return [token.text for token in spacy_german.tokenizer(text)]

    source = Field(tokenize=tokenize_german, init_token='<sos>', eos_token='<eos>', lower=True)
    target = Field(tokenize=tokenize_english, init_token='<sos>', eos_token='<eos>', lower=True)

    train_data, valid_data, test_data = Multi30k.splits(root='./data', exts=('.de', '.en'), fields=(source, target))

    source.build_vocab(train_data, min_freq=2)
    target.build_vocab(train_data, min_freq=2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE
    )

    return train_iterator, valid_iterator, test_iterator, source, target


class Encoder(nn.Module):
    """this encoder has LSTM with two layers, not bidirectional"""
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, drop_out):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=drop_out)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, inputs):
        # embeds size (seq_len, batch_size, embed_dim)
        embeds = self.dropout(self.embedding(inputs))
        # output size (seq_len, batch_size, hidden_dim), hidden and cell size (2, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(embeds)
        return hidden, cell


class Decoder(nn.Module):
    """this decoder has LSTM with two layers, not bidirectional
    Take trg, hidden, cell, Return pred, hidden, cell"""
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, drop_out):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=drop_out)
        self.dropout = nn.Dropout(drop_out)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, trg, hidden, cell):
        # input is one word, ensure size (1, batch_size, output_dim)
        trg = trg.unsqueeze(0)
        # embeds size (1, batch_size, embed_dim)
        embeds = self.dropout(self.embedding(trg))
        # output size (1, batch_size, hidden_dim), hidden and cell size (2, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(embeds, (hidden, cell))
        # remove dim=0, pred size (batch_size, trg_vocab_size)
        pred = self.linear(output.squeeze(0))
        return pred, hidden, cell


class Seq2SeqLSTM(nn.Module):
    """"this seq2seq takes encoder and decoder
    Take src and trg. Return outputs
    1. get hidden and cell from self.encoder(src), this initialize hidden and cell, then updated in step 2
    2. using first word in trg, word-by-word: output, hidden, cell = self.decoder(curr, hidden, cell)
    3. update outputs for that position by teacher_forcing
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_rate=0.5):
        # trg size (seq_len, batch_size)
        batch_size = trg.shape[1]
        target_len = trg.shape[0]
        target_vocab_size = self.decoder.output_dim

        # initialize outputs, which is the return value, at zero
        outputs = torch.zeros(target_len, batch_size, target_vocab_size)
        # initialize hidden and cell with last output from encoder
        # hidden and cell size (2, batch_size, hidden_dim)
        hidden, cell = self.encoder(src)
        # take the first word in trg
        curr = trg[0, :]
        # for target sequence, decode one by one after the first word
        for t in range(1, target_len):
            # use curr as input to decoder, output size (1, batch_size, hidden_dim)
            output, hidden, cell = self.decoder(curr, hidden, cell)
            outputs[t] = output
            top = output.argmax(dim=1)
            # if teacher forcing, use actual next token; else use highest predicted token
            curr = trg[t] if (random.random() < teacher_forcing_rate) else top

        return outputs


def initialize_weights(m):
    """initialize all parameters in model"""
    print("initialize: ", m.named_parameters)
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)


def train(model, iterator, optimizer, criterion, clip):
    epoch_loss = 0
    model.train()
    for i, batch in enumerate(iterator):
        src = batch.src  # size (seq_len, batch_size)
        trg = batch.trg  # size (seq_len, batch_size)
        print("batch number: ", i, src.size(), trg.size())

        optimizer.zero_grad()
        # output size (seq_len, batch_size, trg_vocab_size)
        output = model(src, trg)
        output_dim = output.shape[-1]
        # reshape output to (seq_len * batch_size, trg_vocab_size)
        output = output[1:].view(-1, output_dim)
        # remove first token and flatten to 1D
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval_pred()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def run():
    train_iterator, valid_iterator, test_iterator, source, target = get_data()

    input_dim = len(source.vocab)
    output_dim = len(target.vocab)

    encoder = Encoder(input_dim, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROP_OUT)
    decoder = Decoder(output_dim, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROP_OUT)
    model = Seq2SeqLSTM(encoder, decoder)
    model.apply(initialize_weights)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=target.vocab.stoi[target.pad_token])

    num_epoch = 10
    grad_clip = 1
    lowest_validation_loss = float('inf')

    for epoch in range(num_epoch):
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, grad_clip)
        valid_loss = evaluate(model, valid_iterator, criterion)
        end_time = time.time()

        if valid_loss < lowest_validation_loss:
            lowest_validation_loss = valid_loss
            torch.save(model.state_dict(), "tut1-model.pt")

        print(f'Epoch: {epoch + 1:02} | Time: {np.round(end_time - start_time, 0)}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {np.exp(train_loss):7.3f}')
        print(f'\tVal. Loss: {valid_loss:.4f} | Val. PPL: {np.exp(valid_loss):7.3f}')

    model.load_state_dict(torch.load("tut1-model.pt"))
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f}')


if __name__ == '__main__':
    run()



