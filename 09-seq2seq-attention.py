import torch
import torch.nn as nn
import numpy as np
import math
import random
import time
import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k


HIDDEN_DIM = 256
BATCH_SIZE = 128
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, pf_dim, drop_out, max_len=100):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, num_heads, pf_dim, drop_out)
                                     for _ in range(num_layers)])
        self.dropout = nn.Dropout(drop_out)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, src, src_mask):
        # src size (batch_size, src_len), src_mask size (batch_size, 1, 1, src_len)
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # create pos for position embedding, pos size (batch_size, src_len)
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1)
        # add scaled token_embedding to position embedding, src size (batch_size, src_len, hidden_dim)
        src = self.dropout((self.token_embedding(src) * self.scale) + self.pos_embedding(pos))

        # after token_embedding and position_embedding, go through each EncoderLayer
        for layer in self.layers:
            src = layer(src, src_mask)
        # src size (batch_size, src_len, hidden_dim)
        return src


class EncoderLayer(nn.Module):
    """Layer within Encoder
    First layer is self_attention
    Second layer is LayerNorm with dropout and residual connection
    Third layer is positionwise_feedforward
    Fourth and Final layer is Layernorm with dropout and residual connection
    """
    def __init__(self, hidden_dim, num_heads, pf_dim, drop_out):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, num_heads, drop_out)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, drop_out)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, src, src_mask):
        # First block is self attention, use src/src/src as query/key/value, size (batch_size, src_len, hidden_dim)
        _src, _ = self.self_attention(src, src, src, src_mask)
        # after attention, apply dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # Second block is positionwise_feedforward
        _src = self.positionwise_feedforward(src)
        # after positionwise_feedforward, apply dropout, residual connection and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src size (batch_size, src_len, hidden_dim)
        return src


class MultiHeadAttentionLayer(nn.Module):
    """
    single-head attention is scaled dot attention, =dot(softmax(dot(Q,K_T)/sqrt(d_k)), V)
    Multi-head attention is to first linear, then divide into n-heads, apply same scaled-dot attention to each head,
    concat all heads together, apply final linear layer for output
    """
    def __init__(self, hidden_dim, num_heads, drop_out):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # dimension of each head in multihead

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(drop_out)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, query, key, value, mask=None):
        # apply Linear layer to query/key/value
        Q = self.fc_q(query)  # size (batch_size, query_len, hidden_dim)
        K = self.fc_k(key)    # size (batch_size, key_len, hidden_dim)
        V = self.fc_v(value)  # size (batch_size, value_len, hidden_dim)

        batch_size = query.shape[0]
        # divide into multi-heads, size (batch_size, num_heads, query_len/key_len/value_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # energy is scaled dot of Q and K, dot(Q, K_T)/scale, size (batch_size, num_heads, query_len, key_len)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # mask the energy
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        # apply softmax, attention size (batch_size, num_heads, query_len, key_len)
        attention = torch.softmax(energy, dim=-1)
        # multiply attention to V, and apply dropout, size (batch_size, num_heads, query_len, head_dim)
        x = torch.matmul(self.dropout(attention), V)

        # to concat heads together,first permute, size (batch_size, query_len, num_heads, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        # after view, heads concated together, size (batch_size, query_len, hidden_dim)
        x = x.view(batch_size, -1, self.hidden_dim)
        # with all heads concated, apply final linear layer, size (batch_size, query_len, hidden_dim)
        x = self.fc_o(x)

        # x size (batch_size, query_len, hidden_dim), attention size (batch_size, num_heads, query_len, key_len)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    """transform from hidden_dim to pf_dim
    original trnasformer used hidden_dim=256, pf_dim=2048, ReLU activation"""
    def __init__(self, hidden_dim, pf_dim, drop_out):
        super().__init__()
        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, input):
        # input size (batch_size, seq_len, hidden_dim)
        x = self.dropout(torch.relu(self.fc_1(input)))
        # after two layers, x size (batch_size, seq_len, hidden_din)
        x = self.fc_2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, num_heads, pf_dim, drop_out, max_len=100):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, num_heads, pf_dim, drop_out)
                                     for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop_out)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """Input size
        trg size (batch_size, trg_len)
        enc_src size (batch_size, src_len, hidden_dim)
        trg_mask (batch_size, 1, trg_len, trg_len)
        src_mask size (batch_size, 1, 1, src_len)
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # create pos for position embedding, pos size (batch_size, trg_len)
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1)
        # apply token_embedding, add to pos_embedding, apply dropout, size (batch_size, trg_len, hidden_dim)
        trg = self.dropout((self.token_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # after token_embedding and position_embedding, go through each Decoder layer
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # final linear layer
        output = self.fc_out(trg)

        # output size (batch_size, trg_len, output_dim), attention size (batch_size, num_heads, trg_len, src_len)
        return output, attention


class DecoderLayer(nn.Module):
    """Decoder layer within Decoder
    First layer is self_attention to trg
    Second layer is LayerNorm with residual connection and dropout
    Third layer is encoder_attention with trg as query, enc_src as key and value
    Fourth layer is LayerNorm with residual connection and dropout
    Fifth layer is positionwise_feedforward
    Six and final layer is LayerNorm with residual connection and dropout
    """
    def __init__(self, hidden_dim, num_heads, pf_dim, drop_out):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, num_heads, drop_out)
        self.encoder_attention = MultiHeadAttentionLayer(hidden_dim, num_heads, drop_out)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, drop_out)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """Input size
        trg size (batch_size, trg_len)
        enc_src size (batch_size, src_len, hidden_dim)
        trg_mask (batch_size, 1, trg_len, trg_len)
        src_mask size (batch_size, 1, 1, src_len)
        """
        # first perform self_attention, trg as query/key/value
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # after attention, apply dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # second perform encoder_attention, trg as query, enc_src as key and value
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # after attention, apply dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # third perform positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # after feedforward, apply dropout, residual connection and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg size (batch_size, trg_len, hidden_dim), attention size (batch_size, num_heads, trg_len, src_len)
        return trg, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        # from src (batch_size, src_len) to (batch_size, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_len = trg.shape[1]
        # from trg (batch_size, trg_len) to (batch_size, 1, 1, trg_len)
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # tril is lower triangular matrix, upper is zero, size (batch_size, 1, 1, trg_len)
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).bool()
        # combine two masks
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, trg):
        # src_mask size (batch_size, 1, 1, src_len)
        src_mask = self.make_src_mask(src)
        # trg_mask size (batch_size, 1, 1, trg_len)
        trg_mask = self.make_trg_mask(trg)

        # encoder step, size (batch_size, src_len, hidden_dim)
        enc_src = self.encoder(src, src_mask)

        # decoder step, output size (batch_size, trg_len, output_dim)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        print("batch: ", i, src.size(), trg.size())

        optimizer.zero_grad()
        # output size (batch_size, trg_len-1, output_dim)
        output, _ = model(src, trg[:, :-1])

        output_dim = output.shape[-1]
        # reshape to (batch_size * (trg_len - 1), output_dim)
        output = output.contiguous().view(-1, output_dim)
        # remove first token, flatten (batch_size * (trg_len - 1))
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def get_data():
    spacy_german = spacy.load('de_core_news_sm')
    spacy_english = spacy.load('en_core_web_sm')

    def tokenize_english(text):
        return [token.text for token in spacy_english.tokenizer(text)]

    def tokenize_german(text):
        return [token.text for token in spacy_german.tokenizer(text)]

    SOURCE = Field(tokenize=tokenize_german, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
    TARGET = Field(tokenize=tokenize_english, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

    train_data, valid_data, test_data = Multi30k.splits(root='../data', exts=('.de', '.en'), fields=(SOURCE, TARGET))
    print("train_data: ", len(train_data), len(valid_data), len(test_data))

    SOURCE.build_vocab(train_data, min_freq=2)
    TARGET.build_vocab(train_data, min_freq=2)
    print("Germany/source vocab size: ", len(SOURCE.vocab))
    print("English/target vocab size: ", len(TARGET.vocab))

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE
    )

    return train_iterator, valid_iterator, test_iterator, SOURCE, TARGET


def run():
    train_iterator, valid_iterator, test_iterator, SRC, TRG = get_data()
    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
    trg_pad_dix = TRG.vocab.stoi[TRG.pad_token]

    enc = Encoder(input_dim, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT)
    dec = Decoder(output_dim, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, src_pad_idx, trg_pad_dix)

    model.apply(initialize_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_dix)

    num_epochs = 10
    clip = 1
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut6-model.pt')

        print(f'Epoch: {epoch+1:02} | Time: {end_time - start_time:d}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tValidation Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}')

    model.load_state_dict(torch.load("tut6-model.pt"))
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f}')


if __name__ == '__main__':
    run()

