import torch
import torch.nn as nn
import torch.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, enc_emb_dim, enc_hid_dim, dec_hid_dim, num_layers=2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(enc_emb_dim, enc_hid_dim, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, input):
        embed = self.embedding(input)   # (max_len, bs, emb_dim)
        output, hidden = self.gru(embed) 
        hidden = self.fc(torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim=-1))

        return output, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, 1)

    def forward(self, enc_output, hidden):
        # hidden    (bs, dec_hid_dim)
        # enc_output (seq_len, bs, enc_hid_dim * 2)
        seq_len = enc_output.size(0)
        hidden = hidden.squeeze(1)
        hidden = hidden.repeat(1, seq_len, 1)
        enc_output = enc_output.transpose(0,1)
        energy = self.attn(torch.cat([enc_output, hidden], dim=-1))
        attention = F.softmax(energy.squeeze(-1))

        return attention  # (bs, seq_len)


class Decoder(nn.Module):
    def __init__(self, vocab_size, dec_emb_dim, dec_hid_dim, attention, num_layers=2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dec_emb_dim)
        self.gru = nn.GRU(dec_emb_dim, dec_hid_dim, num_layers=num_layers)
        self.attention = attention
        self.logit = nn.Linear(dec_hid_dim, vocab_size)

    def forward(self, input, hidden, enc_output):
        embed = self.embedding(input)
        attn_weight = self.attention(enc_output, hidden) # (bs, seq_len)
                                                         # (seq_len, bs, enc_hid_dim * 2)
        attn_weight = attn_weight.unsqueeze(1)
        enc_output = enc_output.transpose(0,1)
        context = torch.bmm(attn_weight, enc_output)    # (bs, 1, enc_hid_dim * 2)
        context = context.transpose(0,1)
        inp = torch.cat([context, embed], dim=-1)

        output, hidden = self.gru(inp, hidden)

        return output

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()

# vocab
EN_VOCAB = 1000
FR_VOCAB = 800
# embedding dim
EN_EMB_DIM = 100
FR_EMB_DIM = 100
# hidden dimension
ENC_HID_DIM = 20
DEC_HID_DIM = 15
# other
BATCH_SIZE = 32
SEQ_LEN = 10





