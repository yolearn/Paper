import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    '''Encoder 
    Args:
        src_dict_dim: source dictionary dimention 
        embed_dim: embedding dimension
        hidden_size: GRU hidden size
    '''

    def __init__(self, src_dict_dim: int, embed_dim: int, enc_hidden_size: int, dec_hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(src_dict_dim, embed_dim)
        self.GRU = nn.GRU(embed_dim, enc_hidden_size, bidirectional=True)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x):
        '''
        Args:
            x: input (seq_len, bs)
        Return:
            output: RNN output (seq_len, bs, num_directions * hidden_size)
            hidden: RNN hidden state (bs, num_directions * hidden_size)
        '''
        embedding = self.embed(x)               # (seq_len, bs, embed_dim)
        output, hidden = self.GRU(embedding)
        hidden = self.fc(torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=-1))

        return output, hidden


class AdaptiveAttention(nn.Module):
    '''Attention
    Args:
        enc_hid_dim: encoding hidden dimension
        dec_hid_dim: decoding hidder dimension
    Return:
        attn_weight: attention weight (bs, max_len)
    '''

    def __init__(self, enc_hid_dim: int, dec_hid_dim: int):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, 1)

    def forward(self, hidden, enc_outputs):
        ''' Do Attention betweeen encoding outputs and the previous hidden state
        Args:
            hidden: the previous hidden state (bs, num_directions * hidden_size)
            enc_outputs: encoding output (seq_len, bs, num_directions * hidden_size)
        Return:
            attn_weight: attention weight (bs, seq_len)
        '''
        max_len = enc_outputs.size(0)
        hidden = hidden.unsqueeze(1).repeat(1, max_len, 1)
        enc_outputs = torch.transpose(enc_outputs, 0, 1)   # (bs, seq_len, num_directions * hidden_size)    
        energy = self.attn(torch.cat([enc_outputs, hidden], dim=-1))
        attn_weight = F.softmax(energy.squeeze(-1), dim=-1)

        return attn_weight


class Decoder(nn.Module):
    '''Decoder
    Args:
        tgt_dict_dim: target dictionary dimenstion
        embed_dim: embedding dimension
        hidden_size: GRU hidden size 
        attention: attention layer
    '''
    def __init__(self, tgt_dict_dim: int, enc_hid_dim: int, embed_dim: int, dec_hidden_size: int, attention):
        super().__init__()
        self.embed = nn.Embedding(tgt_dict_dim, embed_dim)
        self.GRU = nn.GRU(enc_hid_dim * 2 + embed_dim, dec_hidden_size)
        self.attention = attention
        self.fc = nn.Linear(dec_hidden_size, tgt_dict_dim)

    def forward(self, x, enc_outputs, hidden):
        '''
        Args:
            x: input (1, bs)
            hidden: the previous hidden state (bs, num_directions * hidden_size)
            enc_outputs: encoding output (seq_len, bs, num_directions * hidden_size)
        Return:
            output: RNN output (1, bs, num_directions * hidden_size)
            hidden: RNN hidden state (1, bs, hidden_size):
        '''
        embedding = self.embed(x)     
             
        weight = self.attention(hidden, enc_outputs)    # (bs, seq_len)
        weight = weight.unsqueeze(1)                    # (bs, 1, seq_len)
        enc_outputs = enc_outputs.transpose(0,1)        # (bs, seq_len, num_directions * hidden_size) 
        context = torch.bmm(weight, enc_outputs)
        context = context.transpose(0,1)                
        context = torch.cat([embedding, context], dim=-1)
        
        output, hidden = self.GRU(context, hidden.unsqueeze(0))
        output = output.squeeze(0)
        prediction = self.fc(output)

        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg):
        '''
        Args:
            src: (src_len, batch size)
            trg: (trg_len, batch size)
        Return:

        '''

        bs = trg.size(1)
        trg_len = trg.size(0)
        tgt_dict_dim = TGT_DICT_DIM 
        outputs = torch.zeros(trg_len, bs, tgt_dict_dim)

        enc_output, hidden = self.encoder(src)
        input = trg[0, :]
        input = input.unsqueeze(0)

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, enc_output, hidden)
            outputs[t] = output
            input = trg[t, :]
            input = input.unsqueeze(0)
            
        return outputs


if __name__ == "__main__":
    SRC_DICT_DIM = 1000
    TGT_DICT_DIM = 1000
    ENC_EMBED_DIM = 30
    DEC_EMBED_DIM = 25
    ENC_HIDDEN_SIZE = 20
    DEC_HIDDEN_SIZE = 15
    BATCH_SIZE = 32
    LENGTH = 10

    # Encoder test
    encoder = Encoder(SRC_DICT_DIM, ENC_EMBED_DIM, ENC_HIDDEN_SIZE, DEC_HIDDEN_SIZE)
    input = torch.zeros(LENGTH, BATCH_SIZE).type(torch.long)
    enc_output, hidden = encoder(input)
    print(f'Encoder output size: {enc_output.size()}')
    print(f'Encoder hidden size: {hidden.size()}')

    # Attention test
    attention = AdaptiveAttention(ENC_HIDDEN_SIZE, DEC_HIDDEN_SIZE)
    init_h = torch.zeros(BATCH_SIZE, DEC_HIDDEN_SIZE)
    atten_weight = attention(init_h, enc_output)
    print(f'Attention weight size: {atten_weight.size()}')

    # Decoder test
    decoder = Decoder(TGT_DICT_DIM, ENC_HIDDEN_SIZE, DEC_EMBED_DIM, DEC_HIDDEN_SIZE, attention)
    input = torch.zeros(1, BATCH_SIZE).type(torch.long)
    hidden = torch.zeros(BATCH_SIZE, DEC_HIDDEN_SIZE) 
    dec_output, hidden = decoder(input, enc_output, hidden)
    print(f'Decoder output size: {dec_output.size()}')
    print(f'Decoder hidden size: {hidden.size()}')

    #Seq2Seq test
    
    seq2seq = Seq2Seq(encoder, decoder)
    src = torch.zeros(LENGTH, BATCH_SIZE).type(torch.long)
    tgt = torch.zeros(LENGTH, BATCH_SIZE).type(torch.long)
    outputs = seq2seq(src, tgt)
    print(f'Seq2Seq outputs size: {outputs.size()}')
    