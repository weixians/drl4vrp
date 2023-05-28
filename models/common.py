import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        """
        @param input_size: 2，表示每个节点特征(static/dynamic)的维度
        @param hidden_size: 128
        """
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size), device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, rnn_out):
        """
        @param static_hidden: (B,H,L)
        @param dynamic_hidden: (B,H,L)
        @param rnn_out: (B,H)
        """
        batch_size, hidden_size, _ = static_hidden.size()
        # hidden: (B,H,L)，扩展成和static_hidden、dynamic_hidden一样的维度
        hidden = rnn_out.unsqueeze(2).expand_as(static_hidden)
        # 三个变量cat在一起：(B,3H,L)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply (为方便做矩阵乘法，扩展第一维从1到batch_size)
        # (1,1,H) -> (B,1,H)
        v = self.v.expand(batch_size, -1, -1)
        # (1,H,3H) -> (B,H,3H)
        W = self.W.expand(batch_size, -1, -1)

        # (B,1,L) = (B,1,H)@(B,H,L) = (B,1,H)@((B,H,3H)@(B,3H,L))
        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        # (B,1,L)
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)
        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size), device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        """
        @param static_hidden: (B,H,L)
        @param dynamic_hidden: (B,H,L)
        @param decoder_hidden: (B,H,1)
        @param last_hh: (1,B,H)
        """
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)

        # Given a summary of the output, find an  input context
        # (B,1,L)
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        # (B,1,H) = (B,1,L)@(B,L,H)：attention加权和所有特征
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        # 复制L份：(B,1,H) -> (B,H,1) -> (B,H,L)，给每个节点配上一份特征
        context = context.transpose(1, 2).expand_as(static_hidden)  # (B, num_feats, 1)
        # (B,2H,L)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)
        # (1,1,H) -> (B,1,H)
        v = self.v.expand(static_hidden.size(0), -1, -1)
        # (1,H,2H) -> (B,H,2H)
        W = self.W.expand(static_hidden.size(0), -1, -1)
        # (B,1,L) = (B,1,H)@(B,H,L) = (B,1,H)@((B,H,2H)@(B,2H,L))
        # (B,1,L) -> (B,L)
        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh
