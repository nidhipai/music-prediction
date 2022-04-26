import torch
import torch.nn as nn
import torch.optim as optim

class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

    def forward(self, source, out_seq_len = None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]
        if out_seq_len is None:
            out_seq_len = source.shape[1]

        outputs = torch.zeros(batch_size, out_seq_len, self.decoder.output_size, device=self.device)
        _, hidden = self.encoder(source)
        sos_token = source[:, 0]
        for i in range(1, out_seq_len):
            output, hidden = self.decoder(sos_token.unsqueeze(1), hidden)
            outputs[:, i, :] = output

        return outputs



        

