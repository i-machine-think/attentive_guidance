import torch
import torch.nn as nn

from machine.models.attention import Attention as AttentionBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(AttentionBase):
    """
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output
        method(str): The method to compute the alignment, mlp or dot

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
        method (torch.nn.Module): layer that implements the method of computing the attention vector

    Examples::

         >>> attention = machine.models.Attention(256)
         >>> context = torch.randn(5, 3, 256)
         >>> output = torch.randn(5, 5, 256)
         >>> output, attn = attention(output, context)

    """

    def get_method(self, method, dim):
        """
        Set method to compute attention
        """
        try:
            method = super(Attention, self).get_method(method, dim)
        except ValueError:
            if method == 'hard':
                method = HardGuidance()
        return method

class HardGuidance(nn.Module):
    """
    Attention method / attentive guidance method for data sets that are annotated with attentive guidance.
    """

    def forward(self, decoder_states, encoder_states, step, provided_attention):
        """
        Forward method that receives provided attentive guidance indices and returns proper
        attention scores vectors.

        Args:
            decoder_states (torch.FloatTensor): Hidden layer of all decoder states (batch, dec_seqlen, hl_size)
            encoder_states (torch.FloatTensor): Output layer of all encoder states (batch, dec_seqlen, hl_size)
            step (int): The current decoder step for unrolled RNN. Set to -1 for rolled RNN
            provided_attention (torch.LongTensor): Variable containing the provided attentive guidance indices (batch, max_provided_attention_length)

        Returns:
            torch.tensor: Attention score vectors (batch, dec_seqlen, hl_size)
        """

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, _ = encoder_states.size()
        _,          dec_seqlen, _ = decoder_states.size()

        attention_indices = provided_attention.detach()
        # If we have shorter examples in a batch, attend the PAD outputs to the first encoder state
        attention_indices.masked_fill_(attention_indices.eq(-1), 0)

        # In the case of unrolled RNN, select only one column
        if step != -1:
            attention_indices = attention_indices[:, step]

        # Add a (second and) third dimension
        # In the case of rolled RNN: (batch_size x dec_seqlen) -> (batch_size x dec_seqlen x 1)
        # In the case of unrolled:   (batch_size)              -> (batch_size x 1          x 1)
        attention_indices = attention_indices.contiguous().view(batch_size, -1, 1)
        # Initialize attention vectors. These are the pre-softmax scores, so any
        # -inf will become 0 (if there is at least one value not -inf)
        attention_scores = torch.full([batch_size, dec_seqlen, enc_seqlen], fill_value=-float('inf'), device=device)
        attention_scores = attention_scores.scatter_(dim=2, index=attention_indices, value=1)
        attention_scores = attention_scores

        return attention_scores

