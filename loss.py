from machine.loss import NLLLoss
import torch

class AttentionLoss(NLLLoss):
    """ Cross entropy loss over attentions

    Args:
        ignore_index (int, optional): index of token to be masked
    """
    _NAME = "Attention Loss"
    _SHORTNAME = "attn_loss"
    _INPUTS = "attention_score"
    _TARGETS = "attention_target"

    def __init__(self, ignore_index=-1):
        super(AttentionLoss, self).__init__(ignore_index=ignore_index, size_average=True)

    def eval_step(self, step_outputs, step_target):
        batch_size = step_target.size(0)
        outputs = torch.log(step_outputs.contiguous().view(batch_size, -1).clamp(min=1e-20))
        self.acc_loss += self.criterion(outputs, step_target)
        self.norm_term += 1
