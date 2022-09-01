from typing import Optional, List

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


from model.loss_function.SentimentControlLoss import SentimentControlLoss, sentiment_control_loss, \
    sentiment_overall_loss


class ConfidenceCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super(ConfidenceCrossEntropyLoss, self).__init__(weight, size_average, ignore_index,
                                                         reduce, reduction, label_smoothing)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        pass


class ConfidenceSentimentControlLoss(torch.nn.Module):
    def __init__(self, alpha, mask_mean, sub_text_seg=7359, padding_value=-1, start_p=1, end_p=None):
        super(ConfidenceSentimentControlLoss, self).__init__()
        self.sub_text_seg = sub_text_seg
        self.padding_value = padding_value
        self.start_p = start_p
        self.end_p = end_p
        self.alpha = alpha
        self.mask_mean = mask_mean
        self.loss_fct = CrossEntropyLoss(ignore_index=self.padding_value, size_average=True)

    def forward(self,arg_config, input_labels, generation_logits, sentiment_atoms: List[List[List]],
                confidences):
        return sentiment_overall_loss(arg_config=arg_config,
                                      input_labels=input_labels, generation_logits=generation_logits,
                                      sentiment_atoms=sentiment_atoms, cross_entropy=self.loss_fct,
                                      confidences=confidences,
                                      padding_value=self.padding_value,
                                      start_p=self.start_p, end_p=self.end_p, alpha=self.alpha,
                                      mask_mean=self.mask_mean)
