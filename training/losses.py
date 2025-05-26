import torch
from torch import nn
from torch.nn.modules.loss import _WeightedLoss


class CoxLoss(_WeightedLoss):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    def forward(
        self,
        hazard_pred: torch.Tensor,
        survtime: torch.Tensor,
        censor: torch.Tensor,
    ):
        censor = censor.float()
        current_batch_len = len(survtime)
        # modified for speed
        R_mat = survtime.reshape((1, current_batch_len)) >= survtime.reshape(
            (current_batch_len, 1)
        )
        # epsilon = 1e-7 # To prevent log(0)
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean(
            (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor
        )
        return loss_cox
