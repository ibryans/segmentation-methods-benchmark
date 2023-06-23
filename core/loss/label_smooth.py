# Source: https://github.com/CoinCheung/pytorch-loss/blob/master/label_smooth.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp


# Version 1: use torch.autograd
class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, slices, targets):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> slices = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(slices, lbs)
        '''
        # overcome ignored targets
        slices = slices.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = slices.size(1)
            targets = targets.clone().detach()
            ignore = targets.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            targets[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(slices).fill_(
                lb_neg).scatter_(1, targets.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(slices)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


# Version 2: user derived grad computation
class LSRCrossEntropyFunctionV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, slices, targets, lb_smooth, lb_ignore):
        # prepare targets
        num_classes = slices.size(1)
        lb_pos, lb_neg = 1. - lb_smooth, lb_smooth / num_classes
        targets = targets.clone().detach()
        ignore = targets.eq(lb_ignore)
        n_valid = ignore.eq(0).sum()
        targets[ignore] = 0
        lb_one_hot = torch.empty_like(slices).fill_(
            lb_neg).scatter_(1, targets.unsqueeze(1), lb_pos).detach()

        ignore = ignore.nonzero(as_tuple=False)
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(slices.size(1)), *b]
        lb_one_hot[mask] = 0
        coeff = (num_classes - 1) * lb_neg + lb_pos

        ctx.variables = coeff, mask, slices, lb_one_hot

        loss = torch.log_softmax(slices, dim=1).neg_().mul_(lb_one_hot).sum(dim=1)
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        coeff, mask, slices, lb_one_hot = ctx.variables

        scores = torch.softmax(slices, dim=1).mul_(coeff)
        grad = scores.sub_(lb_one_hot).mul_(grad_output.unsqueeze(1))
        grad[mask] = 0
        return grad, None, None, None


class LabelSmoothSoftmaxCEV2(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV2, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, slices, labels):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV2()
            >>> slices = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(slices, lbs)
        '''
        losses = LSRCrossEntropyFunctionV2.apply(
                slices, labels, self.lb_smooth, self.lb_ignore)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            n_valid = (labels != self.lb_ignore).sum()
            losses = losses.sum() / n_valid
        return losses
