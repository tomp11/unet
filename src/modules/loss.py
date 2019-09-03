import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    print(target[:30])
    print(input.size(), target.size())
    print(target)
    loss = F.cross_entropy(
        input, target, weight=weight, ignore_index=250, reduction='mean'
    )
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            target.device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)

class CEDiceLoss(nn.Module):
    # 多分うまくいってない
    def __init__(self):
        super(CEDiceLoss, self).__init__()

    def forward(self, input, target):
        torch.autograd.set_detect_anomaly(True)
        # onehotにできたのでBCElossでもいいかも
        # その時はsigmoid忘れずに
        ce = F.cross_entropy(input, target)
        """
        input [batch_size, num_classes, size, size]
        target [batch_size, size, size]
        """
        smooth = 1e-5
        batch_size = input.size(0)
        n_classes = input.size(1)
        image_size = input.size(2)
        input = torch.sigmoid(input)


        target_one_hot = torch.FloatTensor(batch_size, n_classes, image_sidatarootze, image_size).zero_().cuda()
        target = target.unsqueeze_(1)
        target = target.clone().detach()

        # print(target_one_hot.size(), target.size())
        target_one_hot = target_one_hot.scatter_(1, target, 1)
        # target_one_hot = target_one_hot.clone().detach()

        input = input.view(batch_size, -1)
        target_one_hot = target_one_hot.view(batch_size, -1)

        # target_one_hot = torch.zeros(batch_size, n_classes, image_size*image_size).cuda()
        # for i in range(batch_size):
        #     target_one_hot[i, target[i], torch.arange(image_size*image_size)] = 1
        # target_one_hot = target_one_hot.view(batch_size, -1)

        intersection = (input * target_one_hot)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target_one_hot.sum(1) + smooth)
        dice = 1 - dice.sum() / batch_size
        return 0.5 * ce + dice

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(input, target)
        # binary_cross_entropy_with_logitsは下２つと同じ
        # そう考えるとunetの最後sigmoidでもよかった
        # pred = torch.sigmoid(x)
        # loss = F.binary_cross_entropy(pred, y)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        loss = 0.5 * bce + dice
        return bce


if __name__ == "__main__":
    input = torch.randn(8, 3, 224, 224)
    # target = torch.LongTensor(8, 224, 224)
    target = torch.randint(3,(8, 224, 224), dtype=torch.int64)
    # print(target)
    criterion = CEDiceLoss()
    loss = criterion(input, target)
    print(loss)
