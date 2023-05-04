import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def lossact(predict, target, mask, loss_sum):
    for ind in range(predict.size(0)):
        for i in range(predict.size(1)):
            masktemp = mask[ind]
            pre_onentity = predict[ind, i]
            pre_list = pre_onentity[masktemp]
            tar = target[ind, i]
            tar_list = tar[masktemp]
            if len(pre_list) != 0:
                pre_softmax = F.softmax(pre_list)
                #print(type(pre_softmax), pre_softmax.shape)

                pre_softmax = pre_softmax+torch.tensor(math.pow(math.e, -1))
                pre_list_log = torch.log(pre_softmax)
                res = F.nll_loss(pre_list_log, tar_list)
                res.abs()
                loss_sum = loss_sum+res
    #loss_sum = torch.neg(loss_sum)
    return loss_sum
