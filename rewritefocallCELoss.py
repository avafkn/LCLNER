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
                pre_softmax = F.softmax(pre_list, -1)
                #print(type(pre_softmax), pre_softmax.shape)
                tar_list = tar_list.view(-1, 1)
                pre_softmax = pre_softmax.gather(1, tar_list).view(-1)
                pre_softmax_1 = pre_softmax + \
                    torch.tensor(math.pow(math.e, -0.1))
                pre_list_log = torch.log(pre_softmax_1)
                loss = -0.5 * \
                    torch.pow(torch.sub(1.0, pre_softmax), 1)*pre_list_log
                loss = loss.mean()
                res = loss.abs()

                #res = F.nll_loss(pre_list_log, tar_list)

                res = res.abs()
                loss_sum = loss_sum+res
    #loss_sum = torch.neg(loss_sum)

    return loss_sum
