import torch.nn as nn
import torch.tensor as tensor
import torch.nn.functional as F
import torch
from math import *

# hyperparameters
alpha1   = tensor(0.9).cuda()
alpha   = tensor(0.5).cuda()
beta    = tensor(0.1).cuda()
beta1    = tensor(0.6).cuda()
# alpha   = 0.5
# beta    = 0.1
lambda1 = 0.01
lambda2 = 0.5
gama    = 0.1


def getAnchorPoints(num_points):
    radius = 0.5 * sqrt(2)
    # difference between two anchor points
    diff = 2 * pi / num_points
    cx = [radius * cos(i * diff) for i in range(0, num_points)]
    cy = [radius * sin(i * diff) for i in range(0, num_points)]

    return tensor(cx).view(num_points, -1), tensor(cy).view(num_points, -1)


'''
Loss Function 
=======================================================================================
@Args:
    input  : ingredient score vectors         (torch.cuda.FloatTensor[batch_size, num_ingredient])
    target : ingredient label target          (torch.cuda.FloatTensor[batch_size, num_ingredient])
    M      : transformation matrix            (torch.FloatTensor[num_iterations, batch_size, 2, 3])
    category_out :  category score vectors    (torch.cuda.FloatTensor[batch_size, num_category])
    category_target : category label target   (torch.cuda.FloatTensor[batch_size, num_category])

@Returns:
    total_loss
=======================================================================================
'''
def loss_function(input, target, M, category_out, category_target, add_constraint=False):
    '''
    [variable] 'pp'            : predicted probability vector    
    [variable] 'gtp'           : ground-truth probability vector
    [variable] 'loss_cls'      : loss for ingredient classification
    [variable] 'loss_category' : loss for category classification
    [variable] 'loss_loc' : loss for localizatoin
    '''
    
    # extra arguments from theta(that is transformation matrix)
    # =========================================================
    M = M.permute(1, 0, 2, 3) # Trans the dim
    # print M.size()
    sx = M[1:, :, 0, 0]
    sy = M[1:, :, 1, 1]
    tx = M[1:, :, 0, 2]
    ty = M[1:, :, 1, 2]

    # sx = M[:, 1:, 0, 0]
    # sy = M[:, 1:, 1, 1]
    # tx = M[:, 1:, 0, 2]
    # ty = M[:, 1:, 1, 2]
     
    # anchor point
    # ============
    cx = tensor([0., 0.4,  0.4, -0.4, -0.4]).view(5, -1).cuda()
    cy = tensor([0., 0.4, -0.4,  0.4, -0.4]).view(5, -1).cuda()
    #cx, cy = getAnchorPoints(M.size(0) - 2)

    # calculate the predicted & ground-truth iprobability vector
    # ==========================================================
    pp  = F.softmax(input, dim=1)
    gtp = target.div(target.norm(p=1, dim=1).view(input.size()[0], -1))
    
    # calculate loss for classification
    # =================================
    loss_cls = F.mse_loss(pp, gtp, size_average=False)

    criterion = nn.CrossEntropyLoss().cuda()

    loss_category = criterion(category_out, category_target)
    
    if not add_constraint:
        return loss_cls

    # calculate loss for localization
    # ===============================
    # anchor constraint
    loss_A = torch.sum(0.5 * ((tx - cx)**2 + (ty - cy)**2))
    
    # scale constraint
    loss_sx = torch.sum(torch.max(abs(sx) - alpha, tensor(0.).cuda()) ** 2)
    loss_sy = torch.sum(torch.max(abs(sy) - alpha, tensor(0.).cuda()) ** 2)
    loss_partx = torch.sum(torch.max(abs(M[0, :, 0, 0]) - alpha1, tensor(0.).cuda()) ** 2)
    loss_party = torch.sum(torch.max(abs(M[0, :, 1, 1]) - alpha1, tensor(0.).cuda()) ** 2)
    loss_S  = loss_sx + loss_sy +loss_partx +loss_party

    # positive constraint
    loss_P = torch.sum(torch.max(beta - sx, tensor(0.).cuda()) + torch.max(beta - sy, tensor(0.).cuda()))
    loss_pa = torch.sum(torch.max(beta1 - M[0, :, 0, 0], tensor(0.).cuda()) + torch.max(beta1 - M[0, :, 1, 1], tensor(0.).cuda()))

    loss_loc = (lambda2 *loss_S + lambda1 * loss_A + lambda2 * loss_P + lambda2 * loss_pa).cuda()
    
    # calculate total loss
    # ====================
    total_loss = loss_cls + gama * loss_loc + loss_category
    
    
    return total_loss

