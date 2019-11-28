import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import visdom
import torch.nn.functional as F

import argparse
import os
import PIL
import torch

from IGCMAN_module import IGCMAN_module
from loss import loss_function

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# ==================================================================
# Constants
# ==================================================================
EPOCH         = 80            # number of times for each run-through
BATCH_SIZE    = 16            # number of images for each epoch
LEARNING_RATE = 1e-5         # default learning rate 
WEIGHT_DECAY  = 0             # default weight decay
N             = 224          # size of input images 
MOMENTUM      = (0.9, 0.999)  # momentum in Adam optimization

GPU_IN_USE    = torch.cuda.is_available()  # whether using GPU
DIR_TRAIN_IMAGES_INGREDIENT   = './train_ingredient_174.txt'  # ingredient list
DIR_TEST_IMAGES_INGREDIENT    = './test_ingredient_174.txt'
DIR_TRAIN_IMAGES   = './train_full.txt'            # category list
DIR_TEST_IMAGES    = './test_full.txt'
IMAGE_PATH = './food101/origal_data/images'   # the path of images folder

vgg_multilabel_finetune='./model/food101_vgg.pth'
PATH_MODEL_PARAMS  = './model/acmmm_model_food101.pth'    #
NUM_INGREDIENT     = 174     
LOSS_OUTPUT_INTERVAL = 100


# ==================================================================
# Parser Initialization
# ==================================================================
parser = argparse.ArgumentParser(description='Pytorch Implementation of Food recognition')
parser.add_argument('--lr',              default=LEARNING_RATE,     type=float, help='learning rate')
parser.add_argument('--epoch',           default=EPOCH,             type=int,   help='number of epochs')
parser.add_argument('--trainBatchSize',  default=BATCH_SIZE,        type=int,   help='training batch size')
parser.add_argument('--testBatchSize',   default=BATCH_SIZE,        type=int,   help='testing batch size')
parser.add_argument('--weightDecay',     default=WEIGHT_DECAY,      type=float, help='weight decay')
parser.add_argument('--pathModelParams', default=PATH_MODEL_PARAMS, type=str,   help='path of model parameters')
parser.add_argument('--saveModel',       default=True,              type=bool,  help='save model parameters')
parser.add_argument('--loadModel',       default=False,             type=bool,  help='load model parameters')
parser.add_argument('--vgg_weight', default=vgg_multilabel_finetune, type=str,   help='path of model parameters')

args = parser.parse_args()


# ==================================================================
# Prepare Dataset(training & test)
# ==================================================================
print('***** Prepare Data ******')

# transforms of training dataset 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
                     transforms.RandomHorizontalFlip(p=0.5), # default value is 0.5
                     transforms.Resize((N, N)),
                     transforms.ToTensor(),
                     normalize
                  ])

# transforms of test dataset
test_transforms = transforms.Compose([
                    transforms.Resize((N, N)), 
                    transforms.ToTensor(),
                    normalize
                  ]) 

# root = './data/fashion'
 
def My_loader(path):
    return PIL.Image.open(path).convert('RGB')
 
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, txt_dir_ingredient, txt_dir_category, transform=None, target_transform=None, loader=My_loader):
        data_txt = open(txt_dir_ingredient, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split()
            imgs.append((words[0], words[1:]))
        data_txt = open(txt_dir_category, 'r')
        category = []
        for line in data_txt:
            line = line.strip()
            words = line.split()
            category.append(int(words[1]))
        self.category = category
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = My_loader
 
    def __len__(self):
        
        return len(self.imgs)
  
    def __getitem__(self, index):
        img_name, ingredient_label = self.imgs[index]
        ingredient_label = list(map(int, ingredient_label))
        category_label = self.category[index]

        # print label
        img = self.loader(os.path.join(IMAGE_PATH,img_name))
        if self.transform is not None:
            img = self.transform(img)
            ingredient_label =torch.Tensor(ingredient_label)
            # print label
        return img, ingredient_label, category_label


train_dataset = MyDataset(txt_dir_ingredient=DIR_TRAIN_IMAGES_INGREDIENT, txt_dir_category=DIR_TRAIN_IMAGES,transform=train_transforms)
test_dataset = MyDataset(txt_dir_ingredient=DIR_TEST_IMAGES_INGREDIENT, txt_dir_category=DIR_TEST_IMAGES,transform=test_transforms)
train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=BATCH_SIZE,  shuffle=False, num_workers=2)
print('Data Preparation : Finished')


# ==================================================================
# Prepare Model
# ==================================================================
print('\n***** Prepare Model *****')

IGCMAN = IGCMAN_module(lstm_input_size=14, lstm_hidden_size=4096, zk_size=4096, vgg16_weight=args.vgg_weight)

IGCMAN = torch.nn.DataParallel(IGCMAN)

# IGCMAN.load_state_dict(torch.load('./checkpoint/acmmm_multilable_model_20190210_1e-5.pth'))
# print IGCMAN
print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
print('cuda: move all model parameters and buffers to the GPU')

IGCMAN.cuda()
cudnn.benchmark = True


score_params = list(map(id, IGCMAN.module.get_score.parameters()))
category_params = list(map(id, IGCMAN.module.get_category_score.parameters()))
base_params = filter(lambda p: id(p) not in score_params + category_params,
                     IGCMAN.module.parameters())
optimizer = optim.Adam([
    {'params': base_params},
    {'params': IGCMAN.module.get_category_score.parameters(), 'lr': args.lr*10},
    {'params':IGCMAN.module.get_score.parameters(), 'lr': args.lr*10}], lr=args.lr, weight_decay=args.weightDecay, betas=MOMENTUM)

print('Model Preparation : Finished')


# Train
# ================================================================================
# data:        [torch.cuda.FloatTensor of size [batch_size, 3, N, N] ]
# target:      [torch.cuda.FloatTensor of size [batch_size, num_ingredient]]
# category:    [torch.cuda.FloatTensor of size [batch_size, num_categories]]
# output:      [torch.cuda.FloatTensor of size [batch_size, num_categories]]
# ================================================================================
def train():
    print('train:')
    
    IGCMAN.train()     # set the module in training  mode
    train_loss = 0. # sum of train loss up to current batch
    train_correct = 0
    total = 0

    
    sum_prediction_label         = torch.zeros(1, NUM_INGREDIENT) + 1e-6
    sum_correct_prediction_label = torch.zeros(1, NUM_INGREDIENT)
    sum_ground_truth_label       = torch.zeros(1, NUM_INGREDIENT)
    
    for batch_num, (data, target, category) in enumerate(train_loader):
        if target.sum() == 0:
            continue
        target = target.index_select(0, torch.nonzero(target.sum(dim=1)).view(-1))
        data   = data.index_select(0, torch.nonzero(target.sum(dim=1)).view(-1))
        
        if GPU_IN_USE:
            data, target = data.cuda(), target.cuda() 
            category = category.cuda()
        data = torch.autograd.Variable(data)
        target = torch.autograd.Variable(target)
        category = torch.autograd.Variable(category)

        # -----forward-----
        optimizer.zero_grad()      
        output, M, score_category = IGCMAN(data)
        # ---end forward---
        
        # ---calculate loss and backward---
        loss = loss_function(output, target, M, score_category, category, add_constraint=True)
        loss.backward()
        optimizer.step()
        # ----------end backward-----------
        
        train_loss   += loss

        prediction    = torch.topk(F.softmax(output, dim=1), 10, dim=1) # return the max value and the index tuple
        filter        = prediction[0].eq(0.1) + prediction[0].gt(0.1)
        prediction_index         = torch.mul(prediction[1]+1, filter.type(torch.cuda.LongTensor))
        extend_eye_mat           = torch.cat((torch.zeros(1, NUM_INGREDIENT), torch.eye(NUM_INGREDIENT)), 0)
        prediction_label         = extend_eye_mat[prediction_index.view(-1)].view(-1, 10, NUM_INGREDIENT).sum(dim=1)
        correct_prediction_label = (target.cpu().byte() & prediction_label.byte()).type(torch.FloatTensor)
        
        #count the sum of label vector
        sum_prediction_label         += prediction_label.sum(dim=0)
        sum_correct_prediction_label += correct_prediction_label.sum(dim=0)
        sum_ground_truth_label       += target.cpu().sum(dim=0)
        
        # # calculate  accuracy
        _, train_predict = torch.max(score_category, 1)
        total += BATCH_SIZE
        train_correct += torch.sum(train_predict == category.data)
        train_acc = float(train_correct) / total

        
        if batch_num % LOSS_OUTPUT_INTERVAL == 0:
   
            print('train loss %.3f (batch %d) accuracy %0.3f' % (train_loss/(batch_num+1), batch_num+1), train_acc)
        

    # evaluation metrics
    o_p = torch.div(sum_correct_prediction_label.sum(), sum_prediction_label.sum())
    o_r = torch.div(sum_correct_prediction_label.sum(), sum_ground_truth_label.sum())
    of1 = torch.div(2 * o_p * o_r, o_p + o_r)
    c_p = (torch.div(sum_correct_prediction_label, sum_prediction_label)).sum() / NUM_INGREDIENT
    c_r = (torch.div(sum_correct_prediction_label, sum_ground_truth_label)).sum() / NUM_INGREDIENT
    cf1 = torch.div(2 * c_p * c_r, c_p + c_r)
   
    return c_p, c_r, cf1, o_p, o_r, of1



# ================================================================================

# Test

# ================================================================================
def test():
    print('test:')
    IGCMAN.eval()        # set the module in evaluation mode
    test_loss    = 0. # sum of train loss up to current batch
    test_correct = 0
    total = 0
    
    sum_prediction_label         = torch.zeros(1, NUM_INGREDIENT) + 1e-6
    sum_correct_prediction_label = torch.zeros(1, NUM_INGREDIENT)
    sum_ground_truth_label       = torch.zeros(1, NUM_INGREDIENT)

    for batch_num, (data, target, category) in enumerate(test_loader):
        if target.sum() == 0:
            continue
        target = target.index_select(0, torch.nonzero(target.sum(dim=1)).view(-1))
        data   = data.index_select(0, torch.nonzero(target.sum(dim=1)).view(-1))

        if GPU_IN_USE:
            data, target = data.cuda(), target.cuda()  # set up GPU Tensor
            category = category.cuda()

        # f_I = extract_features(data)        
        output, M, score_category= IGCMAN(data)
        loss = loss_function(output, target, M, score_category, category, add_constraint=True)
        
        test_loss    += loss
        prediction    = torch.topk(F.softmax(output, dim=1), 10, dim=1) 
        filter        = prediction[0].eq(0.1) + prediction[0].gt(0.1)
        prediction_index         = torch.mul(prediction[1]+1, filter.type(torch.cuda.LongTensor))
        extend_eye_mat           = torch.cat((torch.zeros(1, NUM_INGREDIENT), torch.eye(NUM_INGREDIENT)), 0)
        prediction_label         = extend_eye_mat[prediction_index.view(-1)].view(-1, 10, NUM_INGREDIENT).sum(dim=1)
        correct_prediction_label = (target.cpu().byte() & prediction_label.byte()).type(torch.FloatTensor)
        
        #count the sum of label vector
        sum_prediction_label         += prediction_label.sum(dim=0)
        sum_correct_prediction_label += correct_prediction_label.sum(dim=0)
        sum_ground_truth_label       += target.cpu().sum(dim=0)
        # # calculate  accuracy
        _, test_predict = torch.max(score_category, 1)
        total += BATCH_SIZE
        test_correct += torch.sum(test_predict == category.data)
        test_acc = float(test_correct) / total


        if batch_num % LOSS_OUTPUT_INTERVAL == 0:
            print('Test loss %.3f (batch %d) accuracy %0.3f' % (test_loss / (batch_num+1), batch_num+1, test_acc))
           

    # evaluation metrics
    o_p = torch.div(sum_correct_prediction_label.sum(), sum_prediction_label.sum())
    o_r = torch.div(sum_correct_prediction_label.sum(), sum_ground_truth_label.sum())
    of1 = torch.div(2 * o_p * o_r, o_p + o_r)
    c_p = (torch.div(sum_correct_prediction_label, sum_prediction_label)).sum() / NUM_INGREDIENT
    c_r = (torch.div(sum_correct_prediction_label, sum_ground_truth_label)).sum() / NUM_INGREDIENT
    cf1 = torch.div(2 * c_p * c_r, c_p + c_r)
   
    return c_p, c_r, cf1, o_p, o_r, of1


# ==================================================================
# Save Model
# ==================================================================
def save():
    torch.save(IGCMAN.state_dict(), args.pathModelParams)
    print('Checkpoint saved to {}'.format(args.pathModelParams))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    param_groups = optimizer.state_dict()['param_groups']
    param_groups[0]['lr']=lr
    param_groups[1]['lr']=lr*10
    param_groups[2]['lr']=lr*10
    # for param_group in param_groups:
    #     param_group['lr'] = lr



# ==================================================================
# Main Loop
# ==================================================================
for current_epoch in range(1, args.epoch + 1):
    adjust_learning_rate(optimizer, current_epoch)
    print('\n===> epoch: %d/%d' % (current_epoch, args.epoch))
    train_cp, train_cr, train_cf1, train_op, train_or, train_of1 = train()
    with torch.no_grad():
        test_cp, test_cr, test_cf1, test_op, test_or, test_of1 = test()
    
    print(

'''===> epoch: %d/%d<br/>
-------------------------------------------------------------
|    CP   |    CR   |   CF1   |    OP   |    OR   |   OF1   |
-------------------------------------------------------------
|  %.3f  |  %.3f  |  %.3f  |  %.3f  |  %.3f  |  %.3f  |
-------------------------------------------------------------
'''
 % (current_epoch, args.epoch, test_cp, test_cr, test_cf1, test_op, test_or, test_of1))
    
    if test_of1 > of1 and test_cf1 > cf1:
        if args.saveModel:
            save()
        of1 = test_of1
        cf1 = test_cf1

    if current_epoch == args.epoch:
        print('===> BEST PERFORMANCE (OF1/CF1): %.3f / %.3f' % (of1, cf1))

