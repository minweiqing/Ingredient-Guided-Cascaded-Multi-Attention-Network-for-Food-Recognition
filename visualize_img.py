# -*-coding:utf-8-*-

import torch.nn as nn 
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from IGCMAN_module import IGCMAN_module

import shutil
import argparse
import os 
import PIL
import torch
import time
import numpy

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

BATCH_SIZE    = 16           # number of images for each epoch
N             = 224          # size of input images 
DIR_TRAIN_IMAGES   = './food101/origal_data/train_full.txt'
DIR_TEST_IMAGES    = './food101/origal_data/test_full.txt'
vgg_multilabel_finetune='./multilabel_model/food101_vgg.pth'

# ==================================================================
# Parser Initialization
# ==================================================================
parser = argparse.ArgumentParser(description='Pytorch Implementation')
parser.add_argument('--trainBatchSize',  default=BATCH_SIZE,        type=int,   help='training batch size')
parser.add_argument('--testBatchSize',   default=BATCH_SIZE,        type=int,   help='testing batch size')
parser.add_argument('--vgg_weight',     default=vgg_multilabel_finetune,        type=str,  help='inint vgg weight path')
args = parser.parse_args()
print('***** Prepare Data ******')
# transforms of training dataset 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
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

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')
 
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, txt_dir, transform=None, target_transform=None, loader=My_loader):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = My_loader
 
    def __len__(self):
        
        return len(self.imgs)
  
    def __getitem__(self, index):
        img_name, label = self.imgs[index]  
        img = self.loader(os.path.join('./ACMM_method/visualize_img_temp',img_name))
        # print img
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_name
        # if the label is the single-label it can be the int 
        # if the multilabel can be the list to torch.tensor

train_dataset = MyDataset(txt_dir=DIR_TRAIN_IMAGES , transform=train_transforms)
test_dataset = MyDataset(txt_dir=DIR_TEST_IMAGES , transform=test_transforms)
train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.trainBatchSize, shuffle=False,  num_workers=2)
test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=args.testBatchSize,  shuffle=False, num_workers=2)
print('Data Preparation : Finished')


# ==================================================================
# Prepare Model
# ==================================================================

print('\n*****  Model load the  model weight*****')
IGCMAN = IGCMAN_module(lstm_input_size=14, lstm_hidden_size=4096, zk_size=4096, vgg16_weight=args.vgg_weight)

state_dict = torch.load('./checkpoint/acmmm_multilable_model_20190127_1e-5.pth')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
IGCMAN.load_state_dict(new_state_dict)


print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
print('cuda: move all model parameters and buffers to the GPU')

IGCMAN.cuda()
cudnn.benchmark = True

print('Model Preparation : Finished')
def ST( x, theta):
    # determine the output size of STN
    num_channels = x.size()[1]
    batch_size   = x.size()[0]
    output_size =  torch.Size((batch_size, num_channels, 256, 256))
    
    grid = F.affine_grid(theta, output_size) 
    
    grid = grid.cuda()
    # use bilinear interpolation(default) to sample the input pixels
    x = F.grid_sample(x, grid)
    return x
def imshow(inp):
    # print inp.size()
    inp = inp.detach().numpy().transpose((1 , 2 , 0))  # conve the chw to hwc
    mean = np.array([0.485 , 0.456 , 0.406])
    std = np.array([0.229 , 0.224 , 0.225])
    inp = std * inp + mean
    inp = np.clip(inp , 0 , 1) #Clip (limit) the values in an array
    inp = inp * 255
    return PIL.Image.fromarray(inp.astype('uint8'), 'RGB')


imgId = 0

IGCMAN.eval()
for i, (input, target, img_name) in enumerate(train_loader):
    # print img_name
    target = target.cuda()
    input = input.cuda()
    # print input_var.size()[2]
    input_var = torch.autograd.Variable(input, requires_grad=False)
    target_var = torch.autograd.Variable(target, requires_grad=False)# not compute the grad
    # f_I = extract_features(input_var)    
    output, M, score_category = IGCMAN(input)
    M = M.permute(1, 0, 2, 3) # Trans the dim

    for k in range(1,6):
        STN_output = ST(input_var, M[k])

        print('***************** start the batch ********************************')
        for j in range(STN_output.size()[0]):
            class_name = img_name[j].split('/')[0]
            images_name = img_name[j].split('/')[1]
            # print images_name
            # print j
            if not os.path.isdir(os.path.join('./visualize_img_temp_patch5/', class_name)):
                os.mkdir(os.path.join('./visualize_img_temp_patch5/', class_name))

            im = imshow(STN_output.cpu().data[j])
 
            if k==0:
                save_path = './visualize_img_temp_patch5/' + class_name + '/' + images_name
                print save_path
           
                im.save('./visualize_img_temp_patch5/' + class_name + '/' + images_name)
           
                print ('inter:', i)
                print ('the M numble:', k)
                print ('the batch:', j)

            else:
                save_path = './visualize_img_temp_patch5/' + class_name + '/' + images_name[:-4] +'_'+ str(k) +'.jpg'
                print save_path
           
                im.save('./visualize_img_temp_patch5/' + class_name + '/' + images_name[:-4] +'_'+ str(k) +'.jpg')
           
                print ('inter:', i)
                print ('the M numble:', k)
                print ('the batch:', j)
            


# for i, (input, target, img_name) in enumerate(test_loader):
#     # print img_name
#     target = target.cuda()
#     input = input.cuda()
#     # print input_var.size()[2]
#     input_var = torch.autograd.Variable(input, requires_grad=False)
#     target_var = torch.autograd.Variable(target, requires_grad=False)# not compute the grad
#     # f_I = extract_features(input_var)    
#     output, M, score_category = IGCMAN(input)
#     M = M.permute(1, 0, 2, 3) # Trans the dim

#     for k in range(1,6):
#         STN_output = ST(input_var, M[k])

#         print('***************** start the batch ********************************')
#         for j in range(STN_output.size()[0]):
#             class_name = img_name[j].split('/')[0]
#             images_name = img_name[j].split('/')[1]
           
#             if not os.path.isdir(os.path.join('./visualize_img_temp_patch5/', class_name)):
#                 os.mkdir(os.path.join('./visualize_img_temp_patch5/', class_name))

#             im = imshow(STN_output.cpu().data[j])

#             if k==0:
#                 save_path = './visualize_img_temp_patch5/' + class_name + '/' + images_name
#                 print save_path
           
#                 im.save('./visualize_img_temp_patch5/' + class_name + '/' + images_name)
           
#                 print ('inter:', i)
#                 print ('the M numble:', k)
#                 print ('the batch:', j)
                
#             else:
#                 save_path = './visualize_img_temp_patch5/' + class_name + '/' + images_name[:-4] +'_'+ str(k) +'.jpg'
#                 print save_path
           
#                 im.save('./visualize_img_temp_patch5/' + class_name + '/' + images_name[:-4] +'_'+ str(k) +'.jpg')
           
#                 print ('inter:', i)
#                 print ('the M numble:', k)
#                 print ('the batch:', j)
            

