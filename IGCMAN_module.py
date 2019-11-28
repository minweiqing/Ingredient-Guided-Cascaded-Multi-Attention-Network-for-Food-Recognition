import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as tensor
import torch

import torchvision

food101_multilable_class = 174

food101_category = 101  

'''
==============================================================================================
@Parameters:
    lstm_input_size  : number of expected features in the input x of LSTM
    lstm_hidden_size : number of features in the hidden state of LSTM
    zk_size          : size of z_k (about z_k, see 'Update rule of M' in the paper)
    num_itreations   : number of iterations in module (default: 5)
    vgg16_weight     : vgg16 pretrain model on dataset
    num_classes      : number of ingredient
    category_classe  : number of classes/categories
    use_gpu          : whether using gpu (default: True)
@Input:
    f_I : feature map (torch.cuda.FloatTensor[batch_size, num_channels, height, width])
@Output:
    fused_scores : final fused score vectors (torch.cuda.FloatTensor[batch_size, num_ingredient])
    M            : transformation matrices in ST
    score_category: final category score vectors (torch.cuda.FloatTensor[batch_size, num_category])
==============================================================================================
'''
class IGCMAN_module(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, zk_size, vgg16_weight,
                 num_iterations=5, num_classes=food101_multilable_class, category_classe=food101_category,
                 use_gpu=True):
        
        super(IGCMAN_module, self).__init__()

        self.K           = num_iterations 
        self.C           = num_classes
        self.category    = category_classe
        self.use_gpu     = use_gpu
        self.input_size  = lstm_input_size  #14*14
        self.hidden_size = lstm_hidden_size  #4096

        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.fc      = nn.Linear(lstm_input_size * lstm_input_size / 4 * 512, 4096)
        self.lstm = nn.LSTMCell(4096, lstm_hidden_size)
        
        self.get_zk = nn.Sequential(
            # channels of output feature map in vgg16 = 512
            nn.Linear(lstm_hidden_size, zk_size),     
            nn.ReLU(inplace=True)
        )
        self.get_score = nn.Linear(zk_size, num_classes)


        self.get_category_score = nn.Linear(4096, category_classe)
        self.update_m_category = nn.Linear(zk_size, 6)
        self.update_m_category.weight.data = torch.zeros(6, zk_size)
        self.update_m_category.bias.data   = tensor([1, 0., 0., 0., 1, 0.])



        self.update_m  = nn.Linear(zk_size, 6)
        self.update_m.weight.data = torch.zeros(6, zk_size)
        self.update_m.bias.data   = tensor([1, 0., 0., 0., 1, 0.])

        self.vgg_weight = vgg16_weight

        vgg16 = torchvision.models.vgg16(pretrained=False)
        vgg16.classifier = nn.Sequential(nn.Linear(25088, 4096),      #vgg16
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5))
                                 # nn.Linear(4096, 174))



        print('\n*****  Model load the vgg16 multilabel finetune weight*****')
        model_dict = vgg16.state_dict()
        pretrained_dict = torch.load(self.vgg_weight)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
        vgg16.load_state_dict(model_dict)
        self.vgg = vgg16
        # self.extract_features =  vgg16.features[:29] #is the conv5_3   vgg conv5_3 feature
       



    # ST: spatial transformer network forward function
    # ================================================
    def ST(self, x, theta):
        # determine the output size of STN
        num_channels = x.size()[1]
        batch_size   = x.size()[0]
        output_size =  torch.Size((batch_size, num_channels, self.input_size, self.input_size))
        
        grid = F.affine_grid(theta, output_size) 
        if self.use_gpu:
            grid = grid.cuda()
        # use bilinear interpolation(default) to sample the input pixels
        x = F.grid_sample(x, grid)
        return x
    
    # init_hidden: initialize the (h0, c0) in LSTM
    # ============================================
    def init_hidden(self, N):
        if self.use_gpu:
            h0 = torch.zeros(N, self.hidden_size).cuda()
            c0 = torch.zeros(N, self.hidden_size).cuda()
        else:
            h0 = torch.zeros(N, self.hidden_size)
            c0 = torch.zeros(N, self.hidden_size)
        return (h0, c0)
    
    #  moudule forward function
    # ============================
    def forward(self,input, return_whole_scores=False):
        # initialization
        batch_size = input.size()[0]
        if self.use_gpu:
            # scores = torch.randn(self.K, batch_size, self.C).cuda()
            scores = torch.randn(batch_size, self.K, self.C).cuda()
        else:
            # scores = torch.randn(self.K, batch_size, self.C)
            scores = torch.randn(batch_size, self.K, self.C)
        # M      = torch.randn(self.K+1, batch_size, 2, 3) 
        M      = torch.randn(batch_size, self.K+1, 2, 3).cuda() # if use the multi-gpu must be the cuda 





   
        hidden = self.init_hidden(batch_size)
        f_vgg = self.vgg.features[:29](input)  # vgg conv5_3
        output_vgg = self.vgg(input)  #vgg last fully connect
        # M[0] = self.update_m_category(output_vgg).view(batch_size, 2, 3)
        # M[0, :, 0, 1] = tensor(0.)
        # M[0, :, 1, 0] = tensor(0.)
        M[:, 0] = self.update_m_category(output_vgg).view(batch_size, 2, 3)
        M[:, 0, 0, 1] = tensor(0.)
        M[:, 0, 1, 0] = tensor(0.)



        # M[0]   = tensor([[1., 0., 0.], [0., 1., 0.]])
        # f_k_category = self.ST(f_vgg, M[0])
        f_k_category = self.ST(f_vgg, M[:, 0])
        # descend dimension for lower GPU memory requirement
        f_k = self.pooling(f_k_category)
        f_k = self.fc(f_k.view(batch_size, -1))
        hidden = self.lstm(f_k, hidden)  #hidden (two param h and c)
        z_k = self.get_zk(hidden[0])
        score_category = self.get_category_score(z_k)

        M[:, 1] = self.update_m(z_k).view(batch_size, 2, 3)
        M[:, 1, 0, 1] = tensor(0.)
        M[:, 1, 1, 0] = tensor(0.)

        f_I = f_k_category

        # for each iteration
        for k in range(1, self.K+1):
            # locate an attentional region
            # f_k = self.ST(f_I, M[k])
            f_k = self.ST(f_I, M[:, k])
            
            # descend dimension for lower GPU memory requirement
            f_k = self.pooling(f_k)
            f_k = self.fc(f_k.view(batch_size, -1))

            # predict the scores regarding this region
            hidden = self.lstm(f_k, hidden)  #hidden (two param h and c)
            
            # get z_k for further caculating M and scores
            z_k = self.get_zk(hidden[0])

            
                # obtain the score vector of current iteration
            # scores[k-1] = self.get_score(z_k)
            scores[:, k-1] = self.get_score(z_k)
                
            if k != self.K:
                # update transformation matrix for next iteration

                M[:, k+1] = self.update_m(z_k).view(batch_size, 2, 3)
                M[:, k+1, 0, 1] = tensor(0.)
                M[:, k+1, 1, 0] = tensor(0.)
        
        # max pooling to obtain the final fused scores
        fused_scores = scores.max(1)  # reture the max of lie and index

        
        if return_whole_scores:
            return fused_scores[0], M[1:, :, :, :], scores
        else:
            return fused_scores[0], M, score_category

