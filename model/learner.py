import torch
import random
import pickle
import logging

import numpy as np

from torch import nn
from torch.nn import functional as F

logger = logging.getLogger("experiment")

def pickle_dict(dictionary, filename):
    p = pickle.Pickler(open("{0}.p".format(filename),"wb"))
    p.fast = True
    p.dump(dictionary)

def load(file): 
    openFile = open(file, "rb") 
    data = pickle.load(openFile)  
    openFile.close() 
    return(data) 

def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-5, momentum=0.1):
    ''' momentum = 1 restricts stats to the current mini-batch '''
    running_mean = torch.zeros(np.prod(np.array(input.data.size()[1])))
    running_var = torch.ones(np.prod(np.array(input.data.size()[1])))
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def maxpool(input, kernel_size, indices=False, stride=None):
    return F.max_pool2d(input, kernel_size, stride, return_indices=indices)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

class Learner(nn.Module):

    def __init__(self, config, bias=-8, channels=112, nm_channels=192, rep_size=112, nm_rep_size=1728, device='cuda', treatment='TSAR'):

        super(Learner, self).__init__()

        self.bias = bias
        self.device = device
        self.config = config
        self.channels = channels
        self.rep_size = rep_size
        self.treatment = treatment
        self.nm_rep_size = nm_rep_size
        self.nm_channels = nm_channels

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):

            if 'conv' in name:
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif 'linear' in name:

                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]

                if 'nm' in name:
                    bias = self.bias
                    bias_ = nn.Parameter(torch.zeros(param[0]))
                    bias_.data.fill_(bias)
                    self.vars.append(bias_)
                else:    
                    self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'cat':
                pass
            elif name is 'cat_start':
                pass
            elif name is "rep":
                pass

            elif 'instanceNorm' in name:
                self.vars.append(nn.InstanceNorm(param[0]), affine=True)
            elif 'bn' in name:
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''
        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'

            elif name is 'cat':
                tmp = 'cat'
                info += tmp + "\n"
            elif name is 'cat_start':
                tmp = 'cat_start'
                info += tmp + "\n"

            elif name is 'rep':
                tmp = 'rep'
                info += tmp + "\n"


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return(info)
    
    def ANML_Neuromodulate(self, vars):
    
        # Reinit running_mean/var for instance normalization: 
        running_mean = nn.Parameter(torch.zeros(self.nm_channels), 
                                    requires_grad=False).to(self.device)
        running_var = nn.Parameter(torch.ones(self.nm_channels), 
                                   requires_grad=False).to(self.device)

        # =========== C1 =========== 

        w,b = vars[0], vars[1]
        output = conv2d(self.data_, w, b)
        w,b = vars[2], vars[3]

        # training = True because we are doing instance normalization
        # resetting running mean and variance after each instance
        # so traces of prior inputs are removed

        output = F.batch_norm(output, running_mean, running_var, weight=w, bias=b, training=True)
        output = F.relu(output)
        output = maxpool(output, kernel_size=2, stride=2)

        # =========== C2 =========== 

        w,b = vars[4], vars[5]
        output = conv2d(output, w, b)
        w,b = vars[6], vars[7]
        output = F.batch_norm(output, running_mean, running_var, weight=w, bias=b, training=True)
        output = F.relu(output)
        output = maxpool(output, kernel_size=2, stride=2)

        # =========== C3 =========== 

        w,b = vars[8], vars[9]
        output = conv2d(output, w, b)
        w,b = vars[10], vars[11]
        output = F.batch_norm(output, running_mean, running_var, weight=w, bias=b, training=True)
        output = F.relu(output)

        output = output.view(self.data_.size(0),-1)

        w,b = vars[12], vars[13]
        fc_mask = F.sigmoid(F.linear(output, w, b)).view(self.data_.size(0),-1)

        return(fc_mask)


    def Neuromodulate(self, vars, bn_training):

        # Reinit running_mean/var for instance normalization:

        running_mean = nn.Parameter(torch.zeros(self.nm_channels), 
                                    requires_grad=False).to(self.device)
        running_var = nn.Parameter(torch.ones(self.nm_channels), 
                                   requires_grad=False).to(self.device)

        # =========== C1 =========== 

        w,b = vars[0], vars[1]
        output = conv2d(self.data_, w, b)
        w,b = vars[2], vars[3]
        output = F.batch_norm(output, running_mean, running_var, weight=w, bias=b, training=True)
        output = F.relu(output)
        output = maxpool(output, kernel_size=2, stride=2)

        # =========== C2 =========== 
 
        w,b = vars[4], vars[5]
        output = conv2d(output, w, b)
        w,b = vars[6], vars[7]
        output = F.batch_norm(output, running_mean, running_var, weight=w, bias=b, training=True)
        output = F.relu(output)
        output = maxpool(output, kernel_size=2, stride=2)

        # =========== C3 =========== 

        w,b = vars[8], vars[9]
        output = conv2d(output, w, b)
        w,b = vars[10], vars[11]
        output = F.batch_norm(output, running_mean, running_var, weight=w, bias=b, training=True)
        output = F.relu(output)
        output = output.view(1,-1)

        w,b = vars[12], vars[13]
        conv1_mask = F.sigmoid(F.linear(output, w, b)).view(self.channels,3,3,3)
        w,b = vars[14], vars[15]
        conv2_mask = F.sigmoid(F.linear(output, w, b)).view(self.channels,self.channels,3,3)
        w,b = vars[16], vars[17]
        conv3_mask = F.sigmoid(F.linear(output, w, b)).view(self.channels, self.channels, 3, 3)
        w,b = vars[18], vars[19]
        fc_mask = F.sigmoid(F.linear(output, w, b)).view(1000, self.channels)

        return(conv1_mask, conv2_mask, conv3_mask, fc_mask)

    def ANML_Predict(self, vars, mask):

        # Reinit running_mean/var for instance normalization:
        running_mean = nn.Parameter(torch.zeros(256), requires_grad=False).to(self.device)
        running_var = nn.Parameter(torch.ones(256), requires_grad=False).to(self.device)

        # =========== C1 ===========

        w,b = vars[14], vars[15]
        output = conv2d(self.data, w, b)
        w,b = vars[16], vars[17]
        # training = True for instance normalization
        output = F.batch_norm(output, running_mean, running_var, weight=w, bias=b, training=True)
        output = F.relu(output)
        output = maxpool(output, kernel_size=2, stride=2)

        # =========== C2 ===========

        w,b = vars[18], vars[19]
        output = conv2d(output, w, b, stride=1)
        w,b = vars[20], vars[21]
        # training = True for instance normalization
        output = F.batch_norm(output, running_mean, running_var, weight=w, bias=b, training=True)
        output = F.relu(output)
        output = maxpool(output, kernel_size=2, stride=2)

        # =========== C3 ===========

        w,b = vars[22], vars[23]
        output = conv2d(output, w, b, stride=1)
        w,b, = vars[24], vars[25]
        # training = True for instance normalization
        output = F.batch_norm(output, running_mean, running_var, weight=w, bias=b, training=True)
        output = F.relu(output)
        output = output.view(output.size(0), -1) 
        output = output*mask

        w,b = vars[26], vars[27]
        output = F.linear(output, w, b)

        return(output)

    def Predict(self, vars, bn_training, conv1_mask, conv2_mask, conv3_mask, fc_mask):

        channels = 112
        running_mean = nn.Parameter(torch.zeros(self.channels), 
                                    requires_grad=False).to(self.device)
        running_var = nn.Parameter(torch.ones(self.channels), 
                                   requires_grad=False).to(self.device)
        
        # =========== C1 =========== 

        w,b = vars[20], vars[21]
        w = w*conv1_mask
        output = conv2d(self.data, w, b)
        w,b = vars[22], vars[23]
        output = F.batch_norm(output, running_mean, running_var, weight=w, bias=b, training=True)
        output = F.relu(output)
        output, c1_indices = maxpool(output, kernel_size=2, stride=2, indices=True)

        # =========== C2 ===========

        w,b = vars[24], vars[25]
        w = w*conv2_mask 
        output = conv2d(output, w, b, stride=1)
        w,b = vars[26], vars[27]
        output = F.batch_norm(output, running_mean, running_var, weight=w, bias=b, training=True)
        output = F.relu(output)
        output, c2_indices = maxpool(output, kernel_size=2, stride=2, indices=True)

        # =========== C3 ===========

        w,b = vars[28], vars[29]
        w = w*conv3_mask
        output = conv2d(output, w, b, stride=1)
        w,b, = vars[30], vars[31]
        output = F.batch_norm(output, running_mean, running_var, weight=w, bias=b, training=True)
        output = F.relu(output)
        output, c3_indices = maxpool(output, kernel_size=2, stride=2, indices=True)
        output = output.view(1, self.channels)
         
        w,b = vars[32], vars[33]
        w = w*fc_mask
        output = F.linear(output, w, b)

        return(output)

    def forward(self, x, vars=None, bn_training=True, outer_loop=False, feature=False, analysis=False, layer_to_record='C3', ANML=False):

        cat_var = False
        cat_list = []

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
            
        for i in range(x.size(0)):
            
            self.data = x[i].view(1,3,28,28)
            self.data_ = x[i].view(1,3,28,28)        

            if self.treatment == 'ANML':
                fc_mask = self.ANML_Neuromodulate(vars)
                p = self.ANML_Predict(vars, fc_mask)
            elif self.treatment == 'TSAR':
                conv1_mask, conv2_mask, conv3_mask, fc_mask = \
                        self.Neuromodulate(vars, bn_training)
                p = self.Predict(vars, 
                                 bn_training, 
                                 conv1_mask, 
                                 conv2_mask, 
                                 conv3_mask, 
                                 fc_mask)

            if i > 0:
                predictions = torch.cat([predictions, p], dim=0)
            else:
                predictions = p

        if analysis:
            if layer_to_record == 'all_conv':

                first = conv1_mask.cpu().detach().numpy()
                second = conv2_mask.cpu().detach().numpy()
                third = conv3_mask.cpu().detach().numpy()
                return(predictions, first, second, third)

            else:
            
                if layer_to_record == 'C3':
                    mask_to_report = conv3_mask.cpu().detach().numpy()
                    layer_idx = 28
                elif layer_to_record == 'C2':
                    mask_to_report = conv2_mask.cpu().detach().numpy()
                    layer_idx = 24
                elif layer_to_record == 'C1':
                    mask_to_report = conv1_mask.cpu().detach().numpy()
                    layer_idx = 20
                elif layer_to_record == 'FC':
                    mask_to_report = fc_mask.cpu().detach().numpy()
                    layer_idx = 32

                return(predictions, 
                       mask_to_report*vars[layer_idx].cpu().detach().numpy(), 
                       vars[layer_idx].cpu().detach().numpy(), 
                       reg_rep.cpu().detach().numpy())

        else:

            return(predictions)

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
