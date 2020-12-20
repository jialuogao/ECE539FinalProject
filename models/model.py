import os
import sys
import torch
from torch.autograd import Variable
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.models as models
import cv2
import numpy as np
import scipy.stats as st
import math

def dog_n(img,size=(0,0),k=1.6,sigma=0.5,gamma=1):
        img1 = cv2.GaussianBlur(img,size,sigma)
        img2 = cv2.GaussianBlur(img,size,sigma*k)
        return (img1-gamma*img2)

def xdog(img,sigma=0.5,k=1.6, gamma=1,epsilon=1,phi=1):
    img = dog_n(img,sigma=sigma,k=k,gamma=gamma)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            #### 1+tanh(pi*relu(img[i,j]-epsilon))
            if(img[i,j] < epsilon):
                img[i,j] = 1
            else:
                img[i,j] = (1 + np.tanh(phi*(img[i,j])))
    return img

def xdog_thresh(img, sigma=0.5,k=1.6, gamma=1,epsilon=1,phi=1,alpha=1):
    img = xdog(img,sigma=sigma,k=k,gamma=gamma,epsilon=epsilon,phi=phi)
    #cv2.imshow("1",np.uint8(img))
    mean = np.mean(img)
    max = np.max(img)
    img = cv2.GaussianBlur(src=img,ksize=(0,0),sigmaX=sigma*3)
    #cv2.imshow("2",np.uint8(img))
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if(img[i,j] > mean):
                img[i,j] = max
    #cv2.imshow("3",np.uint8(img))
    return img/max

def norm(img):
    max = torch.max(img)
    min = torch.min(img)
    return (img-min) / np.float64(max-min)

class Xdog(nn.Module):
    def __init__(self):
        super(Xdog, self).__init__()
        self.sigma=0.8
        self.k=2.5
        self.gamma=0.98
        self.epsilon=0
        self.phi=1
        kernel=round(self.sigma*self.k*3*2 + 1)|1
        padding=int(kernel/2)
        stride=1
        g_kernel1 = self.gauss_kernel(kernel, self.sigma, 1).transpose((3, 2, 1, 0))
        g_kernel2 = self.gauss_kernel(kernel, self.sigma*self.k, 1).transpose((3, 2, 1, 0))
        g_kerneldog = (g_kernel1 - self.gamma * g_kernel2)
        gauss_conv_dog = nn.Conv2d(1, 1, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        gauss_conv_dog.weight.data.copy_(torch.from_numpy(g_kerneldog))
        gauss_conv_dog.weight.requires_grad = False
        gauss_conv_dog.cuda()
        self.gauss_conv_dog = gauss_conv_dog
        
        g_kernel3 = self.gauss_kernel(kernel, self.sigma*2, 1).transpose((3, 2, 1, 0))
        gauss_conv3 = nn.Conv2d(1, 1, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        gauss_conv3.weight.data.copy_(torch.from_numpy(g_kernel3))
        gauss_conv3.weight.requires_grad = False
        gauss_conv3.cuda()
        self.gauss_conv3 = gauss_conv3

        size = 3
        blur_kernel = np.ones([size, size],dtype=np.float64)
        blur_kernel = blur_kernel/blur_kernel.sum()
        blur = nn.Conv2d(1, 1, kernel_size=size, stride=stride, padding=int(size/2), bias=False)
        blur.weight.data.copy_(torch.from_numpy(blur_kernel))
        blur.weight.requires_grad = False
        blur.cuda()
        self.blur = blur

        avg_pool = nn.AvgPool2d(2)
        avg_pool.cuda()
        self.avg_pool = avg_pool

    '''
    def gauss_kernel2(self, kernlen=21, nsig=3, channels=1):
        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        out_filter = np.array(kernel, dtype=np.float64)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis=2)
        return out_filter
    '''
    def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
        gkern1d = cv2.getGaussianKernel(kernlen, nsig)
        gkern2d = np.outer(gkern1d, gkern1d)
        out_filter = np.array(gkern2d, dtype=np.float64)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis=2)
        return out_filter

    def forward(self, input,name):
        #cuda0 = torch.device('cuda:0')
        #stemp = torch.zeros(256,256, device=cuda0)
        gray = 0.299 * input[:,0,:,:] + 0.587 * input[:,1,:,:] + 0.114 * input[:,2,:,:]
        gray = gray.reshape(1,1,256,256)
        #temp = layer.detach().cpu()
        #after_xdog =xdog_thresh(temp.numpy(),sigma=0.5,k=1.6, gamma=0.98,epsilon=-0.1,phi=150,alpha=1)
        
        ###################
        # gaussian
        
        pool = self.avg_pool(gray)
    #    cv2.imshow("blur"+name,np.uint8(255*norm(self.blur(pool)).data.cpu().numpy()[0,0,:,:]))
        dog = self.gauss_conv_dog(self.blur(pool))
    #    cv2.imshow("dog"+name,np.uint8(255*norm(dog).data.cpu().numpy()[0,0,:,:]))
        #print(dog)
        #print(torch.max(gauss1).cpu().data.numpy())
        #print('dog: ',dog.shape)
        #epsilon = torch.mean(dog) * 1.1
        #print(torch.max(dog), '   ', torch.min(dog))
        #print(epsilon,'  ', self.epsilon)
        #xdog = 1 + torch.tanh(self.phi * (torch.tanh(2*(dog-self.epsilon))/2+0.5)*dog)
        xdog = 1 + torch.tanh(self.phi * dog)
        '''
        xdog2 = dog.data.cpu().numpy()[0,0,:,:]
        for i in range(0,xdog2.shape[0]):
            for j in range(0,xdog2.shape[1]):
                #### 1+tanh(pi*relu(img[i,j]-epsilon))
                if(xdog2[i,j] < self.epsilon):
                    xdog2[i,j] = 1
                else:
                    xdog2[i,j] = (1 + np.tanh(self.phi*(xdog2[i,j])))
        temp = (xdog2 - np.min(xdog2))/(np.max(xdog2)-np.min(xdog2))
        cv2.imshow("xdog2"+name,np.uint8(temp*255))
        '''
    #    cv2.imshow("xdog"+name,np.uint8(255*norm(xdog).data.cpu().numpy()[0,0,:,:]))
        #print(xdog.data.cpu().numpy())
        #print('xdog: ',xdog.shape)
    #    gauss3 = self.gauss_conv3(xdog)
    #    cv2.imshow("gauss3"+name,np.uint8(255*norm(gauss3).data.cpu().numpy()[0,0,:,:]))
    #    mean = torch.mean(gauss3)*0.95
    #    max = torch.max(gauss3)
    #    xdog_threshold = (torch.tanh(2*(gauss3-mean))/2+0.5)*max + (torch.tanh(2*(mean - gauss3))/2+0.5)*gauss3
        #min = torch.min(xdog_threshold)
        #print(max,"asdf",min)
        #print(((xdog_threshold-min)/(max-min)).cpu().data.numpy())
    #    cv2.imshow("xdog_threshold"+name,np.uint8(255*norm(xdog_threshold).data.cpu().numpy()[0,0,:,:]))
        return xdog
    








