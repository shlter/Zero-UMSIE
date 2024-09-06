import numpy
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
from model.RRDNet import RRDNet
from model.UIERetinex import UIEnet
import numpy as np
from PIL import Image
import glob
import time
import torchvision.transforms as transforms
from torch.autograd import Variable
import conf
from loss.loss_functions import normalize01


is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 2
    img_width, img_height, channels = 256, 256, 3
    transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC), transforms.ToTensor(),]    # Image.BICUBIC
    transform = transforms.Compose(transforms_)
    data_lowlight = transform(Image.open(image_path))
    data_lowlight = Variable(data_lowlight).type(Tensor).unsqueeze(0)
    # 1
    print(data_lowlight.shape)
    # Umoblie = RRDNet().to(conf.device)
    Umoblie = UIEnet().to(conf.device)
    Umoblie.load_state_dict(torch.load('epochs/model_epoch_11.pth'))    # 10
    # Umoblie.load_state_dict(torch.load('pths/ss/model_epoch_99.pth'))
    # Umoblie.load_state_dict(torch.load('pths/1600TV_Retinex_0.5color_MSE/model_epoch_10.pth'))
    # Umoblie.load_state_dict(torch.load('pths/new_UIERtinex/model_epoch_11.pth'))

    start = time.time()
    j_out, t_out, a_out = Umoblie(data_lowlight)
    # a_out = get_A(data_lowlight).to(conf.device)
    end_time = (time.time() - start)
    print(end_time)
    image_path = image_path.replace('test_data', 'asd')  # ('test_data', 'asd') ('RUIE', 'RUIE_result_t') ('test_data', 'asd_j')
    result_path = image_path
    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
    torchvision.utils.save_image(j_out, result_path, normalize=True)

if __name__ == '__main__':
    # test_images
    print(torch.cuda.is_available())
    with torch.no_grad():
        filePath = 'data/test_data/' #data/test_data/  RUIE
        file_list = os.listdir(filePath)
        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*")
            for image in test_list:
                print(image)
                lowlight(image)




