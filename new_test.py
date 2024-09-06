import torch
import torchvision

import torch.optim
import os
from model.UIERetinex import UIEnet

from PIL import Image
import glob

import torchvision.transforms as transforms
from torch.autograd import Variable
import conf


is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    img_width, img_height, channels = 256, 256, 3
    transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC), transforms.ToTensor(),]   
    transform = transforms.Compose(transforms_)
    data_lowlight = transform(Image.open(image_path))
    data_lowlight = Variable(data_lowlight).type(Tensor).unsqueeze(0)
    print(data_lowlight.shape)
    Umoblie = UIEnet().to(conf.device)
    Umoblie.load_state_dict(torch.load('epochs/model_epoch.pth'))    
    j_out, t_out, a_out = Umoblie(data_lowlight)
    image_path = image_path.replace('test_data', 'asd') 
    result_path = image_path
    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
    torchvision.utils.save_image(j_out, result_path, normalize=True)

if __name__ == '__main__':
    # test_images
    print(torch.cuda.is_available())
    with torch.no_grad():
        filePath = 'data/test_data/'
        file_list = os.listdir(filePath)
        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*")
            for image in test_list:
                print(image)
                lowlight(image)




