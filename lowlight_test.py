
import torch
import torchvision

import torch.optim
import os

from model.RRDNet import RRDNet

from PIL import Image
import glob
import time
import torchvision.transforms as transforms
from torch.autograd import Variable

is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 2
    img_width, img_height, channels = 256, 256, 3
    transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC), #Image.BICUBIC
                   transforms.ToTensor(),]
    transform = transforms.Compose(transforms_)
    data_lowlight = transform(Image.open(image_path))
    data_lowlight = Variable(data_lowlight).type(Tensor).unsqueeze(0)
    # 1
    print(data_lowlight.shape)
    Umoblie = RRDNet().cuda()
    Umoblie.load_state_dict(torch.load('epochs/model_epoch_4.pth'))
    # Umoblie.load_state_dict(torch.load('pths/ss/model_epoch_99.pth'))
    start = time.time()
    illumination, reflectance, noise = Umoblie(data_lowlight)
    # adjustment
    adjust_illu = torch.pow(illumination, 0.1)
    # print(adjust_illu)
    res_image = adjust_illu*((data_lowlight - noise)/illumination)
    res_image = torch.clamp(res_image, min=0, max=1)

    end_time = (time.time() - start)
    print(end_time)
    image_path = image_path.replace('test_data', 'asd')
    result_path = image_path
    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
    torchvision.utils.save_image(res_image, result_path, normalize=True)

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




