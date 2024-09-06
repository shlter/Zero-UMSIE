import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import GaussianBlur


class Irn(nn.Module):
    def __init__(self):
        super(Irn, self).__init__()

    def forward(self,  illumination, reflectance, noise):

        pass


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size


class MyGaussianBlur(nn.Module):
    def __init__(self, sigma, max_kernel_size=31):
        super(MyGaussianBlur, self).__init__()
        # 计算高斯核大小，通常为 6 * sigma + 1
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保核大小为奇数
        kernel_size = min(kernel_size, max_kernel_size)

        self.blur = GaussianBlur(kernel_size, [sigma, sigma])

    def forward(self, x):
        return self.blur(x)

class MultiScaleLuminanceEstimation(nn.Module):
    def __init__(self):
        super(MultiScaleLuminanceEstimation, self).__init__()
        # 定义三种不同尺度的高斯模糊
        self.gaus_15 = MyGaussianBlur(sigma=15)
        self.gaus_60 = MyGaussianBlur(sigma=45)  # 60
        self.gaus_90 = MyGaussianBlur(sigma=75)  # 90
        # self.gaus_120 = MyGaussianBlur(sigma=120)
    def forward(self, img):
        x_15 = self.gaus_15(img)
        x_60 = self.gaus_60(img)
        x_90 = self.gaus_90(img)

        img = (x_15 + x_60 + x_90) / 3

        return img

class Retinex_loss1(nn.Module):
    def __init__(self):
        super(Retinex_loss1, self).__init__()
        self.MSELoss = nn.MSELoss()
        self.L1Loss = nn.L1Loss()

    def forward(self, x, y):
        luminance_estimator = MultiScaleLuminanceEstimation()
        x_est = luminance_estimator(x)
        y_est = luminance_estimator(y)
        # 计算均方误差
        loss = self.MSELoss(x_est, y_est)
        return loss


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)

        Dr = torch.pow(mr-0.5, 2)
        Dg = torch.pow(mg-0.5, 2)
        Db = torch.pow(mb-0.5, 2)

        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Dg, 2) + torch.pow(Db, 2), 0.5)

        return k

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class C2R(nn.Module):
    def __init__(self, ablation=False):

        super(C2R, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = 1.0
        self.ab = ablation
        print('*******************use normal 6 neg clcr loss****************')

    def forward(self, a, p, inp, weight):
        a_vgg, p_vgg = self.vgg(a), self.vgg(p)
        inp_vgg = self.vgg(inp)
        inp_weight = weight
        loss = 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            # if not self.ab:
            #     d_inp = self.l1(a_vgg[i], inp_vgg[i].detach())
            #     contrastive = d_ap / (d_inp * inp_weight + 1e-7)
            # else:
            #     contrastive = d_ap

            loss += self.weights * d_ap
            # loss += self.weights[i] * contrastive

        return loss
