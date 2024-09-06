import torch
import torch.nn as nn


class G_net(nn.Module):
    def __init__(self):
        super(G_net, self).__init__()

        self.illumination_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            torch.nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, 1, 1),
            torch.nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, 1, 1),

        )
    def forward(self, x):
        illumination = torch.sigmoid(self.illumination_net(x))
        return illumination



class JNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 1, 1),
            torch.nn.InstanceNorm2d(16),
            torch.nn.LeakyReLU(0.2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.InstanceNorm2d(32),
            torch.nn.LeakyReLU(0.2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.LeakyReLU(0.2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, 1, 1),
            torch.nn.InstanceNorm2d(32),
            torch.nn.LeakyReLU(0.2)
        )

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(32, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data1 = self.final(data)

        return data1


class TNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 1, 1),
            torch.nn.InstanceNorm2d(16),
            torch.nn.LeakyReLU(0.2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.InstanceNorm2d(32),
            torch.nn.LeakyReLU(0.2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.LeakyReLU(0.2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, 1, 1),
            torch.nn.InstanceNorm2d(32),
            torch.nn.LeakyReLU(0.2)
        )

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(32, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )
    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data1 = self.final(data)

        return data1

class UIEnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_net = JNet()
        self.mask_net = TNet()
        self.G_net = G_net()


    def forward(self, data):
        x_j = self.image_net(data)
        x_t = self.mask_net(data)
        x_a = self.G_net(data)

        return x_j, x_t, x_a



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.randn(8, 3, 256, 256).to(device)
    model = UIEnet().to(device)
    j_out, t_out, a_out = model(img)
    print(j_out.shape)
    num_params = sum(p.numel() for p in model.parameters()) / 1e3  # 将结果除以1000
    print(f"Number of parameters: {num_params:.2f} k")
