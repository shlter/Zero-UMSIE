from model.UIERetinex import UIEnet
import conf
import dataloader
import argparse
import loss.Myloss as Myloss
import torch
import os

import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train_retinex_model(net, train_loader, epochs):
    net.to(conf.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=conf.lr, betas=(0.9, 0.999), eps=1e-8)

    L_TV = Myloss.L_TV()
    L_color = Myloss.ColorLoss()
    L_Retinex = Myloss.Retinex_loss1()
    mse_loss = torch.nn.MSELoss().cuda()
    for epoch in range(epochs):
        net.train()
        for i, img in enumerate(train_loader):
            img = img.to(conf.device)
            j_out, t_out, a_out = net(img)

            lam = np.random.beta(1, 1)
            gam = np.random.beta(1, 1)
            I_res = t_out * j_out + (1 - t_out) * a_out
            input_mix_multi = lam * (gam * I_res + (1 - t_out) * a_out) + (1 - lam) * j_out

            # No.2
            Loss_MSE = 1 * mse_loss(I_res, img)
            Loss_TV = 1600 * L_TV(t_out)
            Loss_color = 0.5 * torch.mean(L_color(j_out))
            Loss_Retinex = L_Retinex(j_out, input_mix_multi)
            loss = Loss_TV + Loss_MSE + Loss_Retinex + Loss_color

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}")

        # 每个epoch结束后保存模型
        torch.save(net.state_dict(), config.snapshots_folder + f'model_epoch_{epoch}.pth')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str,default="data/train/train_data_SUIM_light/")  # train_EUVP_darkorlight
    parser.add_argument('--lr', type=float, default=0.0001)# 0.0001
    parser.add_argument('--weight_decay', type=float, default=0.0001) #0.0001
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)  # 100
    parser.add_argument('--train_batch_size', type=int, default=8)  # 8
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--snapshots_folder', type=str, default="epochs/")
    parser.add_argument('--load_pretrain', type=bool, default=False)

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    # 模型初始化
    # net = RRDNet()
    # net = UltraLight_VM_UNet()
    net = UIEnet()
    # 假设train_loader是您的数据加载器
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    # 开始训练
    train_retinex_model(net, train_loader, config.num_epochs)
