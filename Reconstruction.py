"""
Copyright (c) [2025] [Ye Zhiwei]  Dalian University of Technology
All rights reserved.

This file is part of Deep recongnition of Moleculer fluorescence.

This code is licensed under the [MIT]
You may not use this file except in compliance with the License.

"""
import os
import tifffile as tiff
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from gaussian_kernel import gauss_kernel
from torch.utils.data import DataLoader, TensorDataset

class modeldata:
    def __init__(self, imgspath, reconspath):
        self.images = tiff.imread(imgspath)

        full_paths = [os.path.join(reconspath, filename) for filename in os.listdir(reconspath)]
        recons = []
        for path in full_paths:
            recons.append(tiff.imread(path))
        self.recons =  np.array(recons)

    def get_image(self, index):
        """Return the image at a given index to visualize."""
        return self.images[index]

    def get_recon(self, index):
        """Return the reconstructed image at a given index."""
        return self.recons[index]

class basicneuralnetwork(nn.Module):
    def __init__(self, imgh=512, imgw=295):
        super().__init__()

        self.imgh = imgh
        self.imgw = imgw

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # 解码器
        self.decoder = nn.Sequential(
             nn.Upsample(size=(self.imgh // 4,self.imgw // 4),mode='bicubic', align_corners=False),

             nn.Conv2d(512,128,kernel_size=3, padding=1),
             nn.BatchNorm2d(128),
             nn.ReLU(),
             nn.Upsample(size=(self.imgh // 2,self.imgw // 2),mode='bicubic', align_corners=False),

             nn.Conv2d(128, 64, kernel_size=3, padding=1),
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.Upsample(size=(self.imgh,self.imgw),mode='bicubic', align_corners=False),

             nn.Conv2d(64, 32, kernel_size=3, padding=1),
             nn.BatchNorm2d(32),
             nn.ReLU(),

             nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = self.encoder(x)
        #print(f"After encoder: {x.shape}")  # Print shape after encoder

        x = self.decoder(x)
        #print(f"After decoder: {x.shape}")  # Print shape after decoder

        return x

    def show_image(self,img):
        # Display the image
        plt.imshow(img)
        plt.axis('off')  # Turn off axis labels
        plt.show()


def train(model, data, gaussiankernel_tensor=None, num_epochs=20, batch_size=16, learning_rate =0.01,
          modeltemppath = r"D:\Deep-learning\MolecularFluorescence\Version1.2\Test\Temp"):
    model.train()

    # 实例化网络
    #expert_weights=[0.7,  0.3]#第一个是单分子专家的权重
    #expert_weights= ToTensor()(expert_weights)

    # 定义损失函数和优化器
    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 添加 channels 维度。假设输入是灰度图像
    images_tensor = torch.tensor(data.images, dtype=torch.float32).unsqueeze(1)  # (7080, 1, 295, 512)
    recons_tensor = torch.tensor(data.recons, dtype=torch.float32).unsqueeze(1)  # (7080, 1, 295, 512)
    #print(f"Input data shape: {images_tensor.shape}")  # 应该是 (7080, 1, 295, 512)
    #print(f"Reconstruction data shape: {recons_tensor.shape}")  # 应该是 (7080, 1, 295, 512)

    # 将图像和重建图像转换为 TensorDataset
    dataset = TensorDataset(images_tensor.cuda()
                                 , recons_tensor.cuda()
                             )

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # creation of gaussian kernel
    gauKern = gauss_kernel

    for epoch in range(num_epochs):
        epoch_loss = 0

        for inputs, targets in dataloader:
            # 前向传播
            outputs = model(inputs)

            # 计算损失
            conved_targets = gauKern.convlayer(gaussiankernel_tensor, targets)
            conved_outputs = gauKern.convlayer(gaussiankernel_tensor, outputs)
            loss = criterion(conved_outputs, conved_targets) + torch.norm(outputs, p=1)

            # 清零之前的梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            epoch_loss += loss.item()

            # 打印每个 epoch 的损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), os.path.join(modeltemppath,f"neuralintermedia{epoch}.pth"))

    print("Training complete.")

if __name__ == "__main__":
    modeldata = modeldata(imgspath=r'D:\Deep-learning\MolecularFluorescence\Version1.2\Dataset\0.05s 640 nm 80%_1_X2.tif',
                          reconspath=r'D:\Deep-learning\MolecularFluorescence\Version1.2\Dataset\Reconstruction')

    #device = torch.device('cuda')
    #print("Input data frame number:", modeldata.images.shape[0])
    #print("Input data width:", modeldata.images.shape[1])
    #print("Input data height:", modeldata.images.shape[2])
    #print("Reconstruction frame number：",modeldata.recons.shape[0])
    #print("Reconstruction data width:", modeldata.recons.shape[1])
    #print("Reconstruction data height:", modeldata.recons.shape[2])

    imgh = modeldata.images.shape[1]
    imgw = modeldata.images.shape[2]

    # create of a kernel on gpu.
    sigma = 3
    size = (7, 7)
    k = gauss_kernel
    gaussiankernel = k.corekernel(sigma, size)
    gaussiankernel_tensor = torch.tensor(gaussiankernel, dtype=torch.float32).to('cuda').view(1,1,size[0],size[1])
    #print(f"size of tensor:{gaussiankernel_tensor.shape}")

    nw = basicneuralnetwork(imgh,imgw)
    nw.cuda()

    modeltemppath = r"D:\Deep-learning\MolecularFluorescence\Version1.2\Test\Temp"
    os.makedirs(modeltemppath, exist_ok=True)  # Create directory if it does not exist

    # 训练模型
    train(nw, modeldata, gaussiankernel_tensor = gaussiankernel_tensor, num_epochs=300, batch_size=32, learning_rate=0.001,
          modeltemppath = r"D:\Deep-learning\MolecularFluorescence\Version1.2\Test\Temp")

    model_path = r"D:\Deep-learning\MolecularFluorescence\Version1.2\Test\basic_neural_network.pth"


    torch.save(nw.state_dict(), model_path)
    print(f"model has been saved to {model_path}")
