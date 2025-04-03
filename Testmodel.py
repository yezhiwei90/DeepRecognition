"""
Copyright (c) [2025] [Ye Zhiwei]  Dalian University of Technology
All rights reserved.

This file is part of Deep recongnition of Moleculer fluorescence.

This code is licensed under the [MIT]
You may not use this file except in compliance with the License.

"""
from Reconstruction import modeldata, basicneuralnetwork
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import tifffile as tiff
import os
from gaussian_kernel import gauss_kernel
# Initialize lists to save outputs and targets



def evaluate(model, data_loader, gaussiankernel_tensor):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0
    criterion = nn.MSELoss()  # 假设我们的损失函数仍然是 L1Loss
    all_outputs = []
    gauKern = gauss_kernel

    with torch.no_grad():  # 评估时不需要计算梯度
        for inputs, targets in data_loader:
            outputs = model(inputs)  # 前向传播
            conved_outputs = gauKern.convlayer(gaussiankernel_tensor, outputs)
            conved_targets = gauKern.convlayer(gaussiankernel_tensor, targets)

            loss = criterion(conved_outputs, conved_targets) + torch.norm(outputs, p=1)  # 计算损失
            total_loss += loss.item()  # 累加损失
            #print(f"outputs shape: {outputs.shape}")

            # Store outputs and targets
            all_outputs.append(outputs.cpu().numpy())  # Move to CPU and convert to numpy

    # Concatenate all outputs for a single array
    all_outputs = np.concatenate(all_outputs, axis=0)  # Shape: (total_frames, 32, height, width, channels)
    # Print the shape of all_outputs
    print("Shape of all_outputs:", all_outputs.shape)

    # Save to a multi-page TIFF file
    # Reshape if necessary based on your specific requirement
    # Assuming you want to save each frame (32 images) as separate pages
    # Prepare output for saving, assuming (N, 32, H, W, C) shape where C is channels
    # If C is 3 for RGB, you can save it directly; if C is 1, you might need to remove the channel dimension.

    # Check the dimensions; the following assumes C = 1 (grayscale)
    if all_outputs.shape[-1] == 1:  # If output has a single channel, squeeze out the channel dimension
        all_outputs = all_outputs.squeeze(-1)  # Shape will be (total_frames, 1, height, width)
    # print("after Shape of all_outputs:", all_outputs.shape)
    # Reshape further to (total_frames * 32, height, width) for saving as multi-page TIFF
    output_for_tiff = all_outputs.reshape(-1, all_outputs.shape[2], all_outputs.shape[3])
    # print("Shape of output_for_tiff :", output_for_tiff.shape)
    # Save using tifffile
    savpath = r"D:\Deep-learning\MolecularFluorescence\Version1.2\Test\EvaReconstruction"
    os.makedirs(savpath, exist_ok=True)  # Create directory if it does not exist

    # Save each frame separately as a TIFF file
    for frame_idx in range(output_for_tiff.shape[0]):  # Iterate over total batches
            # Extract the current frame
            frame_image = output_for_tiff[frame_idx]


            # Construct the output file name
            tiff_filename = os.path.join(savpath, f"reconstruction_frame_{frame_idx}.tiff")

            # Save the individual frame as a TIFF file
            tiff.imwrite(tiff_filename, frame_image, metadata={'axes': 'YX'})  # 'Y' for height, 'X' for width

            print(f"Saved frame {frame_idx} to {tiff_filename}")

    print("All frames saved successfully.")

    average_loss = total_loss / len(data_loader)
    print(f"Evaluation Loss: {average_loss:.4f}")
    return average_loss

if __name__ == "__main__":
    testdata = modeldata(imgspath=r'D:\Deep-learning\MolecularFluorescence\Version1.2\Test\0.05s 640 nm 80%_1.tif',
                          reconspath=r'D:\Deep-learning\MolecularFluorescence\Version1.2\Test\Reconstructions')
    #print(modeldata.get_recon(0))
    #print(modeldata.get_image(0))
    imgh = testdata.images.shape[1]
    imgw = testdata.images.shape[2]

    model_path = r"D:\Deep-learning\MolecularFluorescence\Version1.2\Test\basic_neural_network.pth"
    # 加载模型
    nw_loaded = basicneuralnetwork(imgh, imgw).cuda()  # 确保在相同的设备上
    nw_loaded.load_state_dict(torch.load(model_path,weights_only=True))
    nw_loaded.eval()  # 将模型设置为评估模式
    print("model successfully load")

    # 添加 channels 维度。假设输入是灰度图像
    images_tensor = torch.tensor(testdata.images, dtype=torch.float32).unsqueeze(1)  # (7080, 1, 295, 512)
    recons_tensor = torch.tensor(testdata.recons, dtype=torch.float32).unsqueeze(1)  # (7080, 1, 295, 512)

    # 将重建图像转换为 TensorDataset
    eval_dataset = TensorDataset(images_tensor.cuda(),
                                 recons_tensor.cuda())
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    # create of a kernel on gpu.
    sigma = 3
    size = (7, 7)
    k = gauss_kernel
    gaussiankernel = k.corekernel(sigma, size)
    gaussiankernel_tensor = torch.tensor(gaussiankernel, dtype=torch.float32).to('cuda').view(1,1,size[0],size[1])
    #print(f"size of tensor:{gaussiankernel_tensor.shape}")

    # 评估模型
    evaluate(nw_loaded, eval_dataloader, gaussiankernel_tensor)