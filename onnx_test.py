import argparse
import numpy as np
import os
import torch
import onnxruntime as ort

from datasets.loader import PairLoader
import torch.utils.data as Data
from torch.nn import functional as F
from PIL import Image
import tqdm
import math
from torchmetrics import F1Score
import torchvision.transforms as T
import pytorch_ssim
import time


def check_dir(path,n=0):
    if (not os.path.exists(path)) and n==0:
        # os.makedirs(path)
        return path
    elif not os.path.exists(path+"{:0>2d}".format(n)):
        # os.makedirs(path+"{:0>2d}".format(n))
        return path+"{:0>2d}".format(n)
    else:
        n+=1
        return check_dir(path,n)




def main(args):
    TRANSFROM_SCALES = (480, 640)
    dataset_name = args.DATA_PATH.split('/')[-1]
    ab_test_dir = check_dir(os.path.join(args.OUTPUT_PATH, dataset_name))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ort.InferenceSession(args.weight_path, providers=[
                                               "CUDAExecutionProvider",
                                               "CPUExecutionProvider"       # 使用CPU推理
                                           ])

    test_data = PairLoader(os.path.join(args.DATA_PATH, "test"), "valid",
                           TRANSFROM_SCALES)
    test_loader = Data.DataLoader(test_data, batch_size=1,
                                 shuffle=False, num_workers=4, pin_memory=True)

    SsimLoss = pytorch_ssim.SSIM().to(device)
    if args.half:
        SsimLoss.half()
    print("test_loader", len(test_loader))

    total_psnr = []
    total_ssim = []
    total_time = []
    memory_use = []

    model_name = args.weight_path
    for i in tqdm.tqdm(range(10)):
        input_tensor = np.random.rand(1,3,TRANSFROM_SCALES[0],TRANSFROM_SCALES[1]).astype(np.float32)
        if args.half:
            input_tensor =input_tensor.astype(np.float16)
        ort_inputs = {model.get_inputs()[0].name: input_tensor}
        output_map = model.run(None, ort_inputs)
    for batch_idx, batch in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            img_idx, label_idx ,names= batch["source"], batch["target"],batch["filename"]
            # img = Variable(img_idx.to(device))
            img = img_idx.detach().cpu().numpy()
            label_idx = label_idx.to(device)
            if args.half:
                img = img.astype(np.float16)
                label_idx = label_idx.half()

            ort_inputs = {model.get_inputs()[0].name: img}
            start_time = time.time()
            output_map = model.run(None, ort_inputs)[0]
            end_time = time.time()
            total_time.append(end_time - start_time)
            # ll = ll[0]
            # recon_R = ifm(torch.cat((ll[:, [0]], detail[:, 0:3]), dim=1))
            # recon_G = ifm(torch.cat((ll[:, [1]], detail[:, 3:6]), dim=1))
            # recon_B = ifm(torch.cat((ll[:, [2]], detail[:, 6:9]), dim=1))
            # output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
            if args.scale:
                output_map = torch.from_numpy(output_map).to(device).clamp_(-1, 1) * 0.5 + 0.5
                label_idx = label_idx * 0.5 + 0.5
            else:
                output_map = torch.from_numpy(output_map).to(device).clamp_(0, 1)
            total_psnr.append(10 * torch.log10(1 / F.mse_loss(output_map, label_idx)).item())

            total_ssim.append(SsimLoss(output_map, label_idx).item())




    print("############################")
    print("SSMI ", np.mean(total_ssim) ,"PSNR ",np.mean(total_psnr))
    print("avg inference time:",np.mean(total_time))
    print("avg GPU Memory:", np.mean(memory_use))

def opt_args():
    args = argparse.ArgumentParser()
    args.add_argument('--DATA_PATH', type=str, default="/mnt/d/Train Data/dz_data/RESIDE-6K",
                      help='Path to Dataset')
    args.add_argument('--weight_path', type=str, default="gunet_t.onnx",
                      help='Path to model weight')
    args.add_argument('--OUTPUT_PATH', type=str, default="./result",
                      help='Output Path')
    args.add_argument('--TRANSFROM_SCALES', type=int, default=256,
                      help='train img size')
    args.add_argument('--half', action="store_true", default=False,
                      help='use float16')
    args.add_argument('--scale', action="store_true", default=True,
                      help='scale output')
    return args.parse_args()


if __name__ == '__main__':
    opt = opt_args()
    main(opt)