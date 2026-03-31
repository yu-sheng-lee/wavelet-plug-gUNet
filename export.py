import os
import argparse
import torch
import numpy as np
from collections import OrderedDict
from datasets.loader import PairLoader
import torch.utils.data as Data
from torch.nn import functional as F
from fvcore.nn import FlopCountAnalysis
from models import *
import onnx
import onnxruntime as ort
import tqdm
import pytorch_ssim
from models.wavelet import wt_m,iwt_m
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='wavedown_gnet_two', type=str, help='model name')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--OUTPUT_ONNX', type=str, default="wavedown_gnet_two.onnx",
                      help='Output onnx')
parser.add_argument('--DATA_PATH', type=str, default="/mnt/d/Train Data/dz_data/RESIDE-6K",
                      help='Path to Dataset')

parser.add_argument('--exp', default='reside-6k', type=str, help='experiment setting')
args = parser.parse_args()


class wavelet(nn.Module):
	def __init__(self,model=gunet_t()):
		super(wavelet, self).__init__()
		self.xfm = wt_m(requires_grad=False)
		self.ifm = iwt_m(requires_grad=False)
	def forward(self,x):
		x = self.xfm(x)
		x = self.xfm(x)
		x = self.ifm(x)
		x = self.ifm(x)
		return x

	def set_train(self):
		self.train()
		self.ll_encoder.set_train()
def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict



def main():
	TRANSFROM_SCALES = (480, 640)
	# device = torch.device("cpu")
	network = eval(args.model)()
	# network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')
	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
	#
	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		# network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	# test_data = PairLoader(os.path.join(args.DATA_PATH, "test"), "valid",
	# 					   (480, 640))
	# test_loader = Data.DataLoader(test_data, batch_size=1,
	# 							  shuffle=False, num_workers=4, pin_memory=True)

	# network = gunet_do_ones()
	# network = wavelet()



	network = network.eval().to(device)
	torch_input = torch.randn(1, 3, TRANSFROM_SCALES[0],TRANSFROM_SCALES[1]).to(device)
	torch_output = network(torch_input)
	flops = FlopCountAnalysis(network, torch_input)
	print(flops.total() / 1000 / 1000 / 1000)
	pytorch_total_params = sum(p.numel() for p in network.parameters())
	print(pytorch_total_params)
	network_jit = torch.jit.trace(network, torch_input)
	network.out = True

	torch.onnx.export(
		# network_jit,  # model to export
		network,  # model to export
		(torch_input),  # inputs o。f the model,
		args.OUTPUT_ONNX,  # filename of the ONNX model
		input_names=["input"],  # Rename inputs for the ONNX model
		# opset_version=19,
	)
	print("export done")
	onnx_model = onnx.load(args.OUTPUT_ONNX)
	onnx.checker.check_model(onnx_model)

	model = ort.InferenceSession(args.OUTPUT_ONNX, providers=[
		"CUDAExecutionProvider",
		"CPUExecutionProvider"  # 使用CPU推理
	])
	ort_inputs = {model.get_inputs()[0].name: torch_input.detach().cpu().numpy()}
	onnx_output = model.run(None, ort_inputs)[0]
	np.testing.assert_almost_equal(torch_output.detach().cpu().numpy(), onnx_output, decimal=3)
	#
	# test_data = PairLoader(os.path.join(args.DATA_PATH, "test"), "valid",
	# 					   (TRANSFROM_SCALES[0],TRANSFROM_SCALES[1]))
	# test_loader = Data.DataLoader(test_data, batch_size=1,
	# 							  shuffle=False, num_workers=4, pin_memory=True)
	# total_psnr = []
	# total_onnx_psnr = []
	# total_ssim = []
	#
	# SsimLoss = pytorch_ssim.SSIM().to(device)
	#
	# for idx, batch in tqdm.tqdm(enumerate(test_loader)):
	# 	input = batch['source'].to(device)
	# 	target = batch['target'].to(device)
	#
	# 	filename = batch['filename'][0]
	#
	# 	with torch.no_grad():
	# 		H, W = input.shape[2:]
	# 		input = input.to(device)
	# 		output_torch = network(input).clamp_(-1, 1)
	#
	# 		# [-1, 1] to [0, 1]
	# 		output_torch = output_torch * 0.5 + 0.5
	# 		target = target * 0.5 + 0.5
	#
	# 		ort_inputs = {model.get_inputs()[0].name: input.detach().cpu().numpy()}
	# 		onnx_output = model.run(None, ort_inputs)[0]
	# 		onnx_output = torch.from_numpy(onnx_output).to(device).clamp_(-1, 1) * 0.5 + 0.5
	# 		# np.testing.assert_almost_equal(output_torch.detach().cpu().numpy(), onnx_output, decimal=3)
	# 		total_onnx_psnr.append(10 * torch.log10(1 / F.mse_loss(onnx_output, target)).item())
	# 		total_psnr.append(10 * torch.log10(1 / F.mse_loss(output_torch, target)).item())
	# 		total_ssim.append(SsimLoss(output_torch, target).item())
	# print('onnx PSNR: ', np.mean(total_onnx_psnr))
	# print('PSNR: ',np.mean(total_psnr))
	# print('SSIN: ', np.mean(total_ssim))











if __name__ == '__main__':
	main()
