import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
from fvcore.nn import FlopCountAnalysis
from utils import AverageMeter, write_img, chw_to_hwc, pad_img
from datasets.loader import PairLoader
from models import *
from models.wavelet import wt_m


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gunet_ss', type=str, help='model name')
parser.add_argument('--ll_predict_model', default='gunet_ss', type=str, help='model name')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--data_dir', default='/mnt/d/Train Data/dz_data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--test_set', default='RESIDE-OUT/test', type=str, help='test dataset name')
parser.add_argument('--exp', default='reside-out', type=str, help='experiment setting')
args = parser.parse_args()

def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	PSNR = AverageMeter()
	SSIM = AverageMeter()
	PSNR_L = AverageMeter()
	PSNR_D = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()
	pytorch_total_params = sum(p.numel() for p in network.parameters())
	print('params:',pytorch_total_params)
	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
	input = torch.randn(1, 3, 640, 512).to(device)
	flops = FlopCountAnalysis(network, input)
	print('GFLOPS:',flops.total() / 1000 / 1000 / 1000)


	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')
	sfm = wt_m().to(device)

	for idx, batch in enumerate(test_loader):
		input = batch['source'].cuda()
		target = batch['target'].cuda()

		filename = batch['filename'][0]

		with torch.no_grad():
			coeffs = sfm(target)
			ll_label = coeffs[:, [0, 4, 8]]
			# ll_scale1_label, ll_scale2_label = net.ll_scale(label)
			detail_label = coeffs[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
			coeffs_1 = sfm(ll_label)
			ll_label_1 = coeffs_1[:, [0, 4, 8]]
			detail_label_1 = coeffs_1[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
			H, W = input.shape[2:]
			H_l,W_l = detail_label.shape[2:]
			input = pad_img(input, network.patch_size if hasattr(network, 'patch_size') else 64)
			# output,(ll,detail) = network(input)
			output = network(input)
			ll = None
			if isinstance(output, list):
				output, out_list = output
				output.clamp_(-1, 1)
				# ll = ll[:, :, :H_l, :W_l]
				# detail = detail[:, :, :H_l, :W_l]
			else:
				output.clamp_(-1, 1)

			output.clamp_(-1, 1)
			output = output[:, :, :H, :W]

			# [-1, 1] to [0, 1]
			output = output * 0.5 + 0.5
			target = target * 0.5 + 0.5

			psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()
			if not(ll is None):
				psnr_ll = 10 * torch.log10(4 / F.mse_loss(ll,ll_label)).item()
				psnr_detail = 10 * torch.log10(4/ F.mse_loss(detail,detail_label)).item()
				PSNR_L.update(psnr_ll)
				PSNR_D.update(psnr_detail)
			_, _, H, W = output.size()
			down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
			ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
							F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
							data_range=1, size_average=False).item()

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)

		print('Test: [{0}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
			  .format(idx, psnr=PSNR, ssim=SSIM))
		print(PSNR_L.avg, PSNR_D.avg)

		f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)

	f_result.close()

	os.rename(os.path.join(result_dir, 'results.csv'),
			  os.path.join(result_dir, '%.03f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


def main():
	if args.model == "ll_predict":
		model = eval(args.ll_predict_model)()
		network = eval(args.model)(model)
	else:
		network = eval(args.model)()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, "best_" + args.model+ "_" + args.ll_predict_model +'.pth' if args.model == "ll_predict" else "best_" + args.model+'.pth')
	print(saved_model_dir)

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	dataset_dir = os.path.join(args.data_dir, args.test_set)
	test_dataset = PairLoader(dataset_dir, 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	result_dir = os.path.join(args.result_dir, args.test_set, args.exp, args.model)
	test(test_loader, network, result_dir)


if __name__ == '__main__':
	main()

