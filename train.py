import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from fvcore.nn import FlopCountAnalysis
from utils import AverageMeter, CosineScheduler, pad_img
from datasets import PairLoader
from models import *
from models.wavelet import wt_m,iwt_m


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ll_predict', type=str, help='model name')
parser.add_argument('--ll_predict_model', default='gunet_ss', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--use_mp', action='store_true', default=True, help='use Mixed Precision')
parser.add_argument('--use_ddp', action='store_true', default=False, help='use Distributed Data Parallel')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='/mnt/d/Train Data/dz_data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--train_set', default='RESIDE-6K/train', type=str, help='train dataset name')
parser.add_argument('--val_set', default='RESIDE-6K/test', type=str, help='valid dataset name')
parser.add_argument('--exp', default='reside-6k', type=str, help='experiment setting')
args = parser.parse_args()


# training environment
if args.use_ddp:
	torch.distributed.init_process_group(backend='nccl', init_method='env://')
	world_size = dist.get_world_size()
	local_rank = dist.get_rank()
	torch.cuda.set_device(local_rank)
	if local_rank == 0: print('==> Using DDP.')
else:
	world_size = 1

# training config
with open(os.path.join('configs', args.exp, 'base.json'), 'r') as f:
	b_setup = json.load(f)

variant = args.model.split('_')[-1]
config_name = 'default.json' if "wave" in args.model or "ll_predict" in args.model else ('model_'+variant+'.json' if variant in ['t', 's', 'b', 'd'] else 'default.json')	# default.json as baselines' configuration file
# config_name = 'model_t.json'
with open(os.path.join('configs', args.exp, config_name), 'r') as f:
	m_setup = json.load(f)


def reduce_mean(tensor, nprocs):
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= nprocs
	return rt


def train(train_loader, network, criterion, optimizer, scaler, frozen_bn=False):
	losses = AverageMeter()

	torch.cuda.empty_cache()

	network.eval() if frozen_bn else network.train()  # simplified implementation that other modules may be affected
	sfm = wt_m().cuda()
	for batch in tqdm(train_loader):
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()
		coeffs = sfm(target_img)
		coeffs_two = sfm(coeffs)
		coeffs_three = sfm(coeffs_two)
		ll_label = coeffs[:, [0, 4, 8]]
		# # ll_scale1_label, ll_scale2_label = net.ll_scale(label)
		detail_label = coeffs[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		# coeffs_1 = sfm(ll_label)
		# ll_label_1 = coeffs_1[:, [0, 4, 8]]
		# detail_label_1 = coeffs_1[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]

		with autocast(args.use_mp):
			# output = network(source_img)[0]
			output = network(source_img)
			if isinstance(output, tuple):
				output, (ll, detail) = output
				loss = 0.5 * criterion(output, target_img) + 0.5 * criterion(ll,ll_label) + criterion(detail,detail_label)
			# output,(out) = output
			# loss = criterion(output, target_img) + criterion(out, coeffs_three)
			else:
				loss = criterion(output, target_img)

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		if args.use_ddp: loss = reduce_mean(loss, dist.get_world_size())
		losses.update(loss.item())

	return losses.avg


def valid(val_loader, network):
	PSNR = AverageMeter()
	PSNR_LL = AverageMeter()
	PSNR_DETAILS = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()
	sfm = wt_m().cuda()
	for batch in tqdm(val_loader):
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()
		coeffs = sfm(target_img)
		coeffs_two = sfm(coeffs)
		coeffs_three = sfm(coeffs_two)
		ll_label = coeffs[:, [0, 4, 8]]
		# # ll_scale1_label, ll_scale2_label = net.ll_scale(label)
		detail_label = coeffs[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		# coeffs_1 = sfm(ll_label)
		# ll_label_1 = coeffs_1[:, [0, 4, 8]]
		# detail_label_1 = coeffs_1[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]

		with torch.no_grad():
			H, W = source_img.shape[2:]
			source_img = pad_img(source_img, network.module.patch_size if hasattr(network.module, 'patch_size') else 64)
			# output = network(source_img)[0].clamp_(-1, 1)
			output_map = network(source_img)
			if isinstance(output_map, tuple):
				# output,(out) = output
				output, (ll, detail) = output_map
				output.clamp_(-1, 1)
			else:
				output = output_map.clamp_(-1, 1)
			output = output[:, :, :H, :W]

		mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		if isinstance(output_map, tuple):
			mse_loss_ll = F.mse_loss(ll, ll_label, reduction='none').mean((1, 2, 3))
			mse_loss_detail = F.mse_loss(detail, detail_label, reduction='none').mean((1, 2, 3))
			psnr_ll = 10 * torch.log10(4 / mse_loss_ll).mean()
			psnr_detail = 10 * torch.log10(4 / mse_loss_detail).mean()
			PSNR_LL.update(psnr_ll, source_img.size(0))
			PSNR_DETAILS.update(psnr_detail, source_img.size(0))
		# if args.use_ddp: psnr = reduce_mean(psnr, dist.get_world_size())		# comment this line for more accurate validation

		PSNR.update(psnr.item(), source_img.size(0))
	print('PSNR: {:.4f}\nPSNR_LL: {:.4f}\nPSNR_detail: {:.4f}'.format(PSNR.avg, PSNR_LL.avg, PSNR_DETAILS.avg))

	return PSNR.avg


def main():
	# define network, and use DDP for faster training
	if args.model == "ll_predict":
		model = eval(args.ll_predict_model)()
		save_dir = os.path.join(args.save_dir, args.exp)
		train_all = True
		if os.path.exists(os.path.join(save_dir,"best_" +  args.ll_predict_model + '.pth')):
			checkpoint = torch.load(os.path.join(save_dir, args.ll_predict_model + '.pth'), map_location='cpu')['state_dict']
			for key in list(checkpoint.keys()):
				if 'module.' in key:
					checkpoint[key.replace('module.', '')] = checkpoint[key]
					del checkpoint[key]
			model.load_state_dict(checkpoint)
			# train_all = False
			train_all = True
		network = eval(args.model)(model)
		network.train_all = train_all
		network.cuda()
	else:
		network = eval(args.model)()
		network.cuda()

	if args.use_ddp:
		network = DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)
		if m_setup['batch_size'] // world_size < 16:
			if local_rank == 0: print('==> Using SyncBN because of too small norm-batch-size.')
			nn.SyncBatchNorm.convert_sync_batchnorm(network)
	else:
		network = DataParallel(network)
		if m_setup['batch_size'] // torch.cuda.device_count() < 16:
			print('==> Using SyncBN because of too small norm-batch-size.')
			convert_model(network)
	network.eval().cuda()
	torch_input = torch.randn(1, 3, 640, 512).cuda()
	flops = FlopCountAnalysis(network, torch_input)
	print(flops.total() / 1000 / 1000 / 1000)
	pytorch_total_params = sum(p.numel() for p in network.parameters())
	print(pytorch_total_params)

	# define loss function
	criterion = nn.MSELoss()
	# criterion = nn.L1Loss()

	# define optimizer
	optimizer = torch.optim.AdamW(network.parameters(), lr=m_setup['lr'], weight_decay=b_setup['weight_decay'])
	# lr_scheduler = CosineScheduler(optimizer, param_name='lr', t_max=b_setup['epochs'], value_min=m_setup['lr'] * 1e-2,
	# 							   warmup_t=b_setup['warmup_epochs'], const_t=b_setup['const_epochs'])
	lr_scheduler = CosineAnnealingLR(optimizer, T_max=b_setup['epochs'], eta_min=m_setup['lr'] * 1e-2)
	# wd_scheduler = CosineScheduler(optimizer, param_name='weight_decay', t_max=b_setup['epochs'])	# seems not to work
	wd_scheduler = CosineAnnealingLR(optimizer, T_max=b_setup['epochs'])
	scaler = GradScaler()

	# load saved model
	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)
	if not os.path.exists(os.path.join(save_dir, args.model+ "_" + args.ll_predict_model +'.pth' if args.model == "ll_predict" else args.model+'.pth')):
		best_psnr = 0
		cur_epoch = 0
	else:
		if not args.use_ddp or local_rank == 0: print('==> Loaded existing trained model.')
		model_info = torch.load(os.path.join(save_dir, args.model+ "_" + args.ll_predict_model +'.pth' if args.model == "ll_predict" else args.model+'.pth'), map_location='cpu')
		network.load_state_dict(model_info['state_dict'])
		optimizer.load_state_dict(model_info['optimizer'])
		lr_scheduler.load_state_dict(model_info['lr_scheduler'])
		wd_scheduler.load_state_dict(model_info['wd_scheduler'])
		scaler.load_state_dict(model_info['scaler'])
		cur_epoch = model_info['cur_epoch']
		best_psnr = model_info['best_psnr']

	# define dataset
	train_dataset = PairLoader(os.path.join(args.data_dir, args.train_set), 'train',
							   b_setup['t_patch_size'],
							   b_setup['edge_decay'],
							   b_setup['data_augment'],
							   b_setup['cache_memory'])
	train_loader = DataLoader(train_dataset,
							  batch_size=m_setup['batch_size'] // world_size,
							  sampler=RandomSampler(train_dataset, num_samples=b_setup['num_iter'] // world_size),
							  num_workers=args.num_workers // world_size,
							  pin_memory=True,
							  drop_last=True,
							  persistent_workers=True)  # comment this line for cache_memory
	print(len(train_dataset))

	val_dataset = PairLoader(os.path.join(args.data_dir, args.val_set), b_setup['valid_mode'],
							 b_setup['v_patch_size'])
	val_loader = DataLoader(val_dataset,
							batch_size=max(int(m_setup['batch_size'] * b_setup['v_batch_ratio'] // world_size), 1),
							# sampler=DistributedSampler(val_dataset, shuffle=False),		# comment this line for more accurate validation
							num_workers=args.num_workers // world_size,
							pin_memory=True)

	# start training
	loss_total = []
	PSNR_total = []
	lr_total = []
	if not args.use_ddp or local_rank == 0:
		print('==> Start training, current model name: ' + args.model)
		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

	for epoch in range(cur_epoch, b_setup['epochs'] + 1):
		print(epoch)
		frozen_bn = epoch > (b_setup['epochs'] - b_setup['frozen_epochs'])
		lr_total.append(wd_scheduler.get_last_lr())

		loss = train(train_loader, network, criterion, optimizer, scaler, frozen_bn)
		loss_total.append(loss)
		print(loss)
		lr_scheduler.step(epoch + 1)
		wd_scheduler.step(epoch + 1)

		if not args.use_ddp or local_rank == 0:
			writer.add_scalar('train_loss', loss, epoch)

		if epoch % b_setup['eval_freq'] == 0:
			avg_psnr = valid(val_loader, network)
			PSNR_total.append(avg_psnr)

			if not args.use_ddp or local_rank == 0:
				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					torch.save({'cur_epoch': epoch + 1,
								'best_psnr': best_psnr,
								'state_dict': network.state_dict(),
								'optimizer': optimizer.state_dict(),
								'lr_scheduler': lr_scheduler.state_dict(),
								'wd_scheduler': wd_scheduler.state_dict(),
								'scaler': scaler.state_dict(),
								'loss':loss_total,
								'PSNR':PSNR_total,'lr':lr_total},
							   os.path.join(save_dir, "best_" + args.model+ "_" + args.ll_predict_model +'.pth' if args.model == "ll_predict" else "best_" + args.model+'.pth'))
				torch.save({'cur_epoch': epoch + 1,
              'best_psnr': best_psnr,
							'state_dict': network.state_dict(),
							'optimizer': optimizer.state_dict(),
							'lr_scheduler': lr_scheduler.state_dict(),
							'wd_scheduler': wd_scheduler.state_dict(),
							'scaler': scaler.state_dict(),
							'loss': loss_total,
							'PSNR': PSNR_total,
							'lr':lr_total},
						   os.path.join(save_dir, args.model+ "_" + args.ll_predict_model +'.pth' if args.model == "ll_predict" else args.model+'.pth'))

				writer.add_scalar('valid_psnr', avg_psnr, epoch)
				writer.add_scalar('best_psnr', best_psnr, epoch)

			if args.use_ddp: dist.barrier()


if __name__ == '__main__':
	main()

