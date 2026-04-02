import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from .norm_layer import *
from models.wavelet import wt_m,iwt_m,swt_m



class dwt_down(nn.Module):
	def __init__(self,dim,kernel_size=3,with_hh=False):
		super().__init__()
		self.with_hh = with_hh
		self.dim= dim
		self.dwt = wt_m(requires_grad=True)
		self.conv_lh = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2)
		self.conv_hl = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2)
		self.proj = nn.Conv2d(dim, dim, kernel_size=1)

	def forward(self, X):
		X = self.dwt(X)
		ll = X[:,[i*4 for i in range(self.dim)]]
		lh = X[:,[i*4+1 for i in range(self.dim)]]
		hl = X[:,[i*4+2 for i in range(self.dim)]]
		hh = X[:,[i*4+3 for i in range(self.dim)]]
		lh = self.conv_lh(lh+ll)
		hl = self.conv_hl(hl+ll)
		hi_sub = torch.cat([lh, hl, hh], dim=1)
		out = self.proj(lh+hl)
		return out,hi_sub


class WaveDownampler(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dim = in_channels
        self.dwt = wt_m(requires_grad=True)
        # self.conv_lh = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.conv_att = nn.Conv2d(in_channels*4, in_channels*4, 3,padding = 1)
        # self.conv_att = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels*8,kernel_size=3,padding=1),
		# 	nn.BatchNorm2d(in_channels*8),
		# 	nn.Conv2d(in_channels*8, in_channels * 4, kernel_size=3, padding=1),
        #     nn.MaxPool2d(2,stride=2))
        # self.conv_hl = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        # self.conv_lh_att = nn.Conv2d(in_channels*4, in_channels*2, 3, 1, 1)
        # self.to_att = nn.Sequential(
        #             nn.Conv2d(in_channels, in_channels, 1, 1, 0),
        #             nn.Sigmoid()
        # )
        # self.pw = nn.Conv2d(in_channels * 4, in_channels, 1, 1, 0)

    def forward(self, X):
        # conv_att = self.conv_att(X)
        X = self.dwt(X)
        ll = X[:, [i * 4 for i in range(self.dim)]]
        lh = X[:, [i * 4 + 1 for i in range(self.dim)]]
        hl = X[:, [i * 4 + 2 for i in range(self.dim)]]
        hh = X[:, [i * 4 + 3 for i in range(self.dim)]]
        # get attention
        # lh_out =  self.conv_lh(ll + lh)
        # hl_out =  self.conv_hl(ll + hl)
        conv_att = self.conv_att(X)
        ll_att,lh_att,hl_att,hh_att = conv_att[:,self.dim], conv_att[:,self.dim:self.dim*2],conv_att[:,self.dim*2:self.dim*3],conv_att[:,self.dim*3:]
        att_map = ll * hl_att + hl * hl_att + lh * lh_att + hh * hh_att
        # att_map = self.to_att(lh_out + hl_out)
        # squeeze
        # x_s = self.pw(X)
        # o = torch.mul(x_s, att_map) + x_s
        hi_sub = torch.cat([lh, hl, hh],dim = 1)
        # hi_bands = torch.cat([x_lh, x_hl, x_hh], dim=1)
        # hi_bands = torch.cat([x_lh, x_hl, x_hh], dim=1)
        # return o, hi_sub
        return att_map, hi_sub

class WaveUpsampler(nn.Module):
    def __init__(self, in_channels,with_hh=False):
        super().__init__()
        self.dim = in_channels
        self.idwt = iwt_m(requires_grad=True)
        # self.input_conv  = nn.Conv2d(in_channels*4, in_channels*4, kernel_size=3,padding =1)
        self.hi_sub_pre  = nn.Conv2d(in_channels*3, in_channels*3, kernel_size=3,padding =1)
		#
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2)
        )
        # self.att = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3,padding =1)
        self.att = nn.Conv2d(in_channels, in_channels*2, kernel_size=3,padding =1)
        self.fusions = nn.Conv2d(in_channels*2, in_channels, kernel_size=3,padding =1)

    def forward(self, X,ll_in,hh_in):
        hh_in = hh_in + self.hi_sub_pre(hh_in)
        idw_in = torch.cat((X,hh_in),dim = 1)[:,[0,3,4,5,1,6,7,8,2,9,10,11]]
        # idwt_out = self.idwt(self.input_conv(idw_in))
        idwt_out = self.idwt(idw_in)

        proj_out = self.proj(idw_in)
        # print(10 * torch.log10(1 / F.mse_loss(idwt_out,proj_out)))
        # att = self.att(torch.cat((idwt_out,proj_out),dim = 1))
        att = self.att(ll_in)
        att_idwt ,att_proj =  att[:,:self.dim], att[:,self.dim:]
        # print(torch.mean(att_idwt),torch.mean(att_proj))
        out = idwt_out * att_idwt + proj_out * att_proj
        # return out,idwt_out,out,hh_in
        return out,hh_in
        # return idwt_out,hh_in


class dwt_up(nn.Module):
	def __init__(self, in_channels, with_hh=False):
		super().__init__()
		self.dim = in_channels
		self.idwt = iwt_m(requires_grad=True)
		# self.input_conv  = nn.Conv2d(in_channels*4, in_channels*4, kernel_size=3,padding =1)
		self.hi_sub_pre = nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=3, padding=1)

	def forward(self, X, hh_in):
		hh_in = hh_in + self.hi_sub_pre(hh_in)
		idw_in = torch.cat((X, hh_in), dim=1)[:, [0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11]]
		# idwt_out = self.idwt(self.input_conv(idw_in))
		idwt_out = self.idwt(idw_in)
		return idwt_out, hh_in


class ConvLayer(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
		super().__init__()
		self.dim = dim

		self.net_depth = net_depth
		self.kernel_size = kernel_size

		self.Wv = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			# nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect')
			nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
		)

		self.Wg = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
		)

		self.proj = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			nn.BatchNorm2d(dim)
		)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.net_depth) ** (-1/4)    # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, X):
		out = self.Wv(X) * self.Wg(X)
		out = self.proj(out)
		return out


class ConvLayer_csp(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3,csp_rate = 0.5, gate_act=nn.Sigmoid):
		super().__init__()
		self.dim = dim
		self.csp_rate = round(csp_rate*dim)

		self.net_depth = net_depth
		self.kernel_size = kernel_size

		self.Wv = nn.Sequential(
			nn.Conv2d(self.csp_rate, self.csp_rate, 1),
			# nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect')
			nn.Conv2d(self.csp_rate, self.csp_rate, kernel_size=kernel_size, padding=kernel_size//2, groups=self.csp_rate)
		)

		self.Wg = nn.Sequential(
			nn.Conv2d(self.csp_rate, self.csp_rate, 1),
			gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
		)

		self.fusion = nn.Conv2d(dim, dim, 1)

		self.proj = nn.Conv2d(self.csp_rate, self.csp_rate, 1)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.net_depth) ** (-1/4)    # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, X):
		part1, part2 = X[:,:self.csp_rate],X[:,self.csp_rate:]
		out_part1 = self.proj(self.Wv(part1) * self.Wg(part1))
		out = self.fusion(torch.cat([out_part1, part2], dim=1))
		return out


class BasicBlock(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):
		super().__init__()
		self.norm = norm_layer(dim)
		self.conv = conv_layer(net_depth, dim, kernel_size=kernel_size, gate_act=gate_act)
	def forward(self, x):
		identity = x
		x = self.norm(x)
		x = self.conv(x)
		x = identity + x
		return x


class BasicLayer(nn.Module):
	def __init__(self, net_depth, dim, depth, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):

		super().__init__()
		self.dim = dim
		self.depth = depth

		# build blocks
		self.blocks = nn.ModuleList([
			BasicBlock(net_depth, dim, kernel_size, conv_layer, norm_layer, gate_act)
			for i in range(depth)])

	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x


class resmspblock_sp_v1(nn.Module):
    def __init__(self, in_channels, out_channels,a=0.5):
        super().__init__()
        self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.channel1 = round(out_channels*a)
        self.channel2 = round(out_channels*(a**2))
        self.channel3 = round(out_channels * (a ** 3))
        # self.sort_arg = nn.Parameter(torch.tensor(range(out_channels)),requires_grad=False)
        # self.sort_arg2 = nn.Parameter(torch.tensor(range(self.channel1)),requires_grad=False)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.sort_tensor = torch.zeros((out_channels))
        # self.sort_tensor2 = torch.zeros((self.channel1))

        self.block1 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, padding=1,groups=self.channel1),
            # nn.Conv2d(self.channel1, self.channel1, 3, padding=1),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()
        )
        self.block3_1 = nn.Sequential(
            nn.Conv2d(self.channel3, self.channel3, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel3),
            nn.ReLU()
        )

        self.block3_2 = nn.Sequential(
            nn.Conv2d(self.channel3, self.channel3, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel3),
            nn.ReLU()
        )

        # self.block2_fu = nn.Sequential(
        #     nn.Conv2d(self.channel1, self.channel1, 1),
        #     nn.BatchNorm2d(self.channel1),
        #     nn.ReLU()
        # )

        self.block4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # self.block4 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU()
        # )
        # self.strip_att = cubic_attention(out_channels, group=1, dilation=1, kernel=3)
        self.strip_att = cubic_attention_2(self.channel1, kernel=3)

        self.fushion = nn.Sequential(
            nn.Conv2d((out_channels - self.channel1) + self.channel1*2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = self.resconv(x)
        B,C,H,W = res.shape
        # device = res.device
        # self.sort_tensor = self.sort_tensor.to(device)
        # self.sort_tensor = self.sort_tensor + torch.mean(self.gap(res).reshape((B, C)),dim=0)
        # res = res[:,self.sort_arg]
        part1,part2 = res[:,:self.channel1],res[:,self.channel1:]
        out1 = self.block1(part1)
        # self.sort_tensor2 = self.sort_tensor2.to(device)
        # self.sort_tensor2 = self.sort_tensor2 + torch.mean(self.gap(out1).reshape(out1.shape[:2]), dim=0)
        # out1 = out1[:, self.sort_arg2]
        out1_1,out1_2 = out1[:,:self.channel2],out1[:,self.channel2:]

        out1_1 = self.block2(out1_1)
        # out2 = self.block2_fu(torch.concat((out1_1,out1_2),dim=1))
        out1_1_1,out1_1_2 = out1_1[:,:self.channel3],out1_1[:,self.channel3:]
        out1_1_1 = self.block3_1(out1_1_1)
        out1_1_2 = self.block3_2(out1_1_2)
        out2 = torch.concat((out1_1_1,out1_1_2,out1_2),dim=1)


        out2_res = torch.concat([out2,part2],dim=1)
        out3 = self.block4(out2_res)

        out3_part1,out3_part2 = out3[:,:self.channel1],out3[:,self.channel1:]
        out4 = self.fushion(torch.cat((self.strip_att(out3_part1),out3_part2),dim=1))
        # out4 = self.fushion(torch.cat((self.strip_att(out3_part1),self.block4(out3_part2)),dim=1))

        #
        # return out4 + out3
        return out4
        # return out3


class cubic_attention_2(nn.Module):
    def __init__(self, dim, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att_2(dim, kernel=kernel)
        self.W_spatial_att = spatial_strip_att_2(dim, kernel=kernel, H=False)
        # self.fushion = nn.Sequential(
        #     nn.Conv2d(dim*2, dim, 1),
        #     nn.BatchNorm2d(dim),
        # )
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):
        out1 = self.H_spatial_att(x)
        out2 = self.W_spatial_att(x)
        out = torch.concat((out1,out2),dim=1)
        # out = self.fushion(out)


        # return self.gamma * out1 + out2 * self.beta
        return out


class spatial_strip_att_2(nn.Module):
    def __init__(self, dim, kernel=3, H=True) -> None:
        super().__init__()

        self.k = kernel
        self.dim = dim
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        pad = (kernel - 1) // 2
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel,groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.act = nn.ReLU()
        # self.act = nn.Identity()
        self.act2 = nn.Sigmoid()
        # self.act2 = nn.Tanh()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(dim, (dim // 2))
        self.fc2 = nn.Linear((dim // 2), dim)

    def forward(self, x):
        out = self.block(self.pad(x))
        # SE block part
        w = self.global_pool(out)
        # permute to linear shape
        # (batch, channels, H, W) --> (batch, H, W, channels)
        w = w.permute(0, 2, 3, 1)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.act2(w)
        # recover to (batch, channels, H, W)
        w = w.permute(0, 3, 1, 2)

        out = out * w

        return out


class BasicLayer_wv(nn.Module):
	def __init__(self, net_depth, dim, depth, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):

		super().__init__()
		self.dim = dim
		self.depth = depth

		# build blocks
		self.blocks = nn.ModuleList([
			resmspblock_sp_v1(dim, dim)
			for i in range(depth)])

	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x
class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

		# self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
		# 					  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')
		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size - patch_size + 1) // 2)

	def forward(self, x):
		x = self.proj(x)
		return x


class PatchUnEmbed(nn.Module):
	def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.out_chans = out_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = 1

		self.proj = nn.Sequential(
			# nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
			# 		  padding=kernel_size//2, padding_mode='reflect'),
			nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
					  padding=kernel_size // 2),
			nn.PixelShuffle(patch_size)
		)

	def forward(self, x):
		x = self.proj(x)
		return x


class SKFusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SKFusion, self).__init__()

		self.height = height
		d = max(int(dim/reduction), 4)

		self.mlp = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(dim, d, 1, bias=False),
			nn.ReLU(True),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape

		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)

		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(feats_sum)
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out


class gUNet(nn.Module):
	def __init__(self, in_channel = 3,out_channel = 3,kernel_size=5, base_dim=32, depths=[4, 4, 4, 4, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion):
		super(gUNet, self).__init__()
		# setting
		assert len(depths) % 2 == 1
		stage_num = len(depths)
		half_num = stage_num // 2
		net_depth = sum(depths)
		embed_dims = [2**i*base_dim for i in range(half_num)]
		embed_dims = embed_dims + [2**half_num*base_dim] + embed_dims[::-1]

		self.patch_size = 2 ** (stage_num // 2)
		self.stage_num = stage_num
		self.half_num = half_num

		# input convolution
		self.inconv = PatchEmbed(patch_size=1, in_chans=in_channel, embed_dim=embed_dims[0], kernel_size=3)

		# backbone
		self.layers = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.skips = nn.ModuleList()
		self.fusions = nn.ModuleList()

		for i in range(self.stage_num):
			self.layers.append(BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
										  conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

		for i in range(self.half_num):
			self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
			self.fusions.append(fusion_layer(embed_dims[i]))

		# output convolution
		self.outconv = PatchUnEmbed(patch_size=1, out_chans=out_channel, embed_dim=embed_dims[-1], kernel_size=3)


	def forward(self, x):
		feat = self.inconv(x)

		skips = []

		for i in range(self.half_num):
			feat = self.layers[i](feat)
			skips.append(self.skips[i](feat))
			feat = self.downs[i](feat)

		feat = self.layers[self.half_num](feat)

		for i in range(self.half_num-1, -1, -1):
			feat = self.ups[i](feat)
			feat = self.fusions[i]([feat, skips[i]])
			feat = self.layers[self.stage_num-i-1](feat)

		# x = self.outconv(feat) + x
		x = self.outconv(feat)

		return x

class gUNet_custom(nn.Module):
	def __init__(self, in_channel = 3,out_channel = 3,kernel_size=5, base_dim=[24, 48, 96, 192, 96, 48, 24], depths=[4, 4, 4, 4, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion):
		super(gUNet_custom, self).__init__()
		# setting
		assert len(depths) % 2 == 1
		assert len(depths) == len(base_dim)
		stage_num = len(depths)
		half_num = stage_num // 2
		net_depth = sum(depths)
		embed_dims = base_dim
		self.patch_size = 2 ** (stage_num // 2)
		self.stage_num = stage_num
		self.half_num = half_num

		# input convolution
		self.inconv = PatchEmbed(patch_size=1, in_chans=in_channel, embed_dim=embed_dims[0], kernel_size=3)

		# backbone
		self.layers = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.skips = nn.ModuleList()
		self.fusions = nn.ModuleList()

		for i in range(self.stage_num):
			self.layers.append(BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
										  conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

		for i in range(self.half_num):
			self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
			self.fusions.append(fusion_layer(embed_dims[i]))

		# output convolution
		self.outconv = PatchUnEmbed(patch_size=1, out_chans=out_channel, embed_dim=embed_dims[-1], kernel_size=3)


	def forward(self, x):
		feat = self.inconv(x)

		skips = []

		for i in range(self.half_num):
			feat = self.layers[i](feat)
			skips.append(self.skips[i](feat))
			feat = self.downs[i](feat)

		feat = self.layers[self.half_num](feat)

		for i in range(self.half_num-1, -1, -1):
			feat = self.ups[i](feat)
			feat = self.fusions[i]([feat, skips[i]])
			feat = self.layers[self.stage_num-i-1](feat)

		# x = self.outconv(feat) + x
		x = self.outconv(feat)

		return x

class wv_UNet(nn.Module):
	def __init__(self, in_channel = 3,out_channel = 3,kernel_size=5, base_dim=32, depths=[4, 4, 4, 4, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion):
		super(wv_UNet, self).__init__()
		# setting
		assert len(depths) % 2 == 1
		stage_num = len(depths)
		half_num = stage_num // 2
		net_depth = sum(depths)
		embed_dims = [2**i*base_dim for i in range(half_num)]
		embed_dims = embed_dims + [2**half_num*base_dim] + embed_dims[::-1]

		self.patch_size = 2 ** (stage_num // 2)
		self.stage_num = stage_num
		self.half_num = half_num

		# input convolution
		self.inconv = PatchEmbed(patch_size=1, in_chans=in_channel, embed_dim=embed_dims[0], kernel_size=3)

		# backbone
		self.layers = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.skips = nn.ModuleList()
		self.fusions = nn.ModuleList()
		shuffer_dim = (embed_dims[0]//3 + embed_dims[1]//3 + embed_dims[2]//3)

		for i in range(self.stage_num):
			self.layers.append(BasicLayer_wv(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
										  conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

		for i in range(self.half_num):
			self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			# self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
			self.skips.append(nn.Conv2d(shuffer_dim, embed_dims[i], 1))
			self.fusions.append(fusion_layer(embed_dims[i]))

		# output convolution
		self.outconv = PatchUnEmbed(patch_size=1, out_chans=out_channel, embed_dim=embed_dims[-1], kernel_size=3)

	def shuffle(self, levels, mode='bilinear'):
		level_num = len(levels)
		out_levels = []
		for i in range(level_num):
			channels = levels[i].shape[1] // level_num
			out_level = []
			for j in range(level_num):
				l = levels[i][:, channels * j:channels * (j + 1), :, :]
				out_level.append(F.interpolate(l, scale_factor=float(1 / (2 ** (j - i))), mode=mode))
			out_levels.append(out_level)
		output = []
		for i in range(level_num):
			output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
		return output

	def forward(self, x):
		feat = self.inconv(x)

		skips = []

		for i in range(self.half_num):
			feat = self.layers[i](feat)
			# skips.append(self.skips[i](feat))
			skips.append(feat)
			feat = self.downs[i](feat)
		skips = self.shuffle(skips)

		for i in range(self.half_num):
			skips[i] = self.skips[i](skips[i])

		feat = self.layers[self.half_num](feat)

		for i in range(self.half_num-1, -1, -1):
			feat = self.ups[i](feat)
			feat = self.fusions[i]([feat, skips[i]])
			feat = self.layers[self.stage_num-i-1](feat)

		# x = self.outconv(feat) + x
		x = self.outconv(feat)

		return x


__all__ = ['gUNet', "gunet_ss",'gunet_t', 'gunet_s', 'gunet_b', 'gunet_d','wavelet','wavelet_gnet','wavedown_gnet_two','wavelet_gnet_two','wavelet_gnet_two_csp','wavelet_gnet_two_endep','wavelet_gnet_two_lite','wavelet_gnet_two_weight','wavelet_wvnet_two','wavelet_gnet_three','wavelet_gnet_three_w',"wavelet_gnet_three_endep","ll_predict","ll_predict_lite"]

# Normalization batch size of 16~32 may be good
def gunet_ss():	# 4 cards 2080Ti
	return gUNet(kernel_size=3, base_dim=24, depths=[1, 1, 1, 2, 1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)


def gunet_t():	# 4 cards 2080Ti
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_s():	# 4 cards 3090
	return gUNet(kernel_size=5, base_dim=24, depths=[4, 4, 4, 8, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_b():	# 4 cards 3090
	return gUNet(kernel_size=5, base_dim=24, depths=[8, 8, 8, 16, 8, 8, 8], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_d():	# 4 cards 3090
	return gUNet(kernel_size=5, base_dim=24, depths=[16, 16, 16, 32, 16, 16, 16], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def wavelet(model=gunet_t()):
	return UNet_wavelet(model)

def wavelet_gnet():
	return UNet_wavelet_gnet()

def wavelet_gnet_two():
	return UNet_wavelet_gnet_two()

def wavedown_gnet_two():
	return UNet_wavedown_gnet_two()

def wavelet_gnet_two_csp():
	return UNet_wavelet_gnet_two_csp()

def wavelet_gnet_two_weight():
	return UNet_wavelet_gnet_two(base_dim=24, depths=[2, 2, 2,4, 2, 2,2])

def wavelet_gnet_two_lite():
	return UNet_wavelet_gnet_two(base_dim=24, depths=[1, 1, 2, 1, 1])

def wavelet_wvnet_two():
	return WVNet_wavelet_gnet_two()


def wavelet_gnet_three():
	return UNet_wavelet_gnet_three()

def wavelet_gnet_three_endep():
	return UNet_wavelet_gnet_three_endep()
def wavelet_gnet_two_endep():
	return UNet_wavelet_gnet_two_endep()


def wavelet_gnet_three_w():
	return UNet_wavelet_gnet_three(base_dim=36)

def ll_predict(model):
	return ll_predict_model(model)

def ll_predict_lite(model):
	return ll_predict_model_lite(model)


class UNet_wavelet_gnet(nn.Module):
	def __init__(self):
		super(UNet_wavelet_gnet, self).__init__()
		self.xfm = wt_m(requires_grad=True)
		self.ifm = iwt_m(requires_grad=True)
		self.out = False
		self.encoder = gUNet(in_channel=12,out_channel=12,kernel_size=5, base_dim=24, depths=[1, 1, 1, 2, 1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
	def forward(self,x):
		x = self.xfm(x)
		x = self.encoder(x)
		out_ll = x[:, [0, 4, 8]]
		out_detail = x[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		output_map = self.ifm(x)
		# recon_R = self.ifm(torch.cat((out_ll[:, [0]], out_detail[:, 0:3]), dim=1))
		# recon_R = torch.clamp(recon_R, min=-1, max=1)
		# recon_G = self.ifm(torch.cat((out_ll[:, [1]], out_detail[:, 3:6]), dim=1))
		# recon_G = torch.clamp(recon_G, min=-1, max=1)
		# recon_B = self.ifm(torch.cat((out_ll[:, [2]], out_detail[:, 6:9]), dim=1))
		# recon_B = torch.clamp(recon_B, min=-1, max=1)
		# output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
		# return output_map,(out_ll, out_detail)
		if self.out:
			return output_map
		else:
			return [output_map, [[out_ll, out_detail]]]

class UNet_wavelet_gnet_two(nn.Module):
	def __init__(self,kernel_size=3, base_dim=24, depths=[1, 1, 1, 2, 1, 1, 1]):
		super(UNet_wavelet_gnet_two, self).__init__()
		self.out = False
		self.xfm = wt_m(requires_grad=True)
		self.ifm = iwt_m(requires_grad=True)
		self.encoder = gUNet(in_channel=48,out_channel=48,kernel_size=kernel_size, base_dim=base_dim, depths=depths, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
		# self.encoder = gUNet_custom(in_channel=48,out_channel=48,kernel_size=5, base_dim=[12,24,24,48,24,24,12], depths=[1, 1, 1, 1,1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
	def forward(self,x):
		x = self.xfm(x)
		x = self.xfm(x)
		out = self.encoder(x)
		# out_ll = x[:, [0, 4, 8]]
		# out_detail = x[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		out = self.ifm(out)
		ll = out[:, [0, 4, 8]]
		details = out[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		# recon_R = torch.clamp(recon_R, min=-1, max=1)
		# recon_G = self.ifm(torch.cat((out_ll[:, [1]], out_detail[:, 3:6]), dim=1))
		# recon_G = torch.clamp(recon_G, min=-1, max=1)
		# recon_B = self.ifm(torch.cat((out_ll[:, [2]], out_detail[:, 6:9]), dim=1))
		# recon_B = torch.clamp(recon_B, min=-1, max=1)
		# output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
		# return output_map,out_ll,out_detail
		outmap = self.ifm(out)
		if self.out:
			return outmap
		else:
			return [outmap,[[ll, details]]]


class UNet_wavedown_gnet_two(nn.Module):
	def __init__(self,kernel_size=3, base_dim=24, depths=[1, 1, 1, 2, 1, 1, 1]):
		super(UNet_wavedown_gnet_two, self).__init__()
		self.out = True
		# self.dwt_down1 = dwt_down(dim=3)
		self.dwt_down1 = WaveDownampler(in_channels=3)
		# self.dwt_down2 = dwt_down(dim=3,with_hh=True)
		self.dwt_down2 = WaveDownampler(in_channels=3)
		# self.dwt_up1 = dwt_up(3)
		self.dwt_up1 = WaveUpsampler(3)
		self.dwt_up2 = WaveUpsampler(3)
		# self.dwt_up2 = dwt_up(3)
		self.encoder = gUNet(in_channel=3,out_channel=3,kernel_size=kernel_size, base_dim=base_dim, depths=depths, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
		# self.encoder = gUNet_custom(in_channel=48,out_channel=48,kernel_size=5, base_dim=[12,24,24,48,24,24,12], depths=[1, 1, 1, 1,1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
	def forward(self,x):
		x,hh_1 = self.dwt_down1(x)
		x,hh_2 = self.dwt_down2(x)
		out = self.encoder(x)
		out_ll = self.dwt_up2(out,hh_2)
		# out_ll,out_hh = self.dwt_up2(out_ll,out_hh)
		outmap = self.dwt_up1(out_ll,hh_1)
		if self.out:
			return outmap
		else:
			return [outmap,[[ll, details]]]

class UNet_wavelet_gnet_two_csp(nn.Module):
	def __init__(self,kernel_size=3, base_dim=24, depths=[1, 1, 1, 2, 1, 1, 1]):
		super(UNet_wavelet_gnet_two_csp, self).__init__()
		self.out = False
		self.xfm = wt_m(requires_grad=True)
		self.ifm = iwt_m(requires_grad=True)
		self.encoder = gUNet(in_channel=48,out_channel=48,kernel_size=kernel_size, base_dim=base_dim, depths=depths, conv_layer=ConvLayer_csp, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
		# self.encoder = gUNet_custom(in_channel=48,out_channel=48,kernel_size=5, base_dim=[12,24,24,48,24,24,12], depths=[1, 1, 1, 1,1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
	def forward(self,x):
		x = self.xfm(x)
		x = self.xfm(x)
		out = self.encoder(x)
		# out_ll = x[:, [0, 4, 8]]
		# out_detail = x[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		out = self.ifm(out)
		ll = out[:, [0, 4, 8]]
		details = out[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		# recon_R = torch.clamp(recon_R, min=-1, max=1)
		# recon_G = self.ifm(torch.cat((out_ll[:, [1]], out_detail[:, 3:6]), dim=1))
		# recon_G = torch.clamp(recon_G, min=-1, max=1)
		# recon_B = self.ifm(torch.cat((out_ll[:, [2]], out_detail[:, 6:9]), dim=1))
		# recon_B = torch.clamp(recon_B, min=-1, max=1)
		# output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
		# return output_map,out_ll,out_detail
		outmap = self.ifm(out)
		if self.out:
			return outmap
		else:
			return [outmap, [[ll, details]]]

class WVNet_wavelet_gnet_two(nn.Module):
	def __init__(self):
		super(WVNet_wavelet_gnet_two, self).__init__()
		self.out = False
		self.xfm = wt_m(requires_grad=True)
		self.ifm = iwt_m(requires_grad=True)
		self.encoder = wv_UNet(in_channel=48,out_channel=48,kernel_size=5, base_dim=24, depths=[1, 1, 1, 2, 1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
		# self.encoder = gUNet_custom(in_channel=48,out_channel=48,kernel_size=5, base_dim=[12,24,24,48,24,24,12], depths=[1, 1, 1, 1,1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
	def forward(self,x):
		x = self.xfm(x)
		x = self.xfm(x)
		out = self.encoder(x)
		# out_ll = x[:, [0, 4, 8]]
		# out_detail = x[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		out = self.ifm(out)
		ll = out[:, [0, 4, 8]]
		details = out[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		outmap = self.ifm(out)
		# recon_R = torch.clamp(recon_R, min=-1, max=1)
		# recon_G = self.ifm(torch.cat((out_ll[:, [1]], out_detail[:, 3:6]), dim=1))
		# recon_G = torch.clamp(recon_G, min=-1, max=1)
		# recon_B = self.ifm(torch.cat((out_ll[:, [2]], out_detail[:, 6:9]), dim=1))
		# recon_B = torch.clamp(recon_B, min=-1, max=1)
		# output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
		# return output_map,out_ll,out_detail
		# return outmap
		if self.out:
			return outmap
		else:
			return [outmap, [[ll, details]]]



class UNet_wavelet_gnet_three(nn.Module):
	def __init__(self,base_dim=24):
		super(UNet_wavelet_gnet_three, self).__init__()
		self.xfm = wt_m(requires_grad=True)
		self.ifm = iwt_m(requires_grad=True)
		self.out = False
		self.encoder = gUNet(in_channel=192,out_channel=192,kernel_size=5, base_dim=base_dim, depths=[1, 1, 1, 2, 1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
		# self.encoder = self.encoder = gUNet_custom(in_channel=192,out_channel=192,kernel_size=5, base_dim=[12,24,24,32,24,24,12], depths=[1, 1, 1, 1, 1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
	def forward(self,x):
		x = self.xfm(x)
		x = self.xfm(x)
		x = self.xfm(x)
		out = self.encoder(x)
		# out_ll = x[:, [0, 4, 8]]
		# out_detail = x[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		out = self.ifm(out)
		out = self.ifm(out)
		ll = out[:, [0, 4, 8]]
		details = out[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		outmap = self.ifm(out)
		# recon_R = torch.clamp(recon_R, min=-1, max=1)
		# recon_G = self.ifm(torch.cat((out_ll[:, [1]], out_detail[:, 3:6]), dim=1))
		# recon_G = torch.clamp(recon_G, min=-1, max=1)
		# recon_B = self.ifm(torch.cat((out_ll[:, [2]], out_detail[:, 6:9]), dim=1))
		# recon_B = torch.clamp(recon_B, min=-1, max=1)
		# output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
		# return output_map,out_ll,out_detail
		if self.out:
			return outmap
		else:
			return [outmap, [[ll, details]]]


class UNet_wavelet_gnet_three_endep(nn.Module):
	def __init__(self,base_dim=24):
		super(UNet_wavelet_gnet_three_endep, self).__init__()
		self.out = False
		self.xfm = wt_m(requires_grad=True)
		self.ifm = iwt_m(requires_grad=True)
		self.ll_encoder = gUNet(in_channel=48,out_channel=48,kernel_size=5, base_dim=base_dim, depths=[1, 1, 1, 2, 1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
		self.detail_encoder = gUNet(in_channel=144, out_channel=144, kernel_size=5, base_dim=base_dim,
								depths=[1, 1, 1, 2, 1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d,
								gate_act=nn.Sigmoid, fusion_layer=SKFusion)
		# self.encoder = self.encoder = gUNet_custom(in_channel=192,out_channel=192,kernel_size=5, base_dim=[12,24,24,32,24,24,12], depths=[1, 1, 1, 1, 1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
	def forward(self,x):
		x = self.xfm(x)
		ll_in = x[:, [0, 4, 8]]
		detail_in = x[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		ll_in = self.xfm(ll_in)
		ll_in = self.xfm(ll_in)
		detail_in = self.xfm(detail_in)
		detail_in = self.xfm(detail_in)
		out_ll = self.ll_encoder(ll_in)
		out_detail = self.detail_encoder(detail_in)
		# out_ll = x[:, [0, 4, 8]]
		# out_detail = x[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		out_ll = self.ifm(out_ll)
		out_ll = self.ifm(out_ll)
		out_detail = self.ifm(out_detail)
		out_detail = self.ifm(out_detail)

		recon_R = self.ifm(torch.cat((out_ll[:, [0]], out_detail[:, 0:3]), dim=1))
		# recon_R = torch.clamp(recon_R, min=-1, max=1)
		recon_G = self.ifm(torch.cat((out_ll[:, [1]], out_detail[:, 3:6]), dim=1))
		# recon_G = torch.clamp(recon_G, min=-1, max=1)
		recon_B = self.ifm(torch.cat((out_ll[:, [2]], out_detail[:, 6:9]), dim=1))
		# recon_B = torch.clamp(recon_B, min=-1, max=1)
		outmap = torch.cat((recon_R, recon_G, recon_B), dim=1)
		# return output_map,out_ll,out_detail
		if self.out:
			return outmap
		else:
			return [outmap, [[out_ll, out_detail]]]


class UNet_wavelet_gnet_two_endep(nn.Module):
	def __init__(self,base_dim=24):
		super(UNet_wavelet_gnet_two_endep, self).__init__()
		self.out = False
		self.xfm = wt_m(requires_grad=False)
		self.ifm = iwt_m(requires_grad=False)
		self.ll_encoder = gUNet(in_channel=48,out_channel=12,kernel_size=3, base_dim=base_dim, depths=[1,1, 1,2,1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
		self.detail_encoder = gUNet(in_channel=48, out_channel=36, kernel_size=3, base_dim=base_dim,
								depths=[1, 1, 2, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d,
								gate_act=nn.Sigmoid, fusion_layer=SKFusion)
		# self.encoder = self.encoder = gUNet_custom(in_channel=192,out_channel=192,kernel_size=5, base_dim=[12,24,24,32,24,24,12], depths=[1, 1, 1, 1, 1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
	def forward(self,x):
		x = self.xfm(x)
		x = self.xfm(x)
		out_ll = self.ll_encoder(x)
		out_detail = self.detail_encoder(x)
		# out_ll = x[:, [0, 4, 8]]
		# out_detail = x[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		out_ll = self.ifm(out_ll)
		out_detail = self.ifm(out_detail)
		recon_R = self.ifm(torch.cat((out_ll[:, [0]], out_detail[:, 0:3]), dim=1))
		# recon_R = torch.clamp(recon_R, min=-1, max=1)
		recon_G = self.ifm(torch.cat((out_ll[:, [1]], out_detail[:, 3:6]), dim=1))
		# recon_G = torch.clamp(recon_G, min=-1, max=1)
		recon_B = self.ifm(torch.cat((out_ll[:, [2]], out_detail[:, 6:9]), dim=1))
		# recon_B = torch.clamp(recon_B, min=-1, max=1)
		outmap = torch.cat((recon_R, recon_G, recon_B), dim=1)
		# return output_map,out_ll,out_detail
		if self.out:
			return outmap
		else:
			return [outmap, [[out_ll, out_detail]]]
		# return outmap
class UNet_wavelet(nn.Module):
	def __init__(self,model=gunet_t()):
		super(UNet_wavelet, self).__init__()
		self.out = False
		self.xfm = wt_m(requires_grad=True)
		self.ifm = iwt_m(requires_grad=True)
		self.ll_encoder = ll_predict_model(model)
		self.detail_encoder = gUNet(in_channel=9,out_channel=9,kernel_size=5, base_dim=24, depths=[1, 1, 1, 2, 1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
	def forward(self,x):
		x = self.xfm(x)
		ll = x[:, [0, 4, 8]]
		detail = x[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
		out_ll = self.ll_encoder(x)
		out_detail = self.detail_encoder(detail)
		recon_R = self.ifm(torch.cat((out_ll[:, [0]], out_detail[:, 0:3]), dim=1))
		recon_R = torch.clamp(recon_R, min=-1, max=1)
		recon_G = self.ifm(torch.cat((out_ll[:, [1]], out_detail[:, 3:6]), dim=1))
		recon_G = torch.clamp(recon_G, min=-1, max=1)
		recon_B = self.ifm(torch.cat((out_ll[:, [2]], out_detail[:, 6:9]), dim=1))
		recon_B = torch.clamp(recon_B, min=-1, max=1)
		output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
		if self.out:
			return outmap
		else:
			return [outmap, [[out_ll, out_detail]]]

	def set_train(self):
		self.train()
		self.ll_encoder.set_train()

class ll_predict_model(nn.Module):
	def __init__(self, model):
		super(ll_predict_model, self).__init__()
		self.ll_predict_model = model
		self.dwt_down1 = WaveDownampler(in_channels=3)
		# self.dwt_down2 = dwt_down(dim=3,with_hh=True)
		self.dwt_down2 = WaveDownampler(in_channels=3)
		# self.dwt_up1 = dwt_up(3)
		self.dwt_up1 = WaveUpsampler(3)
		self.dwt_up2 = WaveUpsampler(3)
		self.train_all = True
		self.out = False

	def forward(self, x):
		x_1,hh_1 = self.dwt_down1(x)
		x_2,hh_2 = self.dwt_down2(x_1)
		x_3 = self.ll_predict_model(x_2)
		out_ll,detail = self.dwt_up2(x_3,x_1,hh_2)
		out,out_detail = self.dwt_up1(out_ll,x,hh_1)
		# x = self.end_embedding(x)
		# return [out,[[out_ll,out_detail]]]
		if self.out:
			return out
		else:
			return [out,[[out_ll,out_detail],[x_3,detail]]]

	def train(self, mode=True):
		super().train(mode)
		if not self.train_all:
			for param in self.ll_predict_model.parameters():
				param.requires_grad = False


class ll_predict_model_lite(nn.Module):
	def __init__(self, model):
		super(ll_predict_model_lite, self).__init__()
		self.ll_predict_model = model
		self.dwt_down1 = dwt_down(dim=3)
		self.dwt_down2 = dwt_down(dim=3)
		self.dwt_up1 = dwt_up(3)
		self.dwt_up2 = dwt_up(3)
		self.train_all = True
		self.out = False

	def forward(self, x):
		x_1,hh_1 = self.dwt_down1(x)
		x_2,hh_2 = self.dwt_down2(x_1)
		x_3 = self.ll_predict_model(x_2)
		out_ll,detail = self.dwt_up2(x_3,hh_2)
		out,out_detail = self.dwt_up1(out_ll,hh_1)
		# x = self.end_embedding(x)
		# return [out,[[out_ll,out_detail]]]
		if self.out:
			return out
		else:
			return [out,[[out_ll,out_detail],[x_3,detail]]]

	def train(self, mode=True):
		super().train(mode)
		if not self.train_all:
			for param in self.ll_predict_model.parameters():
				param.requires_grad = False

