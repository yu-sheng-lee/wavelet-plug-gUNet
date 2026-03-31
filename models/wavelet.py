# -*- coding: utf-8 -*-
import pywt
import torch
from torch.autograd import Variable
import numpy as np
import math

def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.
    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to
    Returns:
        (h0_col, h1_col, h0_row, h1_row)
    """
    h0_col = np.array(h0_col[::-1]).ravel()
    h1_col = np.array(h1_col[::-1]).ravel()
    t = torch.get_default_dtype()
    if h0_row is None:
        h0_row = h0_col
    else:
        h0_row = np.array(h0_row[::-1]).ravel()
    if h1_row is None:
        h1_row = h1_col
    else:
        h1_row = np.array(h1_row[::-1]).ravel()
    h0_col = torch.tensor(h0_col, device=device, dtype=t).reshape((1, 1, -1, 1))
    h1_col = torch.tensor(h1_col, device=device, dtype=t).reshape((1, 1, -1, 1))
    h0_row = torch.tensor(h0_row, device=device, dtype=t).reshape((1, 1, 1, -1))
    h1_row = torch.tensor(h1_row, device=device, dtype=t).reshape((1, 1, 1, -1))

    return h0_col, h1_col, h0_row, h1_row

def prep_filt_sfb2d(g0_col, g1_col, g0_row=None, g1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the sfb2d function.  In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.
    Inputs:
        g0_col (array-like): low pass column filter bank
        g1_col (array-like): high pass column filter bank
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to
    Returns:
        (g0_col, g1_col, g0_row, g1_row)
    """
    g0_col = np.array(g0_col).ravel()
    g1_col = np.array(g1_col).ravel()
    t = torch.get_default_dtype()
    if g0_row is None:
        g0_row = g0_col
    if g1_row is None:
        g1_row = g1_col
    g0_col = torch.tensor(g0_col, device=device, dtype=t).reshape((1, 1, -1, 1))
    g1_col = torch.tensor(g1_col, device=device, dtype=t).reshape((1, 1, -1, 1))
    g0_row = torch.tensor(g0_row, device=device, dtype=t).reshape((1, 1, 1, -1))
    g1_row = torch.tensor(g1_row, device=device, dtype=t).reshape((1, 1, 1, -1))

    return g0_col, g1_col, g0_row, g1_row


class wt_m(torch.nn.Module):
    def __init__(self,wave='db1',requires_grad= False):
        super(wt_m,self).__init__()
        # padded = vimg
        # res = torch.zeros(vimg.shape[0],4*vimg.shape[1],int(vimg.shape[2]/2),int(vimg.shape[3]/2))
        # res = res.cuda()
        w = pywt.Wavelet(wave)
        #
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        # print(dec_hi,dec_lo)

        filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
        self.filters = torch.nn.Parameter(filters[:, None], requires_grad=requires_grad)

    def forward(self,vimg):
        ls = []
        for i in range(vimg.shape[1]):
            ls.append((torch.nn.functional.conv2d(vimg[:, i:i+1],self.filters,stride=2))/2)
        return torch.cat(ls,dim=1)


class swt_m(torch.nn.Module):
    def __init__(self,wave='db1',requires_grad= False):
        super(swt_m,self).__init__()
        # padded = vimg
        # res = torch.zeros(vimg.shape[0],4*vimg.shape[1],int(vimg.shape[2]/2),int(vimg.shape[3]/2))
        # res = res.cuda()
        w = pywt.Wavelet(wave)
        #
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        # print(dec_hi,dec_lo)

        filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
        self.filters = torch.nn.Parameter(filters[:, None], requires_grad=requires_grad)

    def forward(self,vimg):
        ls = []
        for i in range(vimg.shape[1]):
            ls.append((torch.nn.functional.conv2d(vimg[:, i:i+1],self.filters,padding="same"))/2)
        return torch.cat(ls,dim=1)



class iwt_m(torch.nn.Module):
    def __init__(self,wave='db1',requires_grad= False):
        super(iwt_m,self).__init__()
        # padded = vimg
        # res = torch.zeros(vimg.shape[0],4*vimg.shape[1],int(vimg.shape[2]/2),int(vimg.shape[3]/2))
        # res = res.cuda()
        w = pywt.Wavelet(wave)
        #
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        inv_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
        self.filters = torch.nn.Parameter(inv_filters[:, None], requires_grad=requires_grad)

    def forward(self,vimg):
        ls = []
        for i in range(vimg.shape[1]//4):
            ls.append(torch.nn.functional.conv_transpose2d(2 * vimg[:, 4*i:4*i+4],self.filters,stride=2))
        return torch.cat(ls,dim=1)