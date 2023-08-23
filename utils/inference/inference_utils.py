'''
Adapted from R. Strudel et al.
https://github.com/rstrudel/segmenter

MIT License
Copyright (c) 2021 Robin Strudel
Copyright (c) INRIA
'''

import torch
import torch.nn.functional as F


def resize(im, smaller_size):
    h, w = im.shape[2:]
    if h < w:
        ratio = w / h
        h_res, w_res = smaller_size, ratio * smaller_size
    else:
        ratio = h / w
        h_res, w_res = ratio * smaller_size, smaller_size
    if min(h, w) < smaller_size:
        im_res = F.interpolate(im, (int(h_res), int(w_res)), mode='bilinear')
    else:
        im_res = im
    return im_res


def sliding_window(im, flip, window_size, window_stride):
    B, C, H, W = im.shape #inference: 32, 2048
    ws_h, ws_w = window_size # 32, 384
    
    windows = {'crop': [], 'anchors': []} #crop: 실제 window 사이즈의 
    h_anchors = torch.arange(0, H, window_stride[0])
    w_anchors = torch.arange(0, W, window_stride[1])
    h_anchors = [h.item() for h in h_anchors if h < H - ws_h] + [H - ws_h] #리스트 마지막에 (H-ws_h 추가) [0]
    w_anchors = [w.item() for w in w_anchors if w < W - ws_w] + [W - ws_w] #[0, 256, 512, 768, 1024, 1280, 1536, 1664]
    for ha in h_anchors:
        for wa in w_anchors:
            window = im[:, :, ha : ha + ws_h, wa : wa + ws_w] #[B,C, 32, 384]
            windows['crop'].append(window)
            windows['anchors'].append((ha, wa))
    windows['flip'] = flip
    windows['shape'] = (H, W)
    return windows


def merge_windows(windows, window_size, ori_shape):
    ws_h, ws_w = window_size
    im_windows = windows['seg_maps']
    anchors = windows['anchors']
    C = im_windows[0].shape[0]
    H, W = windows['shape']
    flip = windows['flip']

    logit = torch.zeros((C, H, W), device=im_windows.device)
    count = torch.zeros((1, H, W), device=im_windows.device)
    for window, (ha, wa) in zip(im_windows, anchors):
        logit[:, ha : ha + ws_h, wa : wa + ws_w] += window
        count[:, ha : ha + ws_h, wa : wa + ws_w] += 1
    logit = logit / count
    logit = F.interpolate( #비어있을수도 있어서...?
        logit.unsqueeze(0),
        ori_shape,
        mode='bilinear')[0]
    
    if flip:
        logit = torch.flip(logit, (2,))
    return logit


def inference(
    model,
    ims,
    ims_metas,
    ori_shape,
    window_size,
    window_stride,
    batch_size,
    use_kpconv=False):

    # window_size and window_stride have to be tuples or lists with two elements
    assert len(window_size) == len(window_stride) == 2

    wsize_h, wsize_w = window_size
    smaller_size = wsize_h if wsize_h < wsize_w else wsize_w

    seg_map = None
    for im, im_metas in zip(ims, ims_metas): #ims, ims_metas를 동시에 순환, 이때 ims_metas는  [{'flip': False}]이므로 im_metas도 계속  [{'flip': False}]
        im = resize(im, smaller_size) # window 보다 작으면 더 키워서 window 사이즈에 맞춤(bilinear)
        flip = im_metas['flip']
        windows = sliding_window(im, flip, window_size, window_stride)
        crops = torch.stack(windows.pop('crop'))[:, 0] # crop: [8,B(=1),C,32,384] -> [8,C,32,384] 
        
        with torch.no_grad():
            if use_kpconv:
                seg_maps = model.forward_2d_features(crops) # shape = [n_windows, d_decoder, wsize_h, wsize_w]
            else:
                seg_maps = model.forward(crops) # shape = [n_windows, n_classes, wsize_h, wsize_w]
        windows['seg_maps'] = seg_maps
        im_seg_map = merge_windows(windows, window_size, ori_shape) # shape = [n_classes or d_decoder, ori_shape[0], ori_shape[1]]

        if seg_map is None:
            seg_map = im_seg_map
        else:
            seg_map += im_seg_map
    seg_map /= len(ims)
    return seg_map
