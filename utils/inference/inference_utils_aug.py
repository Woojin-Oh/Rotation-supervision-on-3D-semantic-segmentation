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
    B, C, H, W = im.shape
    ws_h, ws_w = window_size
    
    windows = {'crop': [], 'anchors': []}
    h_anchors = torch.arange(0, H, window_stride[0])
    w_anchors = torch.arange(0, W, window_stride[1])
    h_anchors = [h.item() for h in h_anchors if h < H - ws_h] + [H - ws_h]
    w_anchors = [w.item() for w in w_anchors if w < W - ws_w] + [W - ws_w]
    for ha in h_anchors:
        for wa in w_anchors:
            window = im[:, :, ha : ha + ws_h, wa : wa + ws_w]
            windows['crop'].append(window)
            windows['anchors'].append((ha, wa))
    windows['flip'] = flip
    windows['shape'] = (H, W)
    return windows


def merge_windows_train(windows, window_size, ori_shape):
    ws_h, ws_w = window_size
    im_windows = windows['seg_maps'] #[4, 8, 128, 64, 384]
    anchors = windows['anchors']
    B,_,C,_,_ = im_windows.shape
    #B = im_windows[0].shape[0]
    #C = im_windows[0][0].shape[0]
    #C = im_windows[0].shape[0]
    H, W = windows['shape']
    flip = windows['flip']

    logit = torch.zeros((B,C, H, W), device=im_windows.device)
    count = torch.zeros((B,1, H, W), device=im_windows.device)
    #logit = torch.zeros((C, H, W), device=im_windows.device)
    #count = torch.zeros((1, H, W), device=im_windows.device)
    for i, batch_windows in enumerate(im_windows):
        for window, (ha, wa) in zip(batch_windows, anchors):
            logit[i,:, ha : ha + ws_h, wa : wa + ws_w] += window
            count[i,:, ha : ha + ws_h, wa : wa + ws_w] += 1
            #logit[:, ha : ha + ws_h, wa : wa + ws_w] += window
            #count[:, ha : ha + ws_h, wa : wa + ws_w] += 1
    #print('logit shape: ', logit.shape)
    logit = logit / count
    logit = F.interpolate(
        logit,
        ori_shape,
        mode='bilinear')
    
    if flip:
        logit = torch.flip(logit, (2,))
    #print('logit shape: ', logit.shape)
    return logit

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
    logit = F.interpolate(
        logit.unsqueeze(0),
        ori_shape,
        mode='bilinear')[0]
    
    if flip:
        logit = torch.flip(logit, (2,))
    return logit


def inference_train(
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
    #print('len ims: ', len(ims)) # 1
    for im, im_metas in zip(ims, ims_metas):
        im = resize(im, smaller_size)
        flip = im_metas['flip']
        windows = sliding_window(im, flip, window_size, window_stride)
        #crops = torch.stack(windows.pop('crop'))[:, 0] # (crop)shape = [n_windows, in_channels, wsize_h, wsize_w]
        #torch.stack shape = [n_windows, batch, C, H, W]
        crops = torch.stack(windows.pop('crop')) # shape: [n_windows, batch, C, H, W]
        #print('crops shape: ',crops.shape) # [8, 4, 5, 64, 384]
        n_windows, B, C, H, W = crops.shape
        crops_trans = crops.view(n_windows*B, C, H, W)

        
        
        #with torch.no_grad():
        if use_kpconv:
            #print('crops_trans shape: ', crops_trans.shape)
            seg_maps = model.forward_2d_features(crops_trans) # shape = [n_windows, d_decoder, wsize_h, wsize_w]
        else:
            seg_maps = model.forward(crops_trans) # shape = [n_windows, n_classes, wsize_h, wsize_w]
        seg_shape = seg_maps.shape
        #print('seg_shape: ', seg_shape) [32, 128, 64, 384]
        seg_maps = seg_maps.view(n_windows, B, *seg_shape[1:]).permute(1, 0, 2, 3, 4)
        #print('seg_maps shape: ', seg_maps.shape)
        windows['seg_maps'] = seg_maps #[batch, n_windows, c, h, w]
        im_seg_map = merge_windows_train(windows, window_size, ori_shape) # shape = [n_classes or d_decoder, ori_shape[0], ori_shape[1]]

        if seg_map is None:
            seg_map = im_seg_map
        else:
            seg_map += im_seg_map
    seg_map /= len(ims)
    return seg_map


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
    for im, im_metas in zip(ims, ims_metas):
        im = resize(im, smaller_size)
        flip = im_metas['flip']
        windows = sliding_window(im, flip, window_size, window_stride)
        crops = torch.stack(windows.pop('crop'))[:, 0] # shape = [n_windows, in_channels, wsize_h, wsize_w]
        
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
