# Copyright 2023 - Valeo Comfort and Driving Assistance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import padding, unpadding
from .kpconv.blocks import KPConv

from utils.inference.inference_utils import inference, inference_train

from einops import rearrange, repeat


from Proto.lib.models.modules.sinkhorn import distributed_sinkhorn
import torch.distributed as dist

def resample_grid(predictions, py, px):
    pypx = torch.stack([px, py], dim=3)
    resampled = F.grid_sample(predictions, pypx)
    return resampled


class KPClassifier(nn.Module):
    # Adapted from D. Kochanov et al.
    # https://github.com/DeyvidKochanov-TomTom/kprnet
    def __init__(self, in_channels=256, out_channels=256, num_classes=17, dummy=False):
        super(KPClassifier, self).__init__()
        self.kpconv = KPConv(
            kernel_size=15,
            p_dim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            KP_extent=1.2,
            radius=0.60,
        )
        self.dummy = dummy
        if self.dummy:
            del self.kpconv
            self.kpconv = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.head = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, x, px, py, pxyz, pknn, num_points):
        assert px.shape[0] == py.shape[0]
        assert px.shape[0] == pxyz.shape[0]
        assert px.shape[0] == pknn.shape[0]
        assert px.shape[0] == num_points.sum().item()
        res = []
        offset = 0
        batch_size = x.shape[0]
        for i in range(batch_size):
            len = num_points[i]
            px_i = px[offset:(offset+len)].unsqueeze(0).unsqueeze(1).contiguous()
            py_i = py[offset:(offset+len)].unsqueeze(0).unsqueeze(1).contiguous()
            points = pxyz[offset:(offset+len)].contiguous()
            pknn_i = pknn[offset:(offset+len)].contiguous()
            resampled = F.grid_sample(
                x[i].unsqueeze(0), torch.stack([px_i, py_i], dim=3),
                align_corners=False, padding_mode='border')
            feats = resampled.squeeze().t()

            if feats.shape[0] != points.shape[0]:
                print(f'feats.shape={feats.shape} vs points.shape={points.shape}')
            assert feats.shape[0] == points.shape[0]
            if self.dummy:
                feats = self.kpconv(feats)
            else:
                feats = self.kpconv(points, points, pknn_i, feats)
            res.append(feats)
            offset += len

        assert offset == px.shape[0]
        #res = torch.cat(res, axis=0).unsqueeze(2).unsqueeze(3)
        res = torch.cat(res, axis=0)
        res = self.relu(self.bn(res))
        #print('relu res shape: ', res.shape) [batch내 모든 num_points, 128, 1, 1]
        #print('head res shape: ', self.head(res).shape) [batch내 모든 num_points, 20, 1, 1]
        #return self.head(res)
        return res


class RangeViT_KPConv(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        kpclassifier,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.patch_stride = encoder.patch_stride
        self.encoder = encoder
        self.decoder = decoder
        del self.decoder.head
        self.kpclassifier = kpclassifier

        #proto
        self.gamma = 0.99
        in_channels = 128
        self.num_prototype = 10
        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(20)

        self.prototypes = nn.Parameter(torch.zeros(20, self.num_prototype, in_channels),
                                       requires_grad=True)
        #print('self.prototypes device: ', self.prototypes.get_device())

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay('encoder.', self.encoder).union(
            append_prefix_no_weight_decay('decoder.', self.decoder)
        )
        return nwd_params

    def forward_2d_features(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        
        x, skip = self.encoder(im, return_features=True) # x.shape = [16, 577, 384]
        
        # remove CLS tokens for decoding
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:] # x.shape = [16, 576, 384]
        
        feats = self.decoder(x, (H, W), skip, return_features=True)
        feats = F.interpolate(feats, size=(H, W), mode='bilinear', align_corners=False)
        feats = unpadding(feats, (H_ori, W_ori))
        return feats

    
    def forward(self, im, px, py, pxyz, pknn, num_points, gt_semantic_seg, is_train, im_meta=False, 
                    window_size = None, window_stride = None):
        if is_train:
            feats = self.forward_2d_features(im)
        else:
            output_features2d = inference(
                            self,
                            [im],
                            [im_meta],
                            ori_shape=im.shape[2:4],
                            window_size=window_size,
                            window_stride=window_stride,
                            batch_size=im.shape[0],
                            use_kpconv=True)
            feats = output_features2d.unsqueeze(0)
        #masks3d = self.kpclassifier(feats, px, py, pxyz, pknn, num_points)
        feature3d = self.kpclassifier(feats, px, py, pxyz, pknn, num_points)

        _c = self.feat_norm(feature3d)
        _c = l2_normalize(_c) # num_points * features

        #print('before prototype device: ', self.prototypes.get_device())
        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        device = _c.get_device()
        self.prototypes =  nn.Parameter(self.prototypes.to(device))
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

        out_seg = torch.amax(masks, dim=1) #dim=1내에서 가장 큰 값, [n,k] 각 class마다 가장 유사한 prototype과의 내적값 반환
        out_seg = self.mask_norm(out_seg) # num_points , k (k: num_class)
        #out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2]) #for pixel-class distance(eq7)

        
        # pretrain_prototype: 처음 몇 iter동안은 true, 
        if gt_semantic_seg is not None:
            #gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest').view(-1) #1차원 벡터
            gt_seg = gt_semantic_seg.view(-1)
            contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
            return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}

        
        return out_seg

        

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
            #pred_seg = torch.max(out_seg, 1)[1] # out_seg:  b k h w (k = num_classes) -> b h w
            pred_seg = torch.max(out_seg, 1)[1] #out_seg : num_points, k -> num_points
            mask = (gt_seg == pred_seg.view(-1))

            #_c: [(b h w) c], mm: matrix multiplication
            cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t()) # prototype: k,m,d -> d, k*m

            proto_logits = cosine_similarity
            proto_target = gt_seg.clone().float()

            # clustering for each class
            protos = self.prototypes.data.clone()
            for k in range(20):
                init_q = masks[..., k] #masks: 전체 cosine similarity(s) , [n,m,k]
                init_q = init_q[gt_seg == k, ...]
                if init_q.shape[0] == 0:
                    continue

                q, indexs = distributed_sinkhorn(init_q)

                m_k = mask[gt_seg == k]

                c_k = _c[gt_seg == k, ...]

                m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

                m_q = q * m_k_tile  # n x self.num_prototype

                c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

                c_q = c_k * c_k_tile  # n x embedding_dim

                f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

                n = torch.sum(m_q, dim=0)

                if torch.sum(n) > 0 :
                    f = F.normalize(f, p=2, dim=-1)

                    new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                                momentum=self.gamma, debug=False)
                    protos[k, n != 0, :] = new_value

                proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

            self.prototypes = nn.Parameter(l2_normalize(protos),
                                        requires_grad=False)

            if dist.is_available() and dist.is_initialized():
                protos = self.prototypes.data.clone()
                dist.all_reduce(protos.div_(dist.get_world_size()))
                self.prototypes = nn.Parameter(protos, requires_grad=False)

            return proto_logits, proto_target
            


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update