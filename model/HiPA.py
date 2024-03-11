r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from .base.swin_transformer import SwinTransformer
from model.base.contrast import Contrast_Activate, PositionalEncoding


class HiPA(nn.Module):

    def __init__(self, backbone, pretrained_path, use_original_imgsize):
        super(HiPA, self).__init__()

        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize
        self.resolution = [(48,48), (24,24), (12,12)]

        # feature extractor initialization
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone == 'swin':
            self.feature_extractor = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128,
                                            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['model'])
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        # lids[1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4]
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
        # stack_ids=tensor([ 3,  7, 13, 16])
        self.model = HiPA_model(in_channels=self.feat_channels, stack_ids=self.stack_ids,resolution=self.resolution)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, support_img, support_mask, lamda=None):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img)
            support_feats = self.extract_feats(support_img)

        logit_mask = self.model(query_feats, support_feats, support_mask.clone(),lamda=lamda)

        return logit_mask

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []

        if self.backbone == 'swin':
            _ = self.feature_extractor.forward_features(img)
            for feat in self.feature_extractor.feat_maps:
                bsz, hw, c = feat.size()
                h = int(hw ** 0.5)
                feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats.append(feat)
        elif self.backbone == 'resnet50' or self.backbone == 'resnet101':
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def predict_mask_nshot(self, batch, nshot):
        r""" n-shot inference """
        query_img = batch['query_img']
        support_imgs = batch['support_imgs']
        support_masks = batch['support_masks']

        if nshot == 1:
            logit_mask = self(query_img, support_imgs[:, 0], support_masks[:, 0])
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                n_support_feats = []
                for k in range(nshot):
                    support_feats = self.extract_feats(support_imgs[:, k])
                    n_support_feats.append(support_feats)
            logit_mask = self.model(query_feats, n_support_feats, support_masks.clone(),nshot)

        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        else:
            logit_mask = F.interpolate(logit_mask, support_imgs[0].size()[2:], mode='bilinear', align_corners=True)

        return logit_mask.argmax(dim=1)

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.feature_extractor.eval()


class HiPA_model(nn.Module):
    def __init__(self, in_channels, stack_ids,resolution):
        # stack_ids=[3, 7, 13, 16]
        super(HiPA_model, self).__init__()

        self.head=8
        self.stack_ids = stack_ids
        # in_channels = [256, 512, 1024, 2048]
        # stack_ids=[3, 7, 13, 16]

        # ----设置权值-------
        self.register_buffer('lamda', torch.tensor(0., dtype=torch.float32))

        # blocks
        self.HiPA_blocks = nn.ModuleList()
        self.pe = nn.ModuleList()
        self.node=((10,4,1),(10,4,1),(10,4,1))
        for index,inch in enumerate(in_channels[1:]):
            # ------------原版----------------------
            self.HiPA_blocks.append(Contrast_Activate(h=self.head, d_model=inch, dropout=0.5, node=self.node[index]))
            # -------------------------------------
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))

        outch1, outch2, outch3 = 32, 64, 128

        # conv blocks
        # ------------初始版本，多头平均后输出----------------------
        # self.conv1 = self.build_conv_block((stack_ids[3] - stack_ids[2]), [outch1, outch2, outch3],
        #                                    [3, 3, 3], [1, 1, 1])  # 1/32
        # self.conv2 = self.build_conv_block((stack_ids[2] - stack_ids[1]), [outch1, outch2, outch3],
        #                                    [5, 3, 3], [1, 1, 1])  # 1/16
        # self.conv3 = self.build_conv_block((stack_ids[1] - stack_ids[0]), [outch1, outch2, outch3],
        #                                    [5, 5, 3], [1, 1, 1])  # 1/8
        # -----------------------------------------------------------------

        # ------------多维度输出----------------------------
        self.conv1 = self.build_conv_block((stack_ids[3]-stack_ids[2])*15, [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1]) # 1/32
        self.conv2 = self.build_conv_block((stack_ids[2]-stack_ids[1])*15, [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1]) # 1/16
        self.conv3 = self.build_conv_block((stack_ids[1]-stack_ids[0])*15, [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1]) # 1/8
        # -------------------------------------------------
        #
        self.conv4 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/32 + 1/16
        self.conv5 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8

        # mixer blocks
        # ----------------------原始mixer1------------------------------------
        self.mixer1 = nn.Sequential(nn.Conv2d(outch3+2*in_channels[1]+2*in_channels[0], outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())
        # ---------------------------------------------------------------


        self.mixer2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer3 = nn.Sequential(nn.Conv2d(outch1, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))

    def forward(self, query_feats, support_feats, support_mask, nshot=1,lamda=None):
        if lamda is not None:
            self.lamda = torch.tensor(lamda,dtype=torch.float32)
        coarse_masks = []
        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx < self.stack_ids[0]: continue

            bsz, ch, ha, wa = query_feat.size()

            # reshape the input feature and mask
            query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
            if nshot == 1:
                support_feat = support_feats[idx]
                mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                     align_corners=True).view(support_feat.size()[0], -1)
                # mask(bsz,h*w)
                # support_feat(bsz,ch,h,w)
                support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2, 1).contiguous()
                # support_feat(bsz,h*w,ch)
            else:
                support_feat = torch.stack([support_feats[k][idx] for k in range(nshot)])
                support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
                mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True)
                                    for k in support_mask])
                mask = mask.view(bsz, -1)

            # HiPA blocks forward
            # ---------注意力输出-----------------------
            if idx < self.stack_ids[1]:
                coarse_mask = self.HiPA_blocks[0](self.pe[0](query), self.pe[0](support_feat), mask, lamda=self.lamda)
                # coarse_mask(bsz,head,h*w,1)
            elif idx < self.stack_ids[2]:
                coarse_mask = self.HiPA_blocks[1](self.pe[1](query), self.pe[1](support_feat), mask, lamda=self.lamda)
            else:
                coarse_mask = self.HiPA_blocks[2](self.pe[2](query), self.pe[2](support_feat), mask, lamda=self.lamda)
            # if idx < self.stack_ids[1]:
            #     coarse_mask = self.HiPA_blocks[0](query, support_feat, mask,lamda)
            #     # coarse_mask(bsz,head,h*w,1)
            # elif idx < self.stack_ids[2]:
            #     coarse_mask = self.HiPA_blocks[1](query, support_feat, mask,lamda)
            # else:
            #     coarse_mask = self.HiPA_blocks[2](query, support_feat, mask,lamda)
            # ----------------------------------------------------

            # -------mask取平均----------------
            coarse_masks.append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, -1, ha, wa))
            # --------------------------------

        # multi-scale conv blocks forward
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[3]-1-self.stack_ids[0]].size()
        coarse_masks1 = torch.stack(coarse_masks[self.stack_ids[2]-self.stack_ids[0]:self.stack_ids[3]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[2]-1-self.stack_ids[0]].size()
        coarse_masks2 = torch.stack(coarse_masks[self.stack_ids[1]-self.stack_ids[0]:self.stack_ids[2]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[1]-1-self.stack_ids[0]].size()
        coarse_masks3 = torch.stack(coarse_masks[0:self.stack_ids[1]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)

        coarse_masks1 = self.conv1(coarse_masks1)
        coarse_masks2 = self.conv2(coarse_masks2)
        coarse_masks3 = self.conv3(coarse_masks3)

        # multi-scale cascade (pixel-wise addition)
        coarse_masks1 = F.interpolate(coarse_masks1, coarse_masks2.size()[-2:], mode='bilinear', align_corners=True)
        mix=coarse_masks1+coarse_masks2
        mix = self.conv4(mix)

        mix = F.interpolate(mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True)
        mix = mix + coarse_masks3
        mix = self.conv5(mix)

        # skip connect 1/8 features
        # --------------------初始版本，query和support一起输出-------------------------
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[1] - 1]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]).max(dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[1] - 1], support_feat), 1)
        # -------------------------------------------------------------------------

        # 上采样
        upsample_size = (mix.size(-1) * 2,) * 2
        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)

        # skip connect 1/4 features
        # ---------------初始版本，query和support一起输出-------------------------
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[0] - 1]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]).max(dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[0] - 1], support_feat), 1)
        # -------------------------------------------------------------------------

        # mixer blocks forward
        out = self.mixer1(mix)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        out = self.mixer2(out)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.mixer3(out)

        return logit_mask


    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)


