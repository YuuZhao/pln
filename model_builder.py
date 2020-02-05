# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck

from plastic_test import PlasticNet


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)

        #print("zf：",zf) zf 送pln_test,返回一个新的模型
        #yy
        #zf= pln_test(zf)
        self.zf = zf

    def template_pln(self,z,IF_TI):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)

        #print("zf：",zf) zf 送pln_test,返回一个新的模型
        #yy
        #zf= pln_test(zf)
        #zf= pln_test(zf,)
        # 此处更新zf ,输入zf到pln ,输出pln , pln 的更新需要， ti ,gt0,~ti
        if IF_TI:
            zff = PlasticNet(self.gt0, zf, IF_TI, self.zf)
        else:
            #gt0 = zf
            #print("zf:",zf)
            #C = zf[0].shape[1]
            #W = zf[0].shape[2]
            #H = zf[0].shape[3]
            zf0=zf[0]
            zf1=zf[1]
            zf2=zf[2]
            zfz=torch.cat([zf0,zf1,zf2],0)
            zfzz =zfz.cpu()
            zfz4d=zfzz.data.numpy()
            zfz3d=np.concatenate(zfz4d)
            zfz2d=np.concatenate(zfz3d)
            zfz1d=np.concatenate(zfz2d)

            gt0 = zfz1d
            zff = PlasticNet(gt0, zf, IF_TI)

            ##恢复形状
            #print("HWC:",H,W,C)
            #re_zfz4d = zfzdd.reshape(3,C,W,H)
            #re_zfzz = torch.from_numpy(re_zfz4d)
            #absdiff=np.abs(zfz4d-re_zfz4d)
            #re_zf =[]
            #re_zf.append(re_zfzz[0])
            #re_zf.append(re_zfzz[1])
            #re_zf.append(re_zfzz[2])
            #zf = re_zf
            self.gt0 = gt0
        #####
        self.zf = zff

        #return zf

    def track_pln(self, x,p):

        zf = self.backbone(p)
        #############################################
        #C = zf[0].shape[1]
        #W = zf[0].shape[2]
        #H = zf[0].shape[3]
        #zf0 = zf[0]
        #zf1 = zf[1]
        #zf2 = zf[2]
        #zfz = torch.cat([zf0, zf1, zf2], 0)
        #zfzz = zfz.cpu()
        #zfz4d = zfzz.data.numpy()
        #zfz3d = np.concatenate(zfz4d)
        #zfz2d = np.concatenate(zfz3d)
        #zfz1d = np.concatenate(zfz2d)
        IF_TI = True
        ####self.zf 更改形状
        zff = PlasticNet(self.gt0,zf,IF_TI,self.zf)
        #re_zfz4d = zfzdd.reshape(3, C, W, H)
        #re_zfzz = torch.from_numpy(re_zfz4d)
        # absdiff=np.abs(zfz4d-re_zfz4d)
        #re_zf = []
        #re_zf.append(re_zfzz[0])
        #re_zf.append(re_zfzz[1])
        #re_zf.append(re_zfzz[2])
        #zf = re_zf
        ################################################
        self.zf = zff

        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }



    def track(self, x):
        xf = self.backbone(x)

        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs
