# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

# MRCNet framework
import torch
import torch.nn as nn
from collections import OrderedDict

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.mrcnet_bev_backbone import MRCNetBEVBackbone
from opencood.models.fuse_modules.mrcnet_fusion import MSRobustFusion
from opencood.models.sub_modules.MEMmodule import MotionEnhancedMech


class PointPillarMRCNet(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarMRCNet, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = MRCNetBEVBackbone(args['dalign_bev_backbone'], 64)
        

        self.fusion_net = MSRobustFusion(self.max_cav, args['MRFmodule'])

        # use temporal
        self.use_temporal = True if 'temporal_model' in args else False

        if self.use_temporal:
            # MEM module for motion context fusion
            in_channel = args['temporal_model']['in_channel']
            shape_size = args['temporal_model']['shape_size']
            self.temporal_fusion = MotionEnhancedMech(input_size=in_channel, hidden_size=in_channel)
            self.down_sampling = nn.Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(in_channel, in_channel, kernel_size=3,
                    stride=2, padding=0, bias=False),
                nn.BatchNorm2d(in_channel, eps=1e-3, momentum=0.01),
                nn.ReLU())
            self.adap_fusion = nn.Conv2d(128*6, 128*3, kernel_size=1)      

        # V2V and motion context fusion
        self.shrink_flag = False 
        self.down_conv = nn.Sequential(
                        nn.Conv2d(128*6, 128, kernel_size=1,
                                stride=1, padding=0),
                        nn.BatchNorm2d(128), 
                        nn.ReLU(inplace=True))
        
        if self.use_temporal:
            self.cls_head = nn.Conv2d(128, args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(128, 7 * args['anchor_number'],
                                    kernel_size=1)
        else:
            self.cls_head = nn.Conv2d(128*3, args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(128*3, 7 * args['anchor_number'],
                                    kernel_size=1)
        if args['backbone_fix']:
            self.backbone_fix()
            
    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len} 
        
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # downsample feature to reduce memory  
        # if self.shrink_flag:
        #     bev_features_2d = self.shrink_conv(batch_dict['bev_features_2d'])
        # compressor
        # if self.compression:
        #     spatial_features_2d = self.naive_compressor(frames_align_features)

        # multi_scale_features  -------> layers = 3
        batch_dict_list = batch_dict['align_features'] 
        bev_features_2d = batch_dict['bev_features_2d']
        if self.use_temporal:
            record_len = record_len.cpu().tolist()
            # bev_features_2d : ([7, 256, 100, 352])
            his_frames_lens = data_dict['his_frames_lens' ]
            # torch.Size([2, 384, 100, 352])
            temporal_fusion_output = self.temporal_fusion(bev_features_2d, record_len, his_frames_lens) 
            temporal_fusion_output_down = self.down_sampling(temporal_fusion_output)
            temporal_fusion_output_down = temporal_fusion_output_down.transpose(-1, -2)


        fusion_output, batch_comm_volume = self.fusion_net(batch_dict_list, record_len)
        # v2v_fusion_feature : ([2, 384, 176, 50])
        v2v_fusion_feature = fusion_output['aggregated_spatial_features_2d'] 
        
        if self.use_temporal:
            target_feature = torch.cat([temporal_fusion_output_down, v2v_fusion_feature], dim=1)
            if self.down_sampling:
                target_feature = self.down_conv(target_feature)
        else:
            target_feature = v2v_fusion_feature
        psm = self.cls_head(target_feature)
        rm = self.reg_head(target_feature)

        output_dict = {'psm': psm,
                       'rm': rm,
                       'com': batch_comm_volume}
        

        return output_dict
