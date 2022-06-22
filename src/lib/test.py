from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        ###############
        #### MODEL ####
        ###############
        
        self.parser.add_argument('--arch', default='hardnet_68', 
                                 help='model architecture. Currently tested'
                                      'hardnet_85 | hardnet_68')

        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '256.')
        
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')
         

    def parse(self, args=''):
        """ Returns parser object"""

        if args == '':
            opt = self.parser.parse_args()

        else:
            opt = self.parser.parse_args(args)

        if opt.head_conv == -1: # init default head_conv
            opt.head_conv = 256 
        
        opt.pad = 127 if 'hourglass' in opt.arch else 31
        opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

        
        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
    
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes
        
        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)
        
        if opt.task == 'exdet':
          # assert opt.dataset in ['coco']
          num_hm = 1 if opt.agnostic_ex else opt.num_classes
          opt.heads = {'hm_t': num_hm, 'hm_l': num_hm, 
                       'hm_b': num_hm, 'hm_r': num_hm,
                       'hm_c': opt.num_classes}
          if opt.reg_offset:
            opt.heads.update({'reg_t': 2, 'reg_l': 2, 'reg_b': 2, 'reg_r': 2})
        
        elif opt.task == 'ddd':
          # assert opt.dataset in ['gta', 'kitti', 'viper']
          opt.heads = {'hm': opt.num_classes, 'dep': 1, 'rot': 8, 'dim': 3}
          if opt.reg_bbox:
            opt.heads.update(
              {'wh': 2})
          if opt.reg_offset:
            opt.heads.update({'reg': 2})
        elif opt.task == 'ctdet':
          # assert opt.dataset in ['pascal', 'coco']
          opt.heads = {'hm': opt.num_classes,
                       'wh': 4 if not opt.cat_spec_wh else 2 * opt.num_classes}
        elif opt.task == 'multi_pose':
          # assert opt.dataset in ['coco_hp']
          opt.flip_idx = dataset.flip_idx
          opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 34}
          if opt.reg_offset:
            opt.heads.update({'reg': 2})
          if opt.hm_hp:
            opt.heads.update({'hm_hp': 17})
          if opt.reg_hp_offset:
            opt.heads.update({'hp_offset': 2})
        else:
          assert 0, 'task not defined!'
        print('heads', opt.heads)
        
        return opt

    def init(self, args=''):

        print("you called me")

        default_dataset_info = {
          'ctdet': {'default_resolution': [512, 512], 'num_classes': 80, 
                    'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                    'dataset': 'coco'},
          'exdet': {'default_resolution': [512, 512], 'num_classes': 80, 
                    'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                    'dataset': 'coco'},
          'multi_pose': {
            'default_resolution': [512, 512], 'num_classes': 1, 
            'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
            'dataset': 'coco_hp', 'num_joints': 17,
            'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
                         [11, 12], [13, 14], [15, 16]]},
          'ddd': {'default_resolution': [384, 1280], 'num_classes': 3, 
                    'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                    'dataset': 'kitti'},
        }

        class Struct:
          def __init__(self, entries):
            for k, v in entries.items():
              self.__setattr__(k, v)

        opt = self.parse(args)
        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        
        return opt


opt = opts().parse()
print(opt.heads)