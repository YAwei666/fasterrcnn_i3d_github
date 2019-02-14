# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import _init_paths

from lib.model.test import test_net

import argparse
import os
import pprint
import sys

import tensorflow as tf
from datasets.factory import get_imdb
from model.config import cfg, cfg_from_file, cfg_from_list
from nets.mobilenet_v1 import mobilenetv1
from nets.resnet_v1 import resnetv1
from nets.vgg16 import vgg16


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--model', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

import numpy as np
if __name__ == '__main__':
    file=open('/home/wbr/cqq/faster-rcnn_endernewton/output/default/voc_2007_test/default/vgg16_faster_rcnn_iter_70000/detections.pkl','rb')
    # file = open(
    #     '/home/wbr/cqq/faster-rcnn_endernewton/output/default/voc_2007_trainval/default'
    #     '/res101_faster_rcnn_iter_1070000/detections.pkl',
    #     'rb')
    print(11)
    all_boxes=pickle.load(file)
    num_images=len(all_boxes[1])
    for i in range(num_images):
        print(i)
        bboxs = [item[i] for item in all_boxes]
        for j,bbox in enumerate(bboxs):
                if len(bbox)!=0:
                    for bbox1 in bbox:
                        print(j)
                        local=bbox1[:4].astype(int)
                        print(local)
                        print((bbox1[4]))
        # print(all_boxes[1][0])
    # print(all_boxes[1][1])
