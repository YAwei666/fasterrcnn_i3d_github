# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import pickle

import cv2

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


def visilize():
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    print(1)
    args = parse_args()
    args.imdb = 'voc_20007_test'
    args.model = '../output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_70000.ckpt'
    args.cfg = '../experiments/cfgs/vgg16.yml'
    args.net = 'vgg16'
    args.set = 'NCHOR_SCALES [8,16,32] ANCHOR_RATIOS [0.5,1,2]'
    print('Called with args:')
    output_dir = '/home/wbr/cqq/faster-rcnn_endernewton/output/default/voc_2007_test/default/vgg16_faster_rcnn_iter_70000'
    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    file = open(
        '/home/wbr/cqq/faster-rcnn_endernewton/output/default/voc_2007_test/default/vgg16_faster_rcnn_iter_70000/detections.pkl',
        'rb')
    # file = open(
    #     '/home/wbr/cqq/tf-faster-rcnn/output/default/voc_2007_test/default/vgg16_faster_rcnn_iter_25000/detections.pkl',
    #     'rb')
    all_boxes = pickle.load(file)[1:]
    # imdb.evaluate_detections(all_boxes, output_dir)
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    for i in range(num_images):
        i=i+2
        bbox = [item[i] for item in all_boxes]
        for item in bbox:
            if len(item) == 0:
                continue
            for j,box in enumerate(item):
                if box[4] > 0.00003:
                    if (box[2]-box[0])>10 and (box[3]-box[1])>10:
                        img = cv2.imread(imdb.image_path_at(i))
                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        print(box)
                        # cv2.imshow('image', img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        cv2.imwrite('{}_{}.jpg'.format(i,j),img)

def analyse_voc_data():
    path='/home/wbr/cqq/faster-rcnn_endernewton/data/VOCdevkit2007/VOC2007/JPEGImages'
    imgs=os.listdir(path)
    shape0=[]
    shape1=[]
    for img in imgs:
        img_pix=cv2.imread(os.path.join(path,img))
        shape0.append(img_pix.shape[0])
        shape1.append(img_pix.shape[1])

    print(max(shape0))
    print(min(shape0))
    print(max(shape1))
    print(min(shape1))


if __name__ == '__main__':
    visilize()


    # analyse_voc_data()