# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import json
import os

from easydict import EasyDict


class sxd():
    def __init__(self):
        self.name='sxd'
        self.json_path = '/home/wbr/github/maskrcnn-benchmark/dataset/groundtruth'
        self.pic_root_path = '/home/wbr/github/maskrcnn-benchmark/dataset/pic'
        self._classes = ['__background__']
        self.roidb = []
        self.load_class_information()

    def load_train_data(self):
        paths=os.listdir('/home/wbr/github/maskrcnn-benchmark/dataset/pic')
        paths=[item.split('.')[0]+'.json' for item in paths]
        # paths = os.listdir(self.json_path)
        for path in paths:
            self.load_ground_truth_json(os.path.join(self.json_path, path))

    def load_class_information(self):
        classes = []
        paths = os.listdir(self.json_path)
        for path in paths:
            file = codecs.open(os.path.join(self.json_path, path), 'r', 'utf-8')
            json_content = json.load(file)
            content = EasyDict(json_content)
            subjects_ojects = content['subject/objects']
            for item in subjects_ojects:
                classes.append(item['category'])

        classes = list(set(classes))
        self._classes.extend(classes)
        self.num_classes = len(self._classes)
        self._class_to_ind = dict(list(zip(self._classes, list(range(self.num_classes)))))

    def load_ground_truth_json_multiframe(self, path):

        file = codecs.open(path, 'r', 'utf-8')
        json_content = json.load(file)
        content = EasyDict(json_content)
        vid_name = path.split('/')[-1].split('.')[0]
        fstart = 0
        # fend = content['frame_count'] - 1
        fend = len(content['trajctories']) - 1
        width = content['width']
        height = content['height']
        tid_to_classes = {}
        for item in content['subject/objects']:
            tid_to_classes[item['tid']] = item['category']

        seg_length = 20
        overlap_length = int(seg_length / 2)
        segs = [[i, i + seg_length] for i in range(fstart, fend - seg_length + 1, overlap_length)]
        # seg:[0,1,2...,20]
        video_roidb = []
        for seg in segs:
            seg_content = [content['trajectories'][indx] for indx in seg]
            segs_infor = []
            for i, item in enumerate(seg_content):
                # 如果当前帧没有轨迹
                if len(item) == 0:
                    infor = {}
                    continue

                infor = {}
                # 提取bbox和class
                bboxs = []
                class_names = []
                for traj in item:
                    bboxs.append([traj['bbox']['xmin'],
                                  traj['bbox']['ymin'],
                                  traj['bbox']['xmax'],
                                  traj['bbox']['ymax']])
                    class_names.append(tid_to_classes[traj['tid']])
                infor['boxes'] = bboxs
                infor['gt_classes'] = class_names
                infor['flipped'] = False
                infor['image'] = os.path.join(self.pic_root_path, vid_name + '.mp4', '{}_raw.jpeg'.format(seg[i]))
                infor['width'] = width
                infor['height'] = height
                segs_infor.append(infor)

            video_roidb.append(segs_infor)

        return video_roidb

    def load_ground_truth_json(self, path):

        file = codecs.open(path, 'r', 'utf-8')
        json_content = json.load(file)
        content = EasyDict(json_content)
        vid_name = path.split('/')[-1].split('.')[0]
        fstart = 0
        # fend = content['frame_count'] - 1
        fend = len(content['trajectories']) - 1
        width = content['width']
        height = content['height']
        tid_to_classes = {}
        for item in content['subject/objects']:
            tid_to_classes[item['tid']] = item['category']

        seg_length = 20
        overlap_length = int(seg_length / 2)
        segs = [[i, i + seg_length] for i in range(fstart, fend - seg_length + 1, overlap_length)]
        segs=[list(range(item[0],item[1])) for item in segs]
        # seg:[0,1,2...,20]
        for seg in segs:
            seg_content = [content['trajectories'][indx] for indx in seg]
            center_item = seg_content[int(seg_length / 2)]
            # 如果当前帧没有轨迹
            if len(center_item) == 0:
                continue

            infor = {}
            # 提取bbox和class
            bboxs = []
            class_names = []
            for traj in center_item:
                bboxs.append([traj['bbox']['xmin'],
                              traj['bbox']['ymin'],
                              traj['bbox']['xmax'],
                              traj['bbox']['ymax']])
                class_names.append(tid_to_classes[traj['tid']])

            class_names=[self._class_to_ind[name] for name in class_names]
            infor['boxes'] = bboxs
            infor['gt_classes'] = class_names
            infor['flipped'] = False
            image_paths=[]
            for indx in seg:
                image_paths.append(os.path.join(self.pic_root_path, vid_name + '.mp4',
                                                '{}_raw.jpeg'.format(indx)))
            infor['image'] = image_paths
            infor['width'] = width
            infor['height'] = height
            self.roidb.append(infor)



if __name__ == '__main__':
    d = sxd()
    # print(d.classes)
    # d.load_class_information()
    d.load_train_data()
    print(len(d.roidb))
