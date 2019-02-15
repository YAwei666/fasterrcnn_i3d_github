from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

from tools.trainval_net_v0 import combined_roidb


def draw_gt_pic():
    imdb, roidb = combined_roidb('voc_2007_trainval')
    class_to_ind = imdb._class_to_ind
    class_name = [item[0] for item in class_to_ind.items()]
    num=int(len(roidb)/2)
    for item in roidb[:num]:
        gt_bbox = item['boxes']
        gt_classes = [class_name[indx] for indx in item['gt_classes']]
        path = item['image']
        img = cv2.imread(path)
        for box in gt_bbox:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        pic_indx=path.split('/')[-1]
        cv2.imwrite('/home/wbr/cqq/faster-rcnn_endernewton/gt_box_pic/{}_{}'
                    .format('+'.join(gt_classes),pic_indx), img)


draw_gt_pic()
