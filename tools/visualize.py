import pickle as pkl
import os

import matplotlib
from extract_trajs import *
from imutils.video import FPS
from dlib import drectangle
import dlib
from trajectory import Trajectory, traj_iou

import cv2
import numpy as np
matplotlib.use('Agg')
# def get_start_end(vid):
#     path = '/home/next/Documents/video_realtion/Dataset-Shangxindi/VidVRD-videos/ResultsValEveryFramesDetection-Conf0.3/ILSVRC2015_val_{}.txt'.format(vid)
#     content = open(path, 'rb').read().decode('utf-8').split('\n')[1:]
#     start = content[0].split('frame number id : ')[1].split(',')[0]
#     end = content[-2].split('frame number id : ')[1].split(',')[0]
#     segs = [[i, i + 30] for i in range(int(start), int(end) - 30 + 1, 15)]
#     return int(start), int(end)

# def extract_boundingbox(inf, frame_number, fstart, fend):
#     boxes = []
#
#     for traj in inf:
#         traj_dict = {}
#         # rois = [[item1.left(), item1.top(), item1.right(), item1.bottom()] for item1 in traj.rois]
#         rois = [[item1[0], item1[1], item1[2], item1[3]] for item1 in traj['rois']]
#         traj_dict['bb'] = rois
#         boxes.append(traj_dict['bb'][frame_number - fstart])
#     return boxes

# def visualization(video_file, item, base_dir, fstart, fend, output_video):
#     writer = None
#     frame_number = 0
#     video = cv2.VideoCapture(video_file)
#     while True:
#         (Bole, frame) = video.read()
#         frame_number = frame_number + 1
#         if Bole is False:
#             break
#         frames = frame
#
#         if output_video is not None and writer is None:
#             fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#             writer = cv2.VideoWriter(output_video, fourcc, 30, (frames.shape[1], frames.shape[0]), True)
#
#         traj_file = os.path.join(base_dir, item)
#         with open(traj_file, 'rb') as reader:
#             inf = pkl.load(reader)
#         if frame_number >= fstart and frame_number < fend:
#             box = extract_boundingbox(inf, frame_number, fstart, fend)
#             for i in box:
#                 box_int = []
#                 for x in i:
#                     if x > 0 and x < max(frames.shape[1], frames.shape[0]):
#                         y = int(x)
#                     elif x < 0:
#                         y = int(0)
#                     elif x > max(frames.shape[1], frames.shape[0]):
#                         y = int(max(frames.shape[1], frames.shape[0]))
#                     box_int.append(y)
#                     # box_int = [int(x) for x in i]
#                 (startX, startY, endX, endY) = tuple(box_int)
#                 cv2.rectangle(frames, (startX, startY), (endX, endY), (0, 225, 0), 2)
#             if writer is not None:
#                 writer.write(frames)
#             cv2.imshow("Frames", frames)
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord("q"):
#                 break
#             fps.update()
#
#     fps.stop()
#     if writer is not None:
#         writer.release()
#     cv2.destroyAllWindows()
#     video.release()
def get_start_end(vid):
    # TODO 给cqq需要修改
    # path = '/home/next/Documents/video_realtion/Dataset-Shangxindi/VidVRD-videos/ResultsValEveryFramesDetection-Conf0.3/ILSVRC2015_val_{}.txt'.format(vid)
    path = '/home/wbr/github/maskrcnn-benchmark/dataset200/pic/{}.mp4'.format(vid)
    if not(os.path.exists(path)):
        path = '/home/wbr/github/maskrcnn-benchmark/dataset200/pic/{}.mp4'.format(vid)
    names=os.listdir(path)
    end=len(names)/2
    return 1, int(end)



def extract_boundingbox(inf, frame_number, startid, endid):
    boxes = []

    for traj in inf:
        traj_dict = {}
        rois = [[item1.left(), item1.top(), item1.right(), item1.bottom()] for item1 in traj.rois]
        # rois = [[item1[0], item1[1], item1[2], item1[3]] for item1 in traj['rois']]
        traj_dict['bb'] = rois
        boxes.append(traj_dict['bb'][frame_number - startid])
    return boxes

def visualization(video_frame, vid, base_dir, startid, endid, output_video):
    writer = None
    fps = FPS().start()
    if output_video is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_video, fourcc, 30, (w, h), True)

    item = '{}_{}_{}_trajs_pred.pkl'.format(vid, startid, endid + 1)
    traj_file = os.path.join(base_dir, vid,item)     #对应的这部分的traj
    with open(traj_file, 'rb') as reader:
        inf = pkl.load(reader)
    for index in range(len(video_frame)):  #0 1 2   29
        frame_number = index + startid     #1 2   30  5 6 34
        box = extract_boundingbox(inf, frame_number, startid, endid)
        for i in box:
            # aaa = box.index(i)
            box_int = []
            for x in i:
                if x > 0 and x < max(w, h):
                    y = int(x)
                elif x < 0:
                    y = int(0)
                elif x > max(w, h):
                    y = int(max(w, h))
                box_int.append(y)
            (startX, startY, endX, endY) = tuple(box_int)
            frame = video_frame[index - 1]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 225, 0), 2)
        if writer is not None:
            writer.write(video_frame[index - 1])
        cv2.imshow("Frames", video_frame[index - 1])
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        fps.update()

    fps.stop()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    # names = [name for name in os.listdir("/home/next/Documents/program/MultiObject-Tracking-dlib/MyCode/trajs-bug/track_pkl_0.35000000000000003_0.7000000000000001/")
    #              if os.path.isfile(os.path.join("/home/next/Documents/program/MultiObject-Tracking-dlib/MyCode/trajs-bug/track_pkl_0.35000000000000003_0.7000000000000001/", name))]
    # names = [name for name in os.listdir('/home/next/Documents/video_realtion/Dataset-Shangxindi/VidVRD-videos/ResultsValEveryFramesDetection-Conf0.3/')
    #          if os.path.isfile(os.path.join(
    #         "/home/next/Documents/video_realtion/Dataset-Shangxindi/VidVRD-videos/ResultsValEveryFramesDetection-Conf0.3/",
    #         name))]
    root_path = 'S:\曹倩倩\VidVRD\\traj'
    names=os.listdir(root_path)
    # base_dir = "/home/next/Documents/program/MultiObject-Tracking-dlib/MyCode/trajs-bug/track_pkl_0.35000000000000003_0.7000000000000001/"

    # detection_dir = '/home/next/Documents/video_realtion/Dataset-Shangxindi/VidVRD-videos/Detection-Results-Val-EveryFramesDetection/'
    # file_names = os.listdir(detection_dir)

    # for item in names:
    #     print(item)
    #     vid = item.split('_')[0]
    #
    #     video_file = '/home/next/Documents/video_realtion/Dataset-Shangxindi/VidVRD-videos/val_detected/ILSVRC2015_val_{}.mp4'.format(vid)
    #     output_video = '/home/next/Documents/video_realtion/Dataset-Shangxindi/VidVRD-videos/VideoResult-Val-EveryFramesDetection-conf0.3/Result_ILSVRC2015_val_{}.avi'.format(vid)
    #     if os.path.exists(output_video):
    #         continue
    #     aaa = item.find("pred")
    #     if aaa != -1:
    #         fstart, fend = get_start_end(vid)
    #         fps = FPS().start()
    #         visualization(video_file, item, base_dir, fstart, fend, output_video)
    #         print("aaa")

    # item = '00169000_1_893_trajs_pred.pkl'
    for item in names:
        print(item)
        # vid = '00007027'
        # video_file = '/home/next/Documents/video_realtion/Dataset-Shangxindi/VidVRD-videos/val_detected/ILSVRC2015_val_{}.mp4'.format(vid)
        video_file = 'S:\曹倩倩\VidVRD\\video_test\{}.mp4'.format(item)
        # aaa = item.find("pred")
        # if aaa != -1:
        video_segs = []
        all_frames = []
        video = cv2.VideoCapture(video_file)
        while True:
            (Bole, frame) = video.read()
            if Bole is False:
                break
            w = frame.shape[1]
            h = frame.shape[0]
            all_frames.append(frame)
        # fstart = 1
        # fend = len(all_frames)
        # fstart, fend = get_start_end(item)
        fstart, fend =1,100

        segs = [[i, i + 30 - 1] for i in range(fstart, fend - 30, 15)]
        for seg in segs:
            startid = seg[0]  #1
            endid = seg[1] #30
            output_video = 'S:\曹倩倩\VidVRD\output_video\\{}'.format(
                item, startid, endid)
            if os.path.exists(output_video):
                continue
            video_frame = all_frames[startid - 1:endid]
            # video_segs.append(video_seg)
            # fstart, fend = get_start_end(vid)
            visualization(video_frame, item, root_path, startid, endid, output_video)
            print("aaa")




