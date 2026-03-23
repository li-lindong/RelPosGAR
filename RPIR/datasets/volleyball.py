import os

import numpy as np
import skimage.io
import skimage.transform
import pickle
import json
import cv2

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models
import torch.nn.functional as F

from PIL import Image
import random

import sys

"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
"""

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

NUM_ACTIVITIES = 8

ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']
NUM_ACTIONS = 9


def volley_read_annotations(path):
    """
    reading annotations for the given sequence
    """
    annotations = {}

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    act_to_id = {name: i for i, name in enumerate(ACTIONS)}

    with open(path) as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            file_name = values[0]
            activity = gact_to_id[values[1]]

            values = values[2:]
            num_people = len(values) // 5

            action_names = values[4::5]
            actions = [act_to_id[name]
                       for name in action_names]

            def _read_bbox(xywh):
                x, y, w, h = map(int, xywh)
                return y, x, y + h, x + w

            bboxes = np.array([_read_bbox(values[i:i + 4])
                               for i in range(0, 5 * num_people, 5)])

            fid = int(file_name.split('.')[0])
            annotations[fid] = {
                'file_name': file_name,
                'group_activity': activity,
                'actions': actions,
                'bboxes': bboxes,
            }
    return annotations


def volley_read_dataset(path, seqs):
    data = {}
    for sid in seqs:
        data[sid] = volley_read_annotations(path + '/%d/annotations.txt' % sid)
    return data


def volley_all_frames(data):
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames


def volley_random_frames(data, num_frames):
    frames = []
    for sid in np.random.choice(list(data.keys()), num_frames):
        fid = int(np.random.choice(list(data[sid]), []))
        frames.append((sid, fid))
    return frames


def volley_frames_around(frame, num_before=5, num_after=4):
    sid, src_fid = frame
    return [(sid, src_fid, fid)
            for fid in range(src_fid - num_before, src_fid + num_after + 1)]

def volleyball_readpose(data_path):
    f = open(data_path,'r')
    f = f.readlines()
    pose_ann=dict()
    for ann in f:
        ann = json.loads(ann)
        filename=ann['filename'].split('/')
        sid=filename[-3]
        src_id=filename[-2]
        fid=filename[-1][:-4]
        center = [ann['tmp_box'][0], ann['tmp_box'][1]]
        keypoint=[]
        for i in range(0,51,3):
            keypoint.append(ann['keypoints'][i])
            keypoint.append(ann['keypoints'][i+1])
        pose_ann[sid+src_id+fid+str(center)]=keypoint
    return pose_ann

class VolleyballDataset(data.Dataset):
    """
    Characterize volleyball dataset for pytorch
    """
    def __init__(self, config, transform=None):
        self.config = config
        self.anns = volley_read_dataset(config.data_path_rgb, config.seqs)
        self.frames = volley_all_frames(self.anns)
        self.tracks = pickle.load(open(config.tracks, 'rb'))
        # 在self.anns加入fformation标签
        original_ff = json.load(open(config.data_fformation, 'r', encoding='utf-8'))
        for video_id, v in self.anns.items():
            for frame_id in v.keys():
                ff_key = '{}-{}'.format(video_id, frame_id)
                top_coordinate = np.array(original_ff[ff_key]['top_coordinate'])
                top_coordinate[:, 0] = (top_coordinate[:, 0] - config.top_coordinate_x[0]) / (config.top_coordinate_x[1] - config.top_coordinate_x[0])
                top_coordinate[:, 1] = (top_coordinate[:, 1] - config.top_coordinate_y[0]) / (config.top_coordinate_y[1] - config.top_coordinate_y[0])
                orientation = original_ff[ff_key]['orientation']
                self.anns[video_id][frame_id]['top_coordinate'] = top_coordinate
                self.anns[video_id][frame_id]['orientation'] = orientation
        self.images_path_rgb = config.data_path_rgb
        assert config.sample in ['train', 'val']
        self.sample = config.sample
        assert config.flip in [True, False]
        self.flip = config.flip

        self.pose_anns = volleyball_readpose(config.keypoints)

        self.num_boxes = 12
        self.num_before = 5
        self.num_after = 5

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)

    def __getitem__(self, index):
        """
        Generate one sample of the dataset
        """
        #index = self.root[index]
        select_frames = self.volley_frames_sample(self.frames[index])
        sample = self.load_samples_sequence(select_frames)

        return sample

    def volley_frames_sample(self, frame):

        sid, src_fid = frame
        if self.sample == 'train':
            sample_frames = random.sample(range(src_fid - self.num_before, src_fid), 3) + [src_fid] + \
                            random.sample(range(src_fid + 1, src_fid + self.num_after + 1), 3)
            sample_frames.sort()
        elif self.sample == 'val':
            sample_frames = range(src_fid - 3, src_fid + 4, 1)
        else:
            assert False 

        return [(sid, src_fid, fid) for fid in sample_frames]

    def load_samples_sequence(self, select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        if self.flip and np.random.rand() > 0.5:
            flip = True
        else:
            flip = False

        poses, tracks = [], []
        for i, (sid, src_fid, fid) in enumerate(select_frames):
            img = Image.open(self.images_path_rgb + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
            W, H = img.size

            temp_poses, temp_tracks = [], []
            for i, track in enumerate(self.tracks[(sid, src_fid)][fid]):
                y1, x1, y2, x2 = track

                X1 = int(round(x1 * W))
                Y1 = int(round(y1 * H))
                X2 = int(round(x2 * W))
                Y2 = int(round(y2 * H))

                X1 = min(max(X1, 0), W)
                X2 = min(max(X2, 0), W)
                Y1 = min(max(Y1, 0), H)
                Y2 = min(max(Y2, 0), H)
                center = [(X1 + X2) / 2., (Y1 + Y2) / 2.]
                try:
                    keypoint = self.pose_anns[str(sid) + str(src_fid) + str(fid) + str(center)]
                except:
                    try:
                        center[1] -= 0.5
                        keypoint = self.pose_anns[str(sid) + str(src_fid) + str(fid) + str(center)]
                    except:
                        try:
                            center[0] -= 0.5
                            keypoint = self.pose_anns[str(sid) + str(src_fid) + str(fid) + str(center)]
                        except:
                            center[1] += 0.5
                            keypoint = self.pose_anns[str(sid) + str(src_fid) + str(fid) + str(center)]
                size = np.sqrt((X2 - X1) * (Y2 - Y1) / 4)
                keypoint = np.array(keypoint).reshape(17, 2)
                center = np.array(center)
                keypoint = (keypoint - center) / size

                if flip:
                    keypoint[:, 0] = keypoint[:, 0] * -1.
                    temp_poses.append(keypoint)
                    track[0] = track[0] * -1. + 1.
                    track[2] = track[2] * -1. + 1.
                    temp_tracks.append(track)
                else:
                    temp_poses.append(keypoint)
                    temp_tracks.append(track)
            if len(temp_poses) != self.num_boxes:
                temp_poses = temp_poses + temp_poses[:self.num_boxes - len(temp_poses)]
            temp_poses = np.vstack(temp_poses)
            poses.append(temp_poses)
            if len(temp_tracks) != self.num_boxes:
                temp_tracks = temp_tracks + temp_tracks[:self.num_boxes - len(temp_tracks)]
            temp_tracks = np.vstack(temp_tracks)
            tracks.append(temp_tracks)

        sid = select_frames[0][0]
        src_fid = select_frames[0][1]

        # 处理fformation
        top_coordinates = self.anns[sid][src_fid]['top_coordinate']
        orientations = self.anns[sid][src_fid]['orientation']
        # 数据增强
        if flip:
            top_coordinates[:, 0] = 1 - top_coordinates[:, 0]
            map_dict = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5}
            def map_element(element):
                return map_dict[element]
            orientations = list(map(map_element, orientations))
        # 转换orientation为one-hot编码
        orientations = np.eye(self.config.orientation_num_classes)[orientations]
        # 在voll中，人数少了就补齐
        if orientations.shape[0] != self.num_boxes:
            orientations = np.vstack((orientations, orientations[0: (self.num_boxes - orientations.shape[0])]))
        if top_coordinates.shape[0] != self.num_boxes:
            top_coordinates = np.vstack((top_coordinates, top_coordinates[0: (self.num_boxes - top_coordinates.shape[0])]))

        actions = self.anns[sid][src_fid]['actions']
        activities = self.anns[sid][src_fid]['group_activity']
        if flip:
            activities = (activities + 4) % 8
        if len(actions) != self.num_boxes:
            actions = actions + actions[:self.num_boxes - len(actions)]

        activities = np.array(activities, dtype=np.int32)
        poses = np.vstack(poses).reshape([-1, 17, 2])
        actions = np.hstack(actions).reshape([-1, self.num_boxes])
        tracks = np.vstack(tracks).reshape([-1, self.num_boxes, 4])

        # convert to pytorch tensor
        poses = torch.from_numpy(poses).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()
        top_coordinates = torch.from_numpy(top_coordinates).float()
        orientations = torch.from_numpy(orientations).float()
        tracks = torch.from_numpy(tracks).float()

        return top_coordinates, orientations, poses, activities, actions, tracks
