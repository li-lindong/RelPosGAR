from torch.utils import data
import json
import pickle
import torch
import numpy as np
from triton.language import dtype


class NBA_Dataset(data.Dataset):
    def __init__(self, config):
        super(NBA_Dataset, self).__init__()
        self.config = config
        self.skeleton_data = np.load(config.data_skeleton, mmap_mode='r')
        with open(config.data_label, 'rb') as f:
            self.label = pickle.load(f)

        assert config.flip in [True, False]
        self.flip = config.flip
        self.num_boxes = 12
        pass

    def __getitem__(self, index):
        if self.flip and np.random.random() > 0.5:
            flip = True
        else: flip = False

        """读取数据"""
        top_coordinates = torch.rand([12, 2], dtype=torch.float32)
        orientations = torch.rand([12, 8], dtype=torch.float32)
        poses = torch.tensor(self.skeleton_data[index], dtype=torch.float32)[:, :, :, :2]
        labels = self.label[index]
        activities = torch.tensor(labels[0], dtype=torch.int64)
        seq_name, img_WH = labels[1], labels[2]
        actions = torch.randint(low=0, high=9, size=(1, 12), dtype=torch.int32)

        """归一化骨架数据与个体中心位置"""
        # 均匀取7帧数据
        indices = np.linspace(0, poses.shape[0] - 1, 7, dtype=int)
        selected_poses = poses[indices]
        # 归一化骨架
        max_xy = torch.max(selected_poses, dim=-2)[0]
        min_xy = torch.min(selected_poses, dim=-2)[0]
        tracks = torch.cat((min_xy, max_xy), dim=-1)
        tracks = tracks / torch.tensor((img_WH + img_WH), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        center = (max_xy + min_xy) / 2.0
        diff_xy = max_xy - min_xy
        epsilon = 1e-6
        width = diff_xy[:, :, 0].clamp(min=epsilon)
        height = diff_xy[:, :, 1].clamp(min=epsilon)
        size = torch.sqrt((width * height) / 4.0)
        selected_poses_normalized = (selected_poses - center.unsqueeze(-2)) / size.unsqueeze(-1).unsqueeze(-1)
        # 随机翻转
        # if flip:
        #     selected_poses_normalized[:, :, :, 0] = selected_poses_normalized[:, :, :, 0] * (-1.0)
        # else: pass

        selected_poses_normalized = torch.reshape(selected_poses_normalized, (-1, 17, 2))

        return top_coordinates, orientations, selected_poses_normalized, activities, actions, tracks


    def __len__(self):
        return len(self.skeleton_data)