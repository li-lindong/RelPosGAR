import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
torch.autograd.set_detect_anomaly(True)
import yaml
import argparse
from pathlib import Path
from pprint import pprint
from RPIR.group import Group
from datetime import datetime
import random
import numpy as np

# lld
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证 cuDNN 确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """
    parse args
    :return:args
    """
    new_parser = argparse.ArgumentParser(description='PyTorch Density parser..')
    new_parser.add_argument('--config', default="./config/RPIR.yaml", help='model config file path')
    new_parser.add_argument('--checkpoint', default=None, help='the checkpoint file')
    new_parser.add_argument('--resume', default=None, help='the checkpoint file to resume from')
    new_parser.add_argument('--evaluate', action='store_true', default=False, help='train or test')
    return new_parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['checkpoint'] = args.checkpoint
    config['resume'] = args.resume
    config['evaluate'] = args.evaluate

    # 存储实验结果的文件夹
    config['basedir'] = os.getcwd() + '/experiments_{}/'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + Path(args.config).resolve().stem
    pprint(config)

    # lld 测试
    config['evaluate'] = True
    # config['checkpoint'] = None
    config['checkpoint'] = r"/mnt/disk/LiLinDong/RPIR/results/rpir_128_4_6_4_6_add_x_N_topx_N_late_GA_N/checkpoint_50.pth.tar"

    print("个体坐标维度：{}".format(config['structure']['ind_rope']['coord_axis']))

    group_helper = Group(config, work_dir=config['basedir'])

    if config['evaluate']:
        print("测试")
        group_helper.epoch = 0
        group_helper.val()
    else:
        print("训练")
        group_helper.train()


if __name__ == '__main__':
    set_seed(666)
    main()
