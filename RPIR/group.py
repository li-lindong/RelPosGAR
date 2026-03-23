import time
import copy
import torch
import random
import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

from pathlib import Path
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from RPIR.models import ModelBuilder
from RPIR.datasets import dataset_entry
from RPIR.utils.log_helper import init_log
from RPIR.utils.utils import save_checkpoint, AverageMeter, print_speed, load_model, load_DDPModel
import pprint
from RPIR.utils.top_k_acc import top_k_accuracy

init_log('group')
logger = logging.getLogger('group')


class Group():
    def __init__(self, config, work_dir):
        self.config = EasyDict(config)
        self.save_path = Path(work_dir)
        self.start_epoch = 0
        self.total_epochs = self.config.train.scheduler.epochs

        self._build()

    def _build(self):
        self._build_seed()
        self._build_dir()
        self._build_model()
        self._build_optimizer()
        self._build_datasetLoader()
        self._build_scheduler()
        self._build_criterion()
        self._build_tb_logger()

    def _build_seed(self):
        seed = self.config.common.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_dir(self):
        self.checkPointPath = self.save_path.joinpath('checkpoints')
        self.saveLogPath = self.save_path.joinpath('log')
        self.saveResultsPath = self.save_path.joinpath('results')
        self.tbSavePath = Path(self.save_path).joinpath('tb_logger')
        self.nartSavePath = self.save_path.joinpath('nart')
        # if self.rank == 0:
        if not self.checkPointPath.exists():
            self.checkPointPath.mkdir(parents=True)
        if not self.saveLogPath.exists():
            self.saveLogPath.mkdir(parents=True)
        if not self.saveResultsPath.exists():
            self.saveResultsPath.mkdir(parents=True)
        if not self.tbSavePath.exists():
            self.tbSavePath.mkdir(parents=True)
        if not self.nartSavePath.exists():
            self.nartSavePath.mkdir(parents=True)

    def _build_model(self):
        config = self.config
        model = ModelBuilder(config)
        self.model = model.cuda()

    def _build_optimizer(self):
        config = self.config
        model = self.model
        try:
            optim = getattr(torch.optim, config.train.optimizer.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' +
                                      config.train.optimizer.type)
        optimizer = optim(filter(lambda p: p.requires_grad, model.parameters()), **config.train.optimizer.kwargs)
        self.optimizer = optimizer

    def _build_datasetLoader(self):
        assert self.config.dataset.name in ['volleyball', 'nba']
        if self.config.dataset.name == 'volleyball':
            train_loader, val_loader = dataset_entry(self.config.dataset.volleyball, self.config.dataset.name, evaluate=False, test_img=False)
        elif self.config.dataset.name == 'nba':
            train_loader, val_loader = dataset_entry(self.config.dataset.nba, self.config.dataset.name, evaluate=False, test_img=False)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _build_scheduler(self):
        config = self.config
        self.lr_scheduler = MultiStepLR(
            self.optimizer,
            milestones = config.train.scheduler.milestones,
            gamma = config.train.scheduler.gamma,
            last_epoch = self.start_epoch - 1)

    def _build_criterion(self):
        config = self.config
        if config.train.criterion == 'ce_loss':
            if config.train.action_loss:
                actions_weight = torch.tensor(config.train.actions_weight).cuda().detach()
                self.actions_criterion = nn.CrossEntropyLoss(weight=actions_weight)
            activities_weight = torch.tensor(config.train.activities_weight).cuda().detach()
            self.activities_criterion = nn.CrossEntropyLoss(weight=activities_weight)
        else:
            raise NotImplementedError('not implemented criterion ' + config.criterion)

    def get_dump_dict(self):
        return {
            'config': copy.deepcopy(self.config),
            'epoch': self.epoch + 1,
            'optimizer': self.lr_scheduler.optimizer.state_dict(),
            'state_dict': self.model.state_dict(),
        }

    def _build_tb_logger(self):
        self.tb_logger = SummaryWriter(self.tbSavePath)


    @staticmethod
    def to_device(input, device="cuda"):
        """Transfer data between devidces"""

        def transfer(x):
            if torch.is_tensor(x):
                return x.to(device=device)
            elif isinstance(x, list) and torch.is_tensor(x[0]):
                return [_.to(device=device) for _ in x]
            return x

        if isinstance(input, dict):
            return {k: transfer(v) for k, v in input.items()}
        return [transfer(k) for k in input]

    def get_batch(self, batch_type='train'):
        """
        Return the batch of the given batch_type.
        The valid batch_type is set in config
        The returned batch will be used to call `forward` function of SpringCommonInterface.
        The first item will be used to forward model like: model(get_batch('train')[0])
        self:
            batch_type: str. default: 'train'. It can also be 'val', 'test' or other custom type.

        Returns:
            a tuple of batch (input, label)
        """
        iter_name = batch_type + '_iterator'
        loader_name = batch_type + '_loader'

        def get_iterator():
            loader = getattr(self, loader_name)
            iterator = iter(loader)
            return iterator

        if not hasattr(self, iter_name):
            iterator = get_iterator()
            setattr(self, iter_name, iterator)
        else:
            iterator = getattr(self, iter_name)

        try:
            batch = next(iterator)
        except StopIteration as e:  # noqa
            iterator = get_iterator()
            setattr(self, iter_name, iterator)
            batch = next(iterator)
        batch = self.to_device(batch)
        return batch

    def update(self):
        self.lr_scheduler.optimizer.step()

    def forward(self, batch):
        output = self.model(batch[0], batch[1], batch[2], batch[5])

        loss = []

        activities = output['activities_scores']
        target_activities = batch[3].view(-1)
        if isinstance(activities, list):
            # activities_loss = sum([self.activities_criterion(activity, target_activities) for activity in activities]) / self.world_size
            activities_loss = sum([self.activities_criterion(activity, target_activities) for activity in activities])
        else:
            # activities_loss = self.activities_criterion(activities, target_activities) / self.world_size
            activities_loss = self.activities_criterion(activities, target_activities)
        loss.append(activities_loss)

        if self.config.train.action_loss:
            actions = output['actions_scores']
            actions = torch.reshape(actions, (actions.shape[0] * actions.shape[1], -1))
            target_actions = batch[4].view(-1)
            actions_loss = self.actions_criterion(actions, target_actions)
            loss.append(actions_loss)

        loss = sum(loss) / len(loss)

        return activities_loss, loss

    def backward(self, loss):
        reduced_loss = loss.clone()
        self.lr_scheduler.optimizer.zero_grad()
        loss.backward(retain_graph=False)
        return reduced_loss

    def train(self):
        # rank = self.rank
        config = self.config
        model = self.model
        # lr_scheduler = self.lr_scheduler
        self.epoch = 0
        for epoch in range(self.start_epoch, self.total_epochs):
            self.epoch = epoch
            self.lr = self.lr_scheduler.get_lr()[0]
            # self.train_loader.sampler.set_epoch(epoch)
            self.train_epoch()
            self.val()
            # if rank == 0:
            save_checkpoint(self.get_dump_dict(), self.checkPointPath)

    def train_epoch(self):
        model = self.model
        self.train_epoch_iters = len(self.train_loader)
        for batch_idx in range(self.train_epoch_iters):
            # train for one epoch
            batch_time = AverageMeter()
            data_time = AverageMeter()
            activities_losses = AverageMeter()
            losses = AverageMeter()
            end = time.time()
            model.train()
            # time1 = time.time()
            batch = self.get_batch("train")
            # time2 = time.time()
            data_time.update(time.time() - end)
            # time3 = time.time()
            activities_loss, loss = self.forward(batch)
            # time4 = time.time()
            reduced_loss = self.backward(loss)
            # time5 = time.time()

            # print('获取数据时间：', time2-time1, '前向推理时间：', time4-time3, '反向传播时间：', time5-time4)

            activities_losses.update(activities_loss.item())
            losses.update(reduced_loss.item())

            self.update()
            batch_time.update(time.time() - end)
            end = time.time()
            self.print_info(batch_idx, activities_losses, reduced_loss, batch_time,
                            data_time, losses, self.lr)

        self.lr_scheduler.step()

    def print_info(self, idx, activities_losses, reduced_loss, batch_time, data_time,
                   losses, lr):
        config = self.config
        tb_logger = self.tb_logger
        # if idx % config.train.print_freq == 0 and self.rank == 0:
        if idx % config.train.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})'
                        ' LR {lr:.8f}\t'
                        'Activities_Loss {activity_loss.val:.4f} ({activity_loss.avg:.4f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                self.epoch,
                idx,
                self.train_epoch_iters,
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                activity_loss=activities_losses,
                lr=lr))
            curr_step = self.epoch * self.train_epoch_iters + idx + 1
            print_speed(curr_step, batch_time.val, self.total_epochs * self.train_epoch_iters)
            tb_logger.add_scalar('LR', lr, curr_step)
            tb_logger.add_scalar('data_time', data_time.avg, curr_step)
            tb_logger.add_scalar('batch_time', batch_time.avg, curr_step)
            tb_logger.add_scalar('Activities_Loss_train', activities_losses.avg, curr_step)
            tb_logger.add_scalar('Loss_train', losses.avg, curr_step)
            # tb_logger.add_scalar('Acc', acces.avg, curr_step)

    @torch.no_grad()
    def val(self):
        model = self.model
        tb_logger = self.tb_logger
        # rank = self.rank
        model.eval()

        num_activities_true = torch.tensor(0).cuda()
        num_activities_total = torch.tensor(0).cuda()

        if self.config.dataset.name == 'volleyball':
            activities_num_classes = 8
        elif self.config.dataset.name == 'nba':
            activities_num_classes = 9
        confusion_matrix = torch.zeros(activities_num_classes, activities_num_classes).cuda()

        y_true, y_pred = [], [] # 整个测试集的标签和预测结果，用于计算准确率
        self.val_epoch_iter = len(self.val_loader)

        # 保存个体特征与个体动作标签，群组特征与群组活动标签
        ind_features, gt_actions, group_features, gt_activities = [], [], [], []
        for batch_idx in range(self.val_epoch_iter):
            batch = self.get_batch('val')
            target_activities = batch[3].view(-1)
            output = model(batch[0], batch[1], batch[2], batch[5])
            activities = output['activities_scores']
            if isinstance(activities, list):
                activities = sum(activities)
            activities = F.softmax(activities, dim=1)
            pred_activities = torch.argmax(activities, dim=1)

            y_true.append(target_activities)
            y_pred.append(activities)

            num_activities_true += (pred_activities == target_activities).sum()
            num_activities_total += target_activities.numel()

            for pred_acty, acty in zip(pred_activities.view(-1), target_activities):
                acty = int(acty)
                pred_acty = int(pred_acty)
                confusion_matrix[acty][pred_acty] += 1

            # 保存个体特征与个体动作标签
            # ind_features_batch = torch.reshape(output['ind_features'], [-1, 128])
            # ind_features.append(ind_features_batch.cpu().numpy())
            gt_actions_batch = torch.reshape(batch[4], [-1])
            gt_actions.append(gt_actions_batch.cpu().numpy())
            group_features.append(output['group_features'].cpu().numpy())
            gt_activities.append(batch[3].cpu().numpy())

        # 计算top-1和top-5
        y_true = np.array(torch.cat(y_true, dim=0).cpu())
        y_pred = np.array(torch.cat(y_pred, dim=0).cpu())
        top_1_acc = top_k_accuracy(y_true, y_pred, k=1)
        top_5_acc = top_k_accuracy(y_true, y_pred, k=5)

        # if rank == 0:
        activities_acc = num_activities_true.float() / num_activities_total.float()
        tb_logger.add_scalar('Validation Activity Accuracy', activities_acc, self.epoch)
        print('Epoch: [%d] '
              'Activity Accuracy %.3f%%\t'
              'Top-1 Accuracy %.3f%%\t'
              'Top-5 Accuracy %.3f%%\t'
              % (self.epoch, activities_acc * 100, top_1_acc * 100, top_5_acc * 100))
        confusion_matrix = confusion_matrix.cpu().numpy()
        confusion_matrix_normalize = confusion_matrix / \
                                     np.sum(confusion_matrix, axis=1, keepdims=True)
        confusion_matrix_normalize = np.round(confusion_matrix_normalize, 4)
        if self.config.evaluate:
            pprint.pprint(confusion_matrix_normalize)
            #with open('volleyball_confusion_matrix.npy', 'wb') as f:
            #    np.save(f, confusion_matrix)

            # 保存个体特征与个体动作标签，群组特征和群组活动标签
            # dir = os.path.join('/', *self.config.checkpoint.split('/')[:-1])
            # np.save(os.path.join(dir, 'ind_features.npy'), np.concatenate(ind_features, axis=0))
            # np.save(os.path.join(dir, 'gt_actions.npy'), np.concatenate(gt_actions, axis=0))
            # np.save(os.path.join(dir, 'group_features.npy'), np.concatenate(group_features, axis=0))
            # np.save(os.path.join(dir, 'gt_activities.npy'), np.concatenate(gt_activities, axis=0))
