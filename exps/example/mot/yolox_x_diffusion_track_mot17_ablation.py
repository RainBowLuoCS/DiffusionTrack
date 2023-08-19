# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train_half.json"
        self.val_ann = "val_half.json"
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        self.max_epoch = 30
        self.print_interval = 20
        self.eval_interval = 5
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.task="tracking"
        self.enable_mixup = True
        self.seed=8823
        self.conf_thresh=0.25
        self.det_thresh=0.7
        self.nms_thresh2d=0.75
        self.nms_thresh3d=0.7
        self.interval=5

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            DiffusionMosaicDetection,
            DiffusionTrainTransform
        )

        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "mot"),
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = DiffusionMosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=DiffusionTrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000, 
            ),
            degrees=self.degrees, 
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset,DiffusionValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "mot"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='train',
            preproc=DiffusionValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000, 
            )
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.conf_thresh,
            nmsthre3d=self.nms_thresh3d,
            detthre=self.det_thresh,
            nmsthre2d=self.nms_thresh2d,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
    
    def get_model(self):
        from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
        from diffusion.models.diffusionnet import DiffusionNet,DiffusionHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            for value in backbone.parameters():
                value.requires_grad=False
            head=DiffusionHead(self.num_classes,self.width)
            self.model = DiffusionNet(backbone, head)

        self.model.apply(init_yolo)
        # self.model.head.initialize_biases(1e-2)
        return self.model

    def get_optimizer(self, batch_size):
        lr=2.5e-05
        weight_decay = 0.0001
        self.optimizer=AdamW(self.model.parameters(),lr=lr,weight_decay=weight_decay) 
        return self.optimizer
