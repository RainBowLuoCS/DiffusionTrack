#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.distributed as dist

from yolox.utils import synchronize

import random


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader,task):
        self.loader = iter(loader)
        self.task=task
        self.stream = torch.cuda.Stream()
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            if self.task=="tracking":
                self.next_input_pre, self.next_target_pre,self.next_input_cur, self.next_target_cur,_, _ = next(self.loader)
            else:
                self.next_input_pre, self.next_target_pre, _, _ = next(self.loader)
        except StopIteration:
            self.next_input_pre = None
            self.next_target_pre = None
            if self.task=="tracking":
                self.next_input_cur = None
                self.next_target_cur = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input_pre = self.next_input_pre.cuda(non_blocking=True)
            self.next_target_pre = self.next_target_pre.cuda(non_blocking=True)
            if self.task=="tracking":
                self.next_input_cur = self.next_input_cur.cuda(non_blocking=True)
                self.next_target_cur = self.next_target_cur.cuda(non_blocking=True)
                

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input_pre = self.next_input_pre
        target_pre = self.next_target_pre
        input_cur = None
        target_cur = None
        if self.task=="tracking":
            input_cur = self.next_input_cur
            target_cur = self.next_target_cur
        if input_pre is not None:
            self.record_stream(input_pre)
        if target_pre is not None:
            target_pre.record_stream(torch.cuda.current_stream())
        if self.task=="tracking":
            if input_cur is not None:
                self.record_stream(input_cur)
            if target_cur is not None:
                target_cur.record_stream(torch.cuda.current_stream())
        self.preload()
        return input_pre,target_pre,input_cur,target_cur
        

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


def random_resize(data_loader, exp, epoch, rank, is_distributed):
    tensor = torch.LongTensor(1).cuda()
    if is_distributed:
        synchronize()

    if rank == 0:
        if epoch > exp.max_epoch - 10:
            size = exp.input_size
        else:
            size = random.randint(*exp.random_size)
            size = int(32 * size)
        tensor.fill_(size)

    if is_distributed:
        synchronize()
        dist.broadcast(tensor, 0)

    input_size = data_loader.change_input_dim(multiple=tensor.item(), random_range=None)
    return 