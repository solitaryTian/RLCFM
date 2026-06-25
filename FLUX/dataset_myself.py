#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse, traceback
import copy, time
import functools
import gc, cv2
import itertools
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Union
from collections import defaultdict

from PIL import Image
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from accelerate.logging import get_logger
from packaging import version
from torch.utils.data import default_collate
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Sampler, BatchSampler, SequentialSampler, RandomSampler
import pytorch_lightning as pl
from itertools import chain, repeat


MAX_SEQ_LENGTH = 77


logger = get_logger(__name__)
def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)






class ComicDatasetBucket(Dataset):
    def __init__(self, file_path, disable_bucket=False,prompt_embeds=None, pooled_prompt_embeds=None, max_size=(1024,1024), divisible=64, stride=16, min_dim=512, base_res=(1024,1024), max_ar_error=4, dim_limit=2048):
        self.disable_bucket = disable_bucket
        if self.disable_bucket:
            print('禁用分桶，全部resize为1024')
            max_ar_error=float('inf')
            min_dim=1024
            dim_limit=1024
            divisible=1



        self.base_res = base_res
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        max_tokens = (max_size[0]/stride) * (max_size[1]/stride)
        self.get_resolution(file_path)  # 从 JSON 文件读取数据
        self.gen_buckets(min_dim, max_tokens, dim_limit, stride, divisible)
        self.assign_buckets(max_ar_error)
        self.gen_index_map()

    def get_resolution(self, file_path):
        # 如果缓存文件存在，直接加载
        if self.disable_bucket:
            file_arr_path = file_path.replace('.json', '_disable_bucket.npy')
            file_arr_path_caption = file_path.replace('.json', '_disable_bucket_caption.npy')
        else:
            file_arr_path = file_path.replace('.json', '.npy')
            file_arr_path_caption = file_path.replace('.json', '_caption.npy')
        if os.path.exists(file_arr_path) and os.path.exists(file_arr_path_caption):
            self.res_map = np.load(file_arr_path, allow_pickle=True).item()
            self.text_map = np.load(file_arr_path_caption, allow_pickle=True).item()
            return

        # 初始化存储字典
        self.res_map = {}
        self.text_map = {}

        # 从 JSON 文件读取数据
        with open(file_path, 'r') as f:
            data = json.load(f)

        print(f'总数据量为{len(data)}')


    # 判断 JSON 数据的类型
        if isinstance(data, dict):  # 字典形式
            for each_file_path, info in tqdm(data.items()):
                original_size = info["original_image_size"]  # [宽, 高]
                caption = info["caption"]  # 图片标签

                # 存储图像尺寸和标签
                self.res_map[each_file_path] = tuple(original_size)  # (宽, 高)
                self.text_map[each_file_path] = caption
        elif isinstance(data, list):  # 列表形式

            for item in tqdm(data):  # 直接遍历列表
                image_path = item["image_path"]  # 图片路径
                original_size = item["size"]  # [宽, 高]
                if ('caption' in item) and (item['caption'] != None):
                    caption = item["caption"]  # 图片标签
                elif ('wd_tag' in item) and (item['wd_tag'] != None):
                    caption = item["wd_tag"]
                else:
                    caption = ''

                
                # 存储图像尺寸和标签
                self.res_map[image_path] = tuple(original_size)  # (宽, 高)
                self.text_map[image_path] = caption
        else:
            raise ValueError("Unsupported JSON format. Expected a dictionary or a list.")


        if self.disable_bucket:
            np.save(file_path.replace('.json', '_disable_bucket.npy'), np.array(self.res_map))
            np.save(file_path.replace('.json', '_disable_bucket_caption.npy'), np.array(self.text_map))
        else:
            # 保存缓存文件
            np.save(file_path.replace('.json', '.npy'), np.array(self.res_map))
            np.save(file_path.replace('.json', '_caption.npy'), np.array(self.text_map))

    def gen_buckets(self, min_dim, max_tokens, dim_limit, stride=8, div=64):
        resolutions = []
        aspects = []
        w = min_dim
        while (w/stride) * (min_dim/stride) <= max_tokens and w <= dim_limit:
            h = min_dim
            got_base = False
            while (w/stride) * ((h+div)/stride) <= max_tokens and (h+div) <= dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                h += div
            if (w != self.base_res[0] or h != self.base_res[1]) and got_base:
                resolutions.append(self.base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            w += div
        h = min_dim
        while (h/stride) * (min_dim/stride) <= max_tokens and h <= dim_limit:
            w = min_dim
            got_base = False
            while (h/stride) * ((w+div)/stride) <= max_tokens and (w+div) <= dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                w += div
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            h += div

        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]

        self.resolutions = sorted(res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.aspects = np.array(list(map(lambda x: res_map[x], self.resolutions)))
        self.resolutions = np.array(self.resolutions)

    def assign_buckets(self, max_ar_error=4):
        self.buckets = {}
        self.aspect_errors = []
        self.res_map_new = {}

        skipped = 0
        skip_list = []
        for post_id in self.res_map.keys():
            w, h = self.res_map[post_id]
            aspect = float(w)/float(h)
            bucket_id = np.abs(self.aspects - aspect).argmin()
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            error = abs(self.aspects[bucket_id] - aspect)
            if error < max_ar_error:
                self.buckets[bucket_id].append(post_id)
                self.res_map_new[post_id] = tuple(self.resolutions[bucket_id])
            else:
                skipped += 1
                skip_list.append(post_id)
        for post_id in skip_list:
            del self.res_map[post_id]

    def gen_index_map(self):
        self.id2path = {}
        self.id2shape = {}
        id = 0
        for path, shape in self.res_map_new.items():
            self.id2path[id] = path
            self.id2shape[id] = shape
            id += 1

    def __len__(self):
        return len(self.res_map)



    def __getitem__(self, idx):
        while True:
            try:
                target_path = self.id2path[idx]
                W, H = self.res_map_new[target_path]
                text = self.text_map[target_path]
                target_path = target_path.strip()

                # 加载图像
                target = cv2.imread(target_path)
                if target is None:
                    raise ValueError(f"Unable to read image at path: {target_path}")
                
                ori_H, ori_W, _ = target.shape

                target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
                target = cv2.resize(target, (W, H))
                target = target.transpose((2,0,1))

                # 归一化到 [-1, 1]
                target = (target.astype(np.float32) / 127.5) - 1.0

                return dict(pixel_values=target, original_sizes=(ori_W, ori_H), crop_top_lefts=(0, 0), target_sizes=(W, H), caption=text)
            
            except Exception as e:
                traceback.print_exc()
                print(f"Skipping sample {idx} due to error: {e}, path: {target_path}")
                
                # 从当前桶中重新选择一个样本
                bucket_id = self.get_bucket_id(idx)
                if bucket_id is not None:
                    new_idx = random.choice(self.buckets[bucket_id])
                    idx = new_idx
                else:
                    idx = random.randint(0, len(self.id2path) - 1)  # 如果没有找到桶，随机选择一个样本

    def get_bucket_id(self, idx):
        target_path = self.id2path[idx]
        W, H = self.res_map_new[target_path]
        aspect = float(W) / float(H)
        bucket_id = np.abs(self.aspects - aspect).argmin()
        return bucket_id if bucket_id in self.buckets else None




    # def __getitem__(self, idx):
    #     target_path = self.id2path[idx]
    #     W, H = self.res_map_new[target_path]
    #     text = self.text_map[target_path]
    #     target_path = target_path.strip()

    #     # 加载图像
    #     try:
    #         target = cv2.imread(target_path)
    #         ori_H, ori_W, _ = target.shape
    #     except Exception as e:
    #         traceback.print_exc()
    #         print(f"Skipping sample {idx} due to error: {e}")
    #         return None  # 返回 None 表示该样本有问题

    #     target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    #     target = cv2.resize(target, (W, H))
    #     target = target.transpose((2,0,1))

    #     # 归一化到 [-1, 1]
    #     target = (target.astype(np.float32) / 127.5) - 1.0

    #     return dict(pixel_values=target, original_sizes=(ori_W, ori_H), crop_top_lefts=(0, 0), target_sizes=(W, H), caption=text)



class GroupedBatchSampler(BatchSampler):
    # def __init__(self, sampler, batch_size, drop_last=True):
    def __init__(self, sampler, dataset, batch_size, drop_last=True):
        if not isinstance(sampler, Sampler):
            raise ValueError(f"sampler should be an instance of torch.utils.data.Sampler, but got sampler={sampler}")
        self.sampler = sampler
        # self.group_ids = self.sampler.dataset.id2shape
        self.group_ids = dataset.id2shape
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            for group_id, _ in sorted(buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size
    
# using LightningDataModule
class ComicDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, file_txt, disable_bucket=False,prompt_embeds=None, pooled_prompt_embeds=None):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        # self.dataset = dataset
        self.file_txt = file_txt
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.disable_bucket = disable_bucket
    
    def setup(self, stage):
        self.dataset = ComicDatasetBucket(file_path=self.file_txt, prompt_embeds=self.prompt_embeds, pooled_prompt_embeds=self.pooled_prompt_embeds,disable_bucket=self.disable_bucket)
        self.sampler = SequentialSampler(self.dataset)

    def __len__(self):
        return len(self.dataset.res_map)


    def train_dataloader(self):
        def collate_fn(examples):
            examples = [sample for sample in examples if sample is not None]
            pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            original_sizes = [example["original_sizes"] for example in examples]
            crop_top_lefts = [example["crop_top_lefts"] for example in examples]
            target_sizes = [example["target_sizes"] for example in examples]
            caption = [example["caption"] for example in examples]
            # prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
            # pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

            return {
                "pixel_values": pixel_values,
                # "prompt_embeds": prompt_embeds,
                # "pooled_prompt_embeds": pooled_prompt_embeds,
                "original_sizes": original_sizes,
                "crop_top_lefts": crop_top_lefts,
                "target_sizes": target_sizes,
                "caption": caption
            }
        # return DataLoader(self.dataset, batch_sampler=GroupedBatchSampler(sampler=self.sampler, batch_size=self.batch_size), num_workers=32, collate_fn=collate_fn)
        return DataLoader(self.dataset, batch_sampler=GroupedBatchSampler(sampler=self.sampler, dataset=self.dataset, batch_size=self.batch_size), num_workers=0, collate_fn=collate_fn, pin_memory=True, persistent_workers=False)

def analyze_buckets(dataset):
    # 查看桶的总数
    num_buckets = len(dataset.buckets)
    print(f"Total number of buckets: {num_buckets}")

    # 查看每个桶的分辨率和图像数量
    for bucket_id, image_ids in dataset.buckets.items():
        resolution = dataset.resolutions[bucket_id]
        num_images = len(image_ids)
        print(f"Bucket {bucket_id} (Resolution: {resolution}) contains {num_images} images")

if __name__ == "__main__":
    # file_path = '/maindata/data/shared/public/nuo.pang/data/images/1000w_text_info_new.json'
    # file_path = '/maindata/data/shared/public/songtao.tian/test_code/my/image_info_20250118.json'
    # file_path = '/maindata/data/shared/public/songtao.tian/data/filter_aesV25_5.5_maxSize_1000_317w_0603_wangwei_after_format_filter_20250115_delete.json'
    file_path = '/maindata/data/shared/public/songtao.tian/data/meta_aesV25_7.0_maxSize_1000_0603_wangwei.meta_two_term.json'

    # 示例用法
    dataset = ComicDatasetBucket(file_path=file_path)
    analyze_buckets(dataset)




    data_module = ComicDataModule(batch_size=2, file_txt=file_path)
    data_module.setup(stage="fit")
    train_dataloader = data_module.train_dataloader()

    for step, batch in enumerate(tqdm(train_dataloader)):
        image, text, orig_size, crop_coords, target_sizes  = batch['pixel_values'], batch['caption'], batch['original_sizes'], batch['crop_top_lefts'], batch['target_sizes']     
        print(f'image.shape={image.shape}')







    # import json
    # from collections import Counter

    # # 打开 JSON 文件
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)  # 读取 JSON 数据

    # # 统计字典关键字的出现次数
    # dict_counter = Counter()

    # # 遍历列表中的每个字典
    # for item in data:
    #     # 获取字典的键（关键字）
    #     keys = frozenset(item.keys())  # 使用 frozenset 作为不可变键
    #     dict_counter[keys] += 1

    # # 输出结果
    # print(f"Total unique dictionaries: {len(dict_counter)}")
    # print("Dictionary counts:")
    # for keys, count in dict_counter.items():
    #     print(f"{keys}: {count}")