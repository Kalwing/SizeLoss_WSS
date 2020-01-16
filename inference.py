#!/usr/bin/env python3.6

import argparse
import warnings
from typing import List
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader import SliceDataset
from utils import save_images, map_, tqdm_, probs2class, uniq
import os


def runInference(args: argparse.Namespace):
    print( [Path(args.data_folder)])
    print('>>> Loading model')

    losses_list = [
        ('CrossEntropy', {'idc': [1]}, None, None, None, 1),
        ('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [60, 9000]}, 'idc': [1]}, 'soft_size', 1e-2)
    ]
    bounds_generators: List[List[Callable]] = []
    for losses in losses_list:
        tmp = []
        print(len(losses))
        _, _, bounds_name, bounds_params, fn, _ = losses
        if bounds_name is None:
            tmp.append(lambda *a: torch.zeros(n_class, 1, 2))
            continue

        bounds_class = getattr(__import__('bounds'), bounds_name)
        tmp.append(bounds_class(C=args.num_classes, fn=fn, **bounds_params))
    bounds_generators.append(tmp)


    net = torch.load(args.model_weights)
    device = torch.device("cuda")
    net.to(device)

    print('>>> Loading the data')
    batch_size: int = args.batch_size
    num_classes: int = args.num_classes

    transform = transforms.Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    folders: List[Path] = [Path(os.path.join(args.data_folder, folder)) for folder in os.listdir(args.data_folder)]
    names: List[str] = [str(p.name) for p in folders[0].glob("*.png")]
    print(folders, folders[0].glob("*.png"), '->', names)
    dt_set = SliceDataset(names,
                          folders,
                          are_hots=[False],  # T: are_hots required. In training, in the make it's alwas True except for the img
                          bounds_generators=bounds_generators,  # T: [bounds_name(C=n_classes, fn=fn, **bounds_params), ...] but isn't it encoded in the pkl ??
                          transforms=[transform],
                          debug=False,
                          C=num_classes)
    assert len(dt_set) > 0
    loader = DataLoader(dt_set,
                        batch_size=batch_size,
                        num_workers=batch_size + 2,
                        shuffle=False,
                        drop_last=False)
    assert len(loader) > 0
    print('>>> Starting the inference')
    savedir: str = args.save_folder
    total_iteration = len(loader)
    print(len(loader))
    desc = f">> Inference"
    tq_iter = tqdm_(enumerate(loader), total=total_iteration, desc=desc)
    with torch.no_grad():
        for j, (filenames, image, _) in tq_iter:
            print(filenames)
            image = image.to(device)

            pred_logits: Tensor = net(image)
            pred_probs: Tensor = F.softmax(pred_logits, dim=1)

            # with warnings.catch_warnings():
            #     # warnings.simplefilter("ignore")
            predicted_class: Tensor = probs2class(pred_probs)
            save_images(predicted_class, filenames, savedir, "", 0)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inference parameters')
    parser.add_argument('--data_folder', type=str, required=True, help="The folder containing the images to predict")
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--model_weights', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()

    print(args)

    return args


if __name__ == '__main__':
    args = get_args()
    runInference(args)
