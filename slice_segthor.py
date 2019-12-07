#!/usr/bin/env python3.6

import re
import random
import argparse
import warnings
from pathlib import Path
from pprint import pprint
from functools import partial
from typing import Any, Callable, List, Tuple

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
from numpy import unique as uniq
from skimage.io import imread, imsave, imshow, show
from skimage.transform import resize

from glob import glob
import os

from utils import mmap_, uc_, map_, augment


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    return res.astype(np.uint8)


def get_p_id(path: Path, regex: str = "(\w+_)(\d+)", get_z_size: bool = False) -> str:
    matched = re.match(regex, path.stem)
    imgs = glob(os.path.join(str(path.parent), matched.group(1) + '*' + '.png'))
    z_size = len(imgs)
    if matched:
        if get_z_size:
            return matched.group(1), z_size
        return matched.group(1)
    raise ValueError(regex, "didn't matched", path.stem, "in", path)


def save_slices(img_p: Path, gt_p: Path,
                dest_dir: Path, shape: Tuple[int], n_augment: int,
                img_dir: str = "img", gt_dir: str = "gt") -> Tuple[int, int, int]:
    p_id, z_size = get_p_id(img_p, get_z_size=True)
    # assert "Case" in p_id
    assert p_id == get_p_id(gt_p)
    print("p_id:", p_id, " z size:", z_size)
    # Load the data as 3d array

    resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)
    # Most img aren't the same size so we resize all of them to 512,512
    img = []
    for p in range(z_size):
        slice_path = os.path.join(img_p.parent, p_id + str(p) + str(img_p.suffix))
        if not os.path.exists(slice_path):  # Andbru13 is missing so we have to skip it
            continue
        slice = imread(
            slice_path,
            plugin='simpleitk', as_gray=True
        )
        img.append(resize_(slice*255, shape))

    assert np.all(np.array([image.shape for image in img]) == img[0].shape), F"All images aren't the same sizes for {p_id}"
    assert np.max(img) > 1
    try:
        img = np.array(img, dtype=np.int16)
    except ValueError:
        print(F"Error loading patient images {p_id}")
        raise ValueError
    # imshow(img[0])
    # show()
    gt = []
    for p in range(z_size):
        slice_path = os.path.join(gt_p.parent, p_id + str(p) + str(gt_p.suffix))
        if not os.path.exists(slice_path):  # Andbru13 is missing so we have to skip it
            continue
        slice = imread(
            slice_path,
            plugin='simpleitk', as_gray=True
        )
        slice = slice/slice.max()  # T: I guess 1 is the max here, see l132
        gt.append(resize_(slice, shape))
    assert np.all(np.array([image.shape for image in gt]) == gt[0].shape), F"All ground truths aren't the same sizes for {p_id}"
    try:
        gt = np.array(gt, dtype=np.uint8)
    except ValueError:
        print(F"Error loading patient ground truth {p_id}")
        raise ValueError

    assert img.shape == gt.shape
    # T: Don't get why they do that ?!
    assert img.dtype in [np.uint8, np.int16], img.dtype
    assert gt.dtype in [np.uint8], gt.dtype
    #
    # img_nib = sitk.ReadImage(str(img_p))
    # dx, dy = img_nib.GetSpacing()
    # print(dx, dy, dz)
    # assert np.abs(dx - dy) <= 0.0000041, (dx, dy, dx - dy)
    # assert 0.27 <= dx <= 0.75, dx
    # assert 2.19994 <= dz <= 4.00001, dz
    x, y, z = img.shape
    # assert (y, z) in [(320, 320), (512, 512), (256, 256), (384, 384)], (y, z)
    # assert 15 <= x <= 54, x

    # Normalize and check data content
    norm_img = norm_arr(img)  # We need to normalize the whole 3d img, not 2d slices
    assert 0 == norm_img.min() and norm_img.max() == 255, (norm_img.min(), norm_img.max())
    assert norm_img.dtype == np.uint8

    save_dir_img: Path = Path(dest_dir, img_dir)
    save_dir_gt: Path = Path(dest_dir, gt_dir)
    sizes_2d: np.ndarray = np.zeros(img.shape[-1])
    for j in range(len(img)):
        img_s = norm_img[j, :, :]
        gt_s = gt[j, :, :]
        assert img_s.shape == gt_s.shape

        # Resize and check the data are still what we expect
        resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)
        r_img: np.ndarray = resize_(img_s, shape).astype(np.uint8)
        r_gt: np.ndarray = resize_(gt_s, shape).astype(np.uint8)
        # print(time() - tic)
        assert r_img.dtype == r_gt.dtype == np.uint8
        assert 0 <= r_img.min() and r_img.max() <= 255  # The range might be smaller
        assert set(uniq(r_gt)).issubset(set(uniq(gt))), F"gt:{set(uniq(gt))}, resized:{set(uniq(r_gt))}"
        assert r_gt.max() == 1, r_gt.max()
        sizes_2d[j] = r_gt[r_gt == 1].sum()  # T: I guess 1 is the max ?

        # for save_dir, data in zip([save_dir_img, save_dir_gt], [r_img, r_gt]):
        #     save_dir.mkdir(parents=True, exist_ok=True)

        #     with warnings.catch_warnings():
        #         warnings.filterwarnings("ignore", category=UserWarning)
        #         imsave(str(Path(save_dir, filename)), data)

        for k in range(n_augment + 1):
            if k == 0:
                a_img, a_gt = r_img, r_gt
            else:
                a_img, a_gt = map_(np.asarray, augment(r_img, r_gt))

            for save_dir, data in zip([save_dir_img, save_dir_gt], [a_img, a_gt]):
                filename = f"{p_id}_{k}_{j:02d}.png"
                save_dir.mkdir(parents=True, exist_ok=True)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    imsave(str(Path(save_dir, filename)), data)

    return sizes_2d.sum(), sizes_2d[sizes_2d > 0].min(), sizes_2d.max()


def main(args: argparse.Namespace):
    print("ARG:", args, args.shape)
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    TRAIN_FOLDER = 'train_ancillary'
    VAL_FOLDER = 'val'
    TEST_FOLDER = 'test'
    # Assume the cleaning up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    # Get all the file names, avoid the temporal ones ???
    # T: Assert that there is an even number of files (each example has a label)
    files_paths: List[Path] = [p for p in src_path.rglob('*_0.png')]
    assert len(files_paths) % 2 == 0, "Uneven number of file, one+ pair is broken"

    # We sort now, but also id matching is checked while iterating later on
    #T: Split the paths between label (gt) and data (img)
    img_paths: List[Path] = sorted(p for p in files_paths if 'img/' in str(p))
    gt_paths: List[Path] = sorted(p for p in files_paths if 'inst/' in str(p))
    assert len(img_paths) == len(gt_paths)
    paths: List[Tuple[Path, Path]] = list(zip(img_paths, gt_paths))

    print(f"Found {len(img_paths)} pairs in total")

    # T: Split those paths between training and val set
    validation_paths: List[Tuple[Path, Path]] = [
        p for p in paths
        if str(p[0]).split('/')[-3] == VAL_FOLDER  # T: [-3] because the last three field are FOLDER/img/file
    ]
    training_paths: List[Tuple[Path, Path]] = [
        p for p in paths
        if str(p[0]).split('/')[-3] == TRAIN_FOLDER
    ]
    test_paths: List[Tuple[Path, Path]] = [
        p for p in paths
        if str(p[0]).split('/')[-3] == TEST_FOLDER
    ]
    assert set(validation_paths).isdisjoint(set(training_paths))
    assert len(validation_paths) > 0
    assert len(training_paths) > 0

    # assert len(paths) == (len(validation_paths) + len(training_paths))
    for mode, _paths, n_augment in zip(["train", "val"], [training_paths, validation_paths], [args.n_augment, 0]):
        img_paths, gt_paths = zip(*_paths)  # type: Tuple[Any, Any]

        dest_dir = Path(dest_path, mode)
        print(f"Slicing {len(img_paths)} pairs to {dest_dir}")
        assert len(img_paths) == len(gt_paths)

        pfun = partial(save_slices, dest_dir=dest_dir, shape=args.shape, n_augment=n_augment)
        sizes = mmap_(uc_(pfun), zip(img_paths, gt_paths))
        # sizes = []
        # for paths in tqdm(list(zip(img_paths, gt_paths)), ncols=50):
        #     sizes.append(uc_(pfun)(paths))
        sizes_3d, sizes_2d_min, sizes_2d_max = map_(np.asarray, zip(*sizes))

        print("2d sizes: ", sizes_2d_min.min(), sizes_2d_max.max())
        print("3d sizes: ", sizes_3d.min(), sizes_3d.mean(), sizes_3d.max())
        with open(os.path.join(dest_path, 'sizes.txt'), 'w') as fout:
            fout.write(F"2d sizes:\n\tmin:{sizes_2d_min.min()} \tmax:{sizes_2d_max.max()}\n")
            fout.write(F"3d sizes:\n\tmin:{sizes_3d.min()} \tmean:{sizes_3d.mean()} \tmax:{sizes_3d.max()}")
    # T: Making Test Set
    print(F"Generating test set in {dest_path}/test")
    resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)
    for img_path, gt_path in test_paths:
        print(img_path)
        dest_dir = Path(dest_path, 'test')
        print(img_path.name)
        filename = img_path.name
        for type, path in zip(['img', 'gt'], [img_path, gt_path]):
            save_dir = Path(dest_dir, type)
            save_dir.mkdir(parents=True, exist_ok=True)
            img = imread(path)
            if type == 'gt':
                dt = np.uint8
            else:
                dt = np.uint16
            img = resize_(img, args.shape).astype(dt)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(Path(save_dir, filename)), img)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--img_dir', type=str, default="IMG")
    parser.add_argument('--gt_dir', type=str, default="GT")
    parser.add_argument('--shape', type=int, nargs="+", default=(256, 256))
    parser.add_argument('--retain', type=int, default=10, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_augment', type=int, default=0, help="Number of augmentation to create per image, only for the training set")

    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)

    main(args)
