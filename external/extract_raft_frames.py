#!/usr/bin/env python3
"""
Optical flow extraction
=======================

Copy and run from the RAFT repository root.
URL: https://github.com/princeton-vl/RAFT
"""

import sys
sys.path.append('core')

import argparse
import os
import re
from multiprocessing.pool import ThreadPool
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm import tqdm

from raft import RAFT
from utils.utils import InputPadder


DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def get_paths(frame_dir):
    base_img_re = re.compile(r'^\d+.jpg$')

    results = []
    def helper(d):
        for f in os.listdir(d):
            if base_img_re.match(f):
                results.append(os.path.join(d, f))
            else:
                fp = os.path.join(d, f)
                if os.path.isdir(fp):
                    helper(fp)

    helper(frame_dir)
    return results


class FrameDataset(Dataset):

    def __init__(self, frame_dir, overwrite, out_dir):
        paths = []
        for im_path in get_paths(frame_dir):
            im_relpath = im_path.split(frame_dir)[1]
            if im_relpath.startswith('/'):
                im_relpath = im_relpath[1:]
            flow_outpath = os.path.join(
                out_dir, os.path.splitext(im_relpath)[0] + '.jpg')
            if not overwrite and os.path.exists(flow_outpath):
                continue

            def get_im_path(x):
                return os.path.join(
                    os.path.dirname(im_path),
                    '{:06d}.jpg'.format(x))

            prev_frame_num = \
                int(os.path.basename(os.path.splitext(im_path)[0])) - 1
            if prev_frame_num < 0:
                prev_frame_num = 0
            prev_im_path = get_im_path(prev_frame_num)
            if os.path.exists(prev_im_path):
                paths.append((prev_im_path, im_path, flow_outpath))
            else:
                print('Not found:', prev_im_path)
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        prev_im_path, im_path, out_path = self.paths[idx]
        image1 = load_image(prev_im_path)
        image2 = load_image(im_path)
        return out_path, image1, image2


def to_img(flow, clip):
    flow = np.clip(flow, -clip, clip) + clip
    flow *= 255 / (2 * clip + 1)
    h, w, _ = flow.shape
    return np.dstack((flow.astype(np.uint8), np.full((h, w, 1), 128, np.uint8)))


def output_batch(out_paths, flow, clip, subtract_median):
    try:
        for i in range(len(out_paths)):
            fi = flow[i]
            if subtract_median:
                mf = np.median(fi, axis=(0, 1))
                fi -= mf
            os.makedirs(os.path.dirname(out_paths[i]), exist_ok=True)
            cv2.imwrite(out_paths[i], to_img(fi, clip),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    except Exception as e:
        print(e)
        raise


def demo(args):
    with ThreadPool() as io_workers:
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(DEVICE)
        model.eval()

        datasets = []
        for video_name in sorted(os.listdir(args.path), reverse=args.reverse):
            dataset = FrameDataset(
                os.path.join(args.path, video_name),
                args.overwrite, os.path.join(args.out_dir, video_name))
            if len(dataset) > 0:
                datasets.append(dataset)

        n = sum(len(d) for d in datasets)
        futures = []
        with tqdm(desc='Processing', total=n) as pbar, torch.no_grad():
            for dataset in datasets:
                loader = DataLoader(
                    dataset, num_workers=os.cpu_count() // 2,
                    batch_size=args.batch_size)
                for flow_out_paths, image1s, image2s in loader:
                    padder = InputPadder(image1s.shape)
                    image1s, image2s = padder.pad(
                        image1s.cuda(), image2s.cuda())

                    flow_low, flow_up = model(
                        image1s, image2s, iters=20, test_mode=True)

                    flow_up = padder.unpad(flow_up)
                    flow_np = flow_up.permute(0, 2, 3, 1).cpu().numpy()
                    futures.append(io_workers.apply_async(
                        output_batch,
                        (flow_out_paths, flow_np, args.clip,
                         args.subtract_median)))
                    pbar.update(flow_np.shape[0])

        for fut in tqdm(futures, desc='Writing'):
            fut.get()
        print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="dataset for evaluation")
    parser.add_argument('--model', help="restore checkpoint",
                        default='models/raft-sintel.pth')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', type=bool, default=True,
                        help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficent correlation implementation')

    parser.add_argument('--clip', type=int, default=20)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--subtract_median', type=bool, default=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()

    demo(args)
