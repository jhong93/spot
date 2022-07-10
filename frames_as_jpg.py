#!/usr/bin/env python3

import os
import argparse
from typing import NamedTuple
import numpy as np
import cv2
cv2.setNumThreads(0)
from tqdm import tqdm
from multiprocessing import Pool

from util.io import load_json


FS_LABEL_DIR = 'data/fs_comp'
TENNIS_LABEL_DIR = 'data/tennis'


class Task(NamedTuple):
    video_name: str
    video_path: str
    out_path: str
    min_frame: int
    max_frame: int
    target_fps: float
    max_height: int


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['fs', 'tennis'],
                        help='Dataset to extract frames for.')
    parser.add_argument('video_dir', help='Path to the videos')
    parser.add_argument('-o', '--out_dir',
                        help='Path to write frames. Dry run if None.')
    parser.add_argument('--max_height', type=int, default=224,
                        help='Max height of the extracted frames')
    parser.add_argument('--parallelism', type=int, default=os.cpu_count() // 4)
    return parser.parse_args()


def get_fs_tasks(video_dir, out_dir, max_height):
    tasks = []

    for split in ['train', 'val', 'test']:
        split_file = os.path.join(FS_LABEL_DIR, split + '.json')
        labels = load_json(split_file)
        for data in labels:
            video_name = data['video']
            base_video_name, _, start_frame, end_frame = video_name.rsplit(
                '_', 3)
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            assert end_frame - start_frame == data['num_frames']

            video_out_path = None
            if out_dir is not None:
                video_out_path = os.path.join(out_dir, video_name)

            video_path = os.path.join(video_dir, base_video_name + '.mkv')
            tasks.append(Task(
                video_name=video_name, video_path=video_path,
                out_path=video_out_path,
                min_frame=start_frame, max_frame=end_frame,
                target_fps=data['fps'], max_height=max_height
            ))
    return tasks


def get_tennis_tasks(video_dir, out_dir, max_height):
    video_files = os.listdir(video_dir)

    def match_video_file(prefix):
        for v in video_files:
            if v.startswith(prefix):
                return v
        else:
            raise Exception('Not found: {}'.format(prefix))

    tasks = []
    for split in ['train', 'val', 'test']:
        split_file = os.path.join(TENNIS_LABEL_DIR, split + '.json')
        labels = load_json(split_file)
        for data in labels:
            video_name = data['video']
            base_video_name, start_frame, end_frame = video_name.rsplit('_', 2)
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            assert end_frame - start_frame == data['num_frames']

            video_out_path = None
            if out_dir is not None:
                video_out_path = os.path.join(out_dir, video_name)

            video_path = os.path.join(
                video_dir, match_video_file(base_video_name))
            tasks.append(Task(
                video_name=video_name, video_path=video_path,
                out_path=video_out_path,
                min_frame=start_frame, max_frame=end_frame,
                target_fps=data['fps'], max_height=max_height
            ))
    return tasks


def extract_frames(task):
    vc = cv2.VideoCapture(task.video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    exp_num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if task.max_height < h:
        oh = task.max_height
        ow = int(w / h * task.max_height)
    else:
        oh, ow = h, w

    assert np.isclose(fps, task.target_fps), (fps, task.target_fps)

    if task.out_path is not None:
        os.makedirs(task.out_path)

    vc.set(cv2.CAP_PROP_POS_FRAMES, task.min_frame)
    i = 0
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        if frame.shape[0] != oh:
            frame = cv2.resize(frame, (ow, oh))

        if task.out_path is not None:
            frame_path = os.path.join(task.out_path, '{:06d}.jpg'.format(i))
            cv2.imwrite(frame_path, frame)

        i += 1
        if task.min_frame + i == task.max_frame:
            break

    vc.release()
    assert i == task.max_frame - task.min_frame, \
        'Expected {} frames, got {}: {}'.format(
            task.max_frame - task.min_frame, i, task.video_name)


def main(dataset, video_dir, out_dir, max_height, parallelism):
    if dataset == 'fs':
        tasks = get_fs_tasks(video_dir, out_dir, max_height)
    elif dataset == 'tennis':
        tasks = get_tennis_tasks(video_dir, out_dir, max_height)
    else:
        raise Exception('Unknown dataset: ' + dataset)

    is_dry_run = False
    if out_dir is None:
        print('No output directory given. Doing a dry run!')
        is_dry_run = True
    else:
        os.makedirs(out_dir)

    with Pool(parallelism) as p:
        for _ in tqdm(
            p.imap_unordered(extract_frames, tasks),
            total=len(tasks), desc='Dry run' if is_dry_run else 'Extracting'
        ):
            pass
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))