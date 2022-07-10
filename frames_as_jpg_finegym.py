#!/usr/bin/env python3

import os
import argparse
from typing import NamedTuple
from multiprocessing import Pool
import numpy as np
import cv2
cv2.setNumThreads(0)
from tqdm import tqdm

from util.io import load_json


LABEL_DIR = 'data/finegym'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_video_dir',
                        help='Path to source videos')
    parser.add_argument('-o', '--out_dir',
                        help='Path to write frames. Dry run if None.')
    parser.add_argument('--max_height', type=int, default=224,
                        help='Max height of the extracted frames')
    parser.add_argument('--parallelism', type=int, default=os.cpu_count() // 4)
    return parser.parse_args()


class Task(NamedTuple):
    video_name: str
    video_path: str
    out_path: str
    min_frame: int
    max_frame: int
    target_fps: float
    target_num_frames: int
    width: int
    height: int
    max_height: int


def extract_frames(task):
    vc = cv2.VideoCapture(task.video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    exp_num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Assert that the version of the videos is the same
    assert (w, h) == (task.width, task.height), \
        'Mismatch in original frame dimensions: {} -- got {}, expected {}'.format(task.video_name, (w, h), (task.width, task.height))

    if task.max_height < h:
        oh = task.max_height
        ow = int(w / h * task.max_height)
    else:
        oh, ow = h, w

    if np.isclose(fps, task.target_fps, atol=0.001):
        # FPS match
        stride = 1
    elif np.isclose(fps / 2, task.target_fps, atol=0.001):
        stride = 2
    else:
        raise Exception('Target FPS does not match source FPS with an acceptable stride: {}'.format(task.video_name))

    if task.out_path is not None:
        os.makedirs(task.out_path)

    vc.set(cv2.CAP_PROP_POS_FRAMES, task.min_frame)
    i = 0
    out_frame_num = 0
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        if i % stride == 0:
            if frame.shape[0] != oh:
                frame = cv2.resize(frame, (ow, oh))

            if task.out_path is not None:
                frame_path = os.path.join(
                    task.out_path, '{:06d}.jpg'.format(out_frame_num))
                cv2.imwrite(frame_path, frame)

            out_frame_num += 1

        i += 1
        if task.min_frame + i == task.max_frame:
            break
    vc.release()

    assert out_frame_num == task.target_num_frames, \
        'Expected {} frames, got {}: {}'.format(
            task.target_num_frames, out_frame_num, task.video_name)


def main(src_video_dir, out_dir, max_height, parallelism):
    is_dry_run = False
    if out_dir is None:
        print('No output directory given. Doing a dry run!')
        is_dry_run = True
    else:
        os.makedirs(out_dir)

    tasks = []
    for split in ['train', 'val', 'test']:
        split_file = os.path.join(LABEL_DIR, split + '.json')
        labels = load_json(split_file)
        for data in labels:
            video_name = data['video']
            yt_id = video_name.split('_E_')[0]

            video_path = os.path.join(src_video_dir, '{}.mp4'.format(yt_id))
            if not os.path.exists(video_path):
                video_path = os.path.join(src_video_dir, '{}.mkv'.format(yt_id))
            if not os.path.exists(video_path):
                video_path = os.path.join(
                    src_video_dir, '{}.webm'.format(yt_id))
            assert os.path.exists(video_path), yt_id

            video_out_path = None
            if out_dir is not None:
                video_out_path = os.path.join(out_dir, video_name)

            # Use the source information to cut videos
            src_info = data['_source_info']
            tasks.append(Task(
                video_name=video_name,
                video_path=video_path, out_path=video_out_path,
                min_frame=src_info['start_frame'] - src_info['pad'][0],
                max_frame=src_info['end_frame'] + src_info['pad'][1],
                target_fps=data['fps'], target_num_frames=data['num_frames'],
                width=data['width'], height=data['height'],
                max_height=max_height))

    with Pool(parallelism) as p:
        for _ in tqdm(
            p.imap_unordered(extract_frames, tasks),
            total=len(tasks), desc='Dry run' if is_dry_run else 'Extracting'
        ):
            pass
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))