#!/usr/bin/env python3

import os
import argparse

from tabulate import tabulate

from util.video import get_metadata


VIDEO_EXTS = ('.mp4', '.mkv')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir')
    return parser.parse_args()


def get_paths(video_dir):
    paths = []

    def _recurse(rel_path):
        abs_path = os.path.join(video_dir, rel_path)
        for ent in os.listdir(abs_path):
            ent_rel_path = os.path.join(rel_path, ent)
            ent_abs_path = os.path.join(video_dir, ent_rel_path)
            if os.path.isdir(ent_abs_path):
                _recurse(ent_rel_path)
            elif os.path.isfile(ent_abs_path) and ent.endswith(VIDEO_EXTS):
                paths.append(ent_rel_path)

    _recurse('')
    return sorted(paths)


def main(video_dir):
    total_seconds = 0
    total_frames = 0
    videos = []

    for video_rel_path in get_paths(video_dir):
        meta = get_metadata(os.path.join(video_dir, video_rel_path))
        videos.append((video_rel_path, meta.fps, meta.num_frames,
                        '{}x{}'.format(meta.width, meta.height)))

        total_seconds += meta.num_frames / meta.fps
        total_frames += meta.num_frames
    print(tabulate(videos, headers=['name', 'fps', '# frames', 'resolution']))

    print('\nTotal: {:d} s  {:d} frames'.format(
        int(total_seconds), total_frames))
    print('Est size as jpgs: {:d} MB'.format(round(0.018 * total_frames)))


if __name__ == '__main__':
    main(**vars(get_args()))