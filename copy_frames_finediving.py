#!/usr/bin/env python3
""" Copy FineDiving frames into our directory structure """

import os
import argparse
import shutil
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_frame_dir', help='Path to the original frames')
    parser.add_argument('out_frame_dir', help='Path to output dir')

    parser.add_argument('--resize_height', type=int,
                        help='Resize the frames instead of copying')
    return parser.parse_args()


def collect_frames(frame_dir, out_dir, resize_height):
    frame_files = sorted(os.listdir(frame_dir))
    assert all(x.endswith('.jpg') for x in frame_files)

    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, frame_file)
        os.makedirs(out_dir, exist_ok=True)
        frame_out_path = os.path.join(out_dir, '{:06d}.jpg'.format(i))

        if resize_height is None:
            shutil.copyfile(frame_path, frame_out_path)
        else:
            im = cv2.imread(frame_path)
            h, w, _ = im.shape
            im = cv2.resize(im, (int(resize_height * w / h), resize_height))
            cv2.imwrite(frame_out_path, im)
    return len(frame_files)


def main(src_frame_dir, out_frame_dir, resize_height):
    if resize_height is not None:
        print('Warning: resizing frames to {} px height!'.format(resize_height))

    for a in os.listdir(src_frame_dir):
        for b in os.listdir(os.path.join(src_frame_dir, a)):
            seq_in_dir = os.path.join(src_frame_dir, a, b)
            seq_out_dir = os.path.join(out_frame_dir, '{}__{}'.format(a, b))
            n = collect_frames(seq_in_dir, seq_out_dir, resize_height)
            print('Copied {} frames: {} -- {}'.format(
                n, seq_in_dir, seq_out_dir))
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))