#!/usr/bin/env python3

import os
import argparse
import cv2
import moviepy.editor
from tqdm import tqdm
from multiprocessing import Pool
cv2.setNumThreads(0)

from util.dataset import read_fps


RECALC_FPS_ONLY = False

FRAME_RETRY_THRESHOLD = 1000

TARGET_HEIGHT = 224
TARGET_WIDTH = 398


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir', help='Path to the downloaded videos')
    parser.add_argument('-o', '--out_dir',
                        help='Path to write frames. Dry run if None.')
    parser.add_argument('--sample_fps', type=int, default=2)
    parser.add_argument('--recalc_fps', action='store_true')
    parser.add_argument('-j', '--num_workers', type=int,
                        default=os.cpu_count() // 4)
    return parser.parse_args()


def get_duration(video_path):
    # Copied from SoccerNet repo
    return moviepy.editor.VideoFileClip(video_path).duration


def worker(args):
    video_name, video_path, out_dir, sample_fps = args

    def get_stride(src_fps):
        if sample_fps <= 0:
            stride = 1
        else:
            stride = int(src_fps / sample_fps)
        return stride

    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    oh = TARGET_HEIGHT
    ow = TARGET_WIDTH

    time_in_s = get_duration(video_path)

    fps_path = None
    if out_dir is not None:
        fps_path = os.path.join(out_dir, 'fps.txt')
        if not RECALC_FPS_ONLY:
            if os.path.exists(fps_path):
                print('Already done:', video_name)
                vc.release()
                return
        else:
            if str(read_fps(out_dir)) == str(fps / get_stride(fps)):
                print('FPS is already consistent:', video_name)
                vc.release()
                return
            else:
                # Recalculate FPS in cases where the actual frame count does not
                # match the metadata
                print('Inconsistent FPS:', video_name)

        os.makedirs(out_dir, exist_ok=True)

    not_done = True
    while not_done:
        stride = get_stride(fps)
        est_out_fps = fps / stride
        print('{} -- effective fps: {} (stride: {})'.format(
            video_name, est_out_fps, stride))

        out_frame_num = 0
        i = 0
        while True:
            ret, frame = vc.read()
            if not ret:
                # fps and num_frames are wrong
                if i != num_frames:
                    print('Failed to decode: {} -- {} / {}'.format(
                        video_path, i, num_frames))

                    if i + FRAME_RETRY_THRESHOLD < num_frames:
                        num_frames = i
                        adj_fps = num_frames / time_in_s
                        if get_stride(adj_fps) == stride:
                            # Stride would not change so nothing to do
                            not_done = False
                        else:
                            print('Retrying:', video_path)
                            # Stride changes, due to large error in fps.
                            # Use adjusted fps instead.
                            vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            fps = adj_fps
                    else:
                        not_done = False
                else:
                    not_done = False
                break

            if i % stride == 0:
                if not RECALC_FPS_ONLY:
                    if frame.shape[0] != oh or frame.shape[1] != ow:
                        frame = cv2.resize(frame, (ow, oh))
                    if out_dir is not None:
                        frame_path = os.path.join(
                            out_dir, '{:06d}.jpg'.format(out_frame_num))
                        cv2.imwrite(frame_path, frame)
                out_frame_num += 1
            i += 1
    vc.release()

    out_fps = fps / get_stride(fps)
    if fps_path is not None:
        with open(fps_path, 'w') as fp:
            fp.write(str(out_fps))
    print('{} - done'.format(video_name))


def main(video_dir, out_dir, num_workers,
         sample_fps=None, recalc_fps=False):
    global RECALC_FPS_ONLY
    RECALC_FPS_ONLY = recalc_fps

    worker_args = []
    for league in os.listdir(video_dir):
        league_dir = os.path.join(video_dir, league)
        for season in os.listdir(league_dir):
            season_dir = os.path.join(league_dir, season)
            for game in os.listdir(season_dir):
                game_dir = os.path.join(season_dir, game)
                for video_file in os.listdir(game_dir):
                    if video_file.endswith('720p.mkv'):
                        video_name = os.path.splitext(video_file)[0].replace(
                            '_720p', '')
                        worker_args.append((
                            os.path.join(league, season, game, video_file),
                            os.path.join(game_dir, video_file),
                            os.path.join(
                                out_dir, league, season, game, video_name
                            ) if out_dir else None,
                            sample_fps
                        ))

    with Pool(num_workers) as p:
        for _ in tqdm(p.imap_unordered(worker, worker_args),
                      total=len(worker_args)):
            pass
    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))
