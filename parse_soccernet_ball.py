#!/usr/bin/env python3

import os
import argparse
import json
from collections import defaultdict

from SoccerNet.utils import getListGames

from util.io import load_json
from util.dataset import read_fps, get_num_frames


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_dir', type=str,
                        help='Path to the SoccerNet-ball labels')
    parser.add_argument('frame_dir', type=str,
                        help='Path to extracted video frames')
    parser.add_argument('-o', '--out_dir', type=str,
                        help='Path to output parsed dataset')
    return parser.parse_args()


def load_split(split):
    if split == 'val':
        split = 'valid'

    videos = []
    for entry in getListGames(split, task="spotting-ball"):
        league, season, game = entry.split('/')
        videos.append((league, season, game))
    return videos


def get_label_names(labels):
    return {e['label'] for v in labels for e in v['events']}


def main(label_dir, frame_dir, out_dir):
    labels_by_split = defaultdict(list)
    for split in ['train', 'val', 'test', 'challenge']:
        videos = load_split(split)
        for video in videos:
            league, season, game = video

            video_label_path = os.path.join(
                label_dir, league, season, game, 'Labels-ball.json')

            if split != 'challenge':
                video_labels = load_json(video_label_path)
            else:
                video_labels = {'annotations': []}

            num_events = 0
            for half in (1, 2):
                video_frame_dir = os.path.join(
                    frame_dir, league, season, game, str(half))

                sample_fps = read_fps(video_frame_dir)
                num_frames = get_num_frames(video_frame_dir)

                video_id = '{}/{}/{}/{}'.format(league, season, game, half)

                half_events = []
                for label in video_labels['annotations']:
                    lhalf = int(label['gameTime'].split(' - ')[0])
                    if half == lhalf:
                        adj_frame = float(label['position']) / 1000 * sample_fps
                        half_events.append({
                            'frame': int(adj_frame),
                            'label': label['label'],
                            'comment': '{}; {}'.format(
                                label['team'], label['visibility'])
                        })

                        if adj_frame >= num_frames:
                            print('Label past end: {} -- {} < {} -- {}'.format(
                                video_id, num_frames, int(adj_frame),
                                label['label']))
                num_events += len(half_events)
                half_events.sort(key=lambda x: x['frame'])

                # max_label_frame = max(e['frame'] for e in half_events) \
                #     if len(half_events) > 0 else 0
                # if max_label_frame >= num_frames:
                #     num_frames = max_label_frame + 1

                labels_by_split[split].append({
                    'video': video_id,
                    'num_frames': num_frames,
                    'num_events': len(half_events),
                    'events': half_events,
                    'fps': sample_fps,
                    'width': 398,
                    'height': 224
                })
            assert len(video_labels['annotations']) == num_events, \
                video_label_path

    train_classes = get_label_names(labels_by_split['train'])
    assert train_classes == get_label_names(labels_by_split['test'])
    assert train_classes == get_label_names(labels_by_split['val'])

    print('Classes:', sorted(train_classes))

    for split, labels in labels_by_split.items():
        print('{} : {} videos : {} events'.format(
            split, len(labels), sum(len(l['events']) for l in labels)))
        labels.sort(key=lambda x: x['video'])

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        class_path = os.path.join(out_dir, 'class.txt')
        with open(class_path, 'w') as fp:
            fp.write('\n'.join(sorted(train_classes)))

        for split, labels in labels_by_split.items():
            out_path = os.path.join(out_dir, '{}.json'.format(split))
            with open(out_path, 'w') as fp:
                json.dump(labels, fp, indent=2, sort_keys=True)

    print('Done!')


if __name__ == '__main__':
    main(**vars(get_args()))