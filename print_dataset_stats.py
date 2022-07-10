#!/usr/bin/env python3

import os
from collections import Counter

from util.dataset import DATASETS
from util.io import load_json, load_text


def get_src_video(dataset, video_name):
    if dataset.startswith('fs'):
        video_name = video_name.rsplit('_', 3)[0]
    elif dataset.startswith('tennis'):
        video_name = video_name.rsplit('_', 2)[0]
    elif dataset.startswith('finegym'):
        video_name = video_name.split('_E_', 1)[0]
    return video_name


def print_dataset_stats(dataset):
    print('=== {} ==='.format(dataset))
    class_file = os.path.join('data', dataset, 'class.txt')
    if not os.path.isfile(class_file):
        print('Dataset not found!')
        return
    print('Categories:', len(load_text(class_file)))

    src_videos = {}

    total_frames = 0
    total_events = 0

    all_videos = set()
    for split in ['train', 'val', 'test']:
        split_file = os.path.join('data', dataset, '{}.json'.format(split))
        if os.path.exists(split_file):
            print('{}:'.format(split.capitalize()))
            labels = load_json(split_file)
            num_events = sum([len(x['events']) for x in labels])
            num_frames = sum([x['num_frames'] for x in labels])
            src_videos_split = {get_src_video(dataset, x['video'])
                                for x in labels}
            print('\torig videos:', len(src_videos_split))
            # if len(src_videos_split) <= 20:
            #     for v in sorted(src_videos_split):
            #         print('\t\t', v)
            print('\tvideos:', len(labels))
            print('\tevents:', num_events)
            print('\tframes:', num_frames)
            print('\tevents / frames (%):', round(
                num_events / num_frames * 100, 2))

            total_frames += num_frames
            total_events += num_events

            first_event = min(
                [min(e['frame'] for e in x['events']) for x in labels])
            last_event = min(
                [min(x['num_frames'] - e['frame'] for e in x['events'])
                 for x in labels])
            print('\tmin frame (of first event):', first_event)
            print('\tmax frame (of last event):', last_event)

            split_videos = {x['video'] for x in labels}
            assert len(split_videos & all_videos) == 0, \
                'Bad video splits!'
            all_videos.update(split_videos)

            src_videos[split] = src_videos_split

            label_counts = Counter()
            for x in labels:
                for e in x['events']:
                    label_counts[e['label']] += 1
            print('\tLabel counts:')
            for l in sorted(label_counts.keys()):
                print('\t\t{} : {}'.format(l, label_counts[l]))

    print('Overall:')
    print('\thas train/test orig video overlap:',
          len(src_videos['train'] &
          src_videos.get('test', src_videos['val'])) > 0)
    print('\tnum frames:', total_frames)
    print('\tnum events:', total_events)
    print('\tevent %:', total_events  * 100 / total_frames)


def main():
    for i, dataset in enumerate(DATASETS):
        print_dataset_stats(dataset)
        if i < len(DATASETS) - 1:
            print()


if __name__ == '__main__':
    main()