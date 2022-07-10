import os

from util.io import load_text


DATASETS = [
    'tennis',
    'fs_perf',
    'fs_comp',
    'finediving',
    'finegym',
    'soccernetv2',
]


def load_classes(file_name):
    return {x: i + 1 for i, x in enumerate(load_text(file_name))}


def read_fps(video_frame_dir):
    with open(os.path.join(video_frame_dir, 'fps.txt')) as fp:
        return float(fp.read())


def get_num_frames(video_frame_dir):
    max_frame = -1
    for img_file in os.listdir(video_frame_dir):
        if img_file.endswith('.jpg'):
            frame = int(os.path.splitext(img_file)[0])
            max_frame = max(frame, max_frame)
    return max_frame + 1


FINEGYM_START_SET = {
    # 'BB_dismounts_end',
    'BB_dismounts_start',
    # 'BB_flight_handspring_end',
    'BB_flight_handspring_start',
    # 'BB_flight_salto_end',
    'BB_flight_salto_start',
    # 'BB_leap_jump_hop_end',
    'BB_leap_jump_hop_start',
    # 'BB_turns_end',
    'BB_turns_start',
    # 'FX_back_salto_end',
    'FX_back_salto_start',
    # 'FX_front_salto_end',
    'FX_front_salto_start',
    # 'FX_leap_jump_hop_end',
    'FX_leap_jump_hop_start',
    # 'FX_side_salto_end',
    'FX_side_salto_start',
    # 'FX_turns_end',
    'FX_turns_start',
    # 'UB_circles_end',
    'UB_circles_start',
    # 'UB_dismounts_end',
    'UB_dismounts_start',
    # 'UB_fligh_same_bar_end',
    'UB_fligh_same_bar_start',
    # 'UB_transition_flight_end',
    'UB_transition_flight_start',
    # 'VT_0',
    'VT_1',
    'VT_2',
    # 'VT_3
}