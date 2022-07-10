# Setting up FineDiving

This directory contains the splits and labels converted from FineGym: https://sdolivia.github.io/FineGym/.

1. Download the videos from the FineGym authors (Shao, et al., CVPR 2020).

We obtain the videos directly from the FineGym authors, but obtaining videos from YouTube may also work with additional effort (not implemented!). There will be mismatch in frame rates with the pre-parsed labels.

2. Run `python3 frames_as_jpg_finegym.py <src_video_dir> -o <out_dir>` to generate the folder structure for frames.

## Notes
* We introduce a train / val / test split since the original split is for action recognition (classification).
* We pad a random number of frames before and after each "Event" label (in FineGym terminology; i.e., an untrimmed performance) when generating these splits originally. This is to break correlations between labels being on the first or last frames.

---

## License from FineGym

Creative Commons Attribution-NonCommercial 4.0 International License\
(See https://sdolivia.github.io/FineGym/)