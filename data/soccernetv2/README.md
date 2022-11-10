# Setting up SoccerNetV2

Set up SoccerNet for the spotting challenge. For more information about the task, refer to https://www.soccer-net.org/.

1. Follow the instructions on the SoccerNet page to obtain the videos and the labels for the action spotting task. Either the 224P or the 720P videos work, though the latter may have fewer encoding errors.

2. Install SoccerNet dependencies: `pip3 install SoccerNet`. Other packages such as `moviepy` may also be required.

3. Extract frames at 2 FPS using `python3 frames_as_jpg_soccernet.py <video_dir> -o <out_dir>`. This will take a while.

4. Parse the labels with `python3 parse_soccernet.py <label_dir> <frame_dir> -o data/soccernetv2`.

## Notes

Refer to the SoccerNet page for NDA and data license.

For the challenge (non-validated) configuration, you will need to combine `train.json`, `val.json`, and `test.json` into a new `train.json`.

We provide a convenience script `eval_soccernetv2.py`, which calls the SoccerNet package for evaluation.