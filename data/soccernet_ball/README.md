# Setting up SoccerNet Ball Action Spotting

Set up SoccerNet for the spotting challenge. For more information about the task, refer to https://www.soccer-net.org/.

1. Follow the instructions on the SoccerNet page to obtain the videos and the labels for the action spotting task. Either the 224P or the 720P videos work.

2. Install SoccerNet dependencies: `pip3 install SoccerNet`. Other packages such as `moviepy` may also be required.

3. Extract frames at 25 FPS using `python3 frames_as_jpg_soccernet_ball.py <video_dir> -o <out_dir> --sample_fps 25`. This will take a while.

4. Parse the labels with `python3 parse_soccernet_ball.py <label_dir> <frame_dir> -o data/soccernet_ball`.

## Notes

Refer to the SoccerNet page for NDA and data license.

We provide a convenience script `eval_soccernet_ball.py`, which calls the SoccerNet package for evaluation.

`python eval_soccernet_ball.py <out_dir> -l <video_dir> --eval_dir <out_dir_pred> -s test --nms_window=25`.
`python eval_soccernet_ball.py <out_dir> -l <video_dir> --eval_dir <out_dir_pred> -s challenge --nms_window=25`.

To submit to the evaluation server, simply zip all files inside `<out_dir_pred>`