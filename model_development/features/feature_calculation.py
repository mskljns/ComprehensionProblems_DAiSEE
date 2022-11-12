"""
Calculate all features based on extracted parameters from OpenFace via the helper functions".
For more information see project report.
"""

import os
import pandas as pd
from model_development.features import utils as helper

cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, "../.."))
data_path = os.path.join(head, 'data\\raw\\')
interim_path = os.path.join(head, 'data\\interim\\')
if not os.path.isdir(interim_path):
    os.mkdir(interim_path)

dataset = ['Test', 'Train', 'Validation']

feature_names = ['clipID_avi', 'clipID', 'AU01_c_mean', 'AU02_c_mean', 'AU04_c_mean', 'AU05_c_mean',
                 'AU06_c_mean', 'AU07_c_mean', 'AU09_c_mean', 'AU10_c_mean',
                 'AU12_c_mean', 'AU14_c_mean', 'AU15_c_mean', 'AU17_c_mean',
                 'AU20_c_mean', 'AU23_c_mean', 'AU25_c_mean', 'AU26_c_mean',
                 'AU28_c_mean', 'AU45_c_mean', 'AU01_c_sum', 'AU02_c_sum', 'AU04_c_sum',
                 'AU05_c_sum', 'AU06_c_sum', 'AU07_c_sum', 'AU09_c_sum', 'AU10_c_sum',
                 'AU12_c_sum', 'AU14_c_sum', 'AU15_c_sum', 'AU17_c_sum', 'AU20_c_sum',
                 'AU23_c_sum', 'AU25_c_sum', 'AU26_c_sum', 'AU28_c_sum', 'AU45_c_sum',
                 'gaze_angle_x_std', 'gaze_angle_y_std', 'gaze_angle_x_diff_std',
                 'gaze_angle_y_diff_std', 'gaze_fixation_count', 'blinks', 'blink_interval_mean',
                 'blink_len_mean', 'mouth_open_frame_len', 'AU25_mouth_open', 'AU25_mouth_open_interval_mean',
                 'AU25_mouth_open_frame_len_mean', 'pose_Tz_avgdist',
                 'pose_Tz_max_away', 'pose_Tz_min_away', 'left_rot_avg', 'left_rot_max',
                 'right_rot_mean', 'right_rot_max', 'skew_left_avg', 'skew_left_max',
                 'skew_right_avg', 'skew_right_max']

for ttv in dataset:
    path_calc_feat = interim_path + 'calculated_features\\'
    if not os.path.isdir(path_calc_feat):
        os.mkdir(path_calc_feat)
    path_2save = interim_path + 'calculated_features\\' + ttv + '\\'
    if not os.path.isdir(path_2save):
        os.mkdir(path_2save)

    features = pd.DataFrame(columns=feature_names)
    users = os.listdir(data_path + ttv + '/')
    users = users[0:-1]

    for user in users:

        user_clips = os.listdir(data_path + ttv + '/' + user + '/')  # also log file
        clips_path = data_path + ttv + '/' + user + '/'
        # list all scv files only
        files = os.listdir(clips_path)
        files = list(filter(lambda f: f.endswith('.csv'), files))

        feature_extraction_user = pd.DataFrame(columns=feature_names)

        for file in files:
            df = pd.read_csv(clips_path + file)

            # Remove empty spaces in column names. Important when calling the names!
            df.columns = [col.replace(" ", "") for col in df.columns]

            # Print few values of data.
            # print(f"Max number of frames {df.frame.max()}", f"\nTotal shape of dataframe {df.shape}")
            # print(df.head())

            clipID_avi = pd.DataFrame([file], columns=['clipID_avi'])
            clipID = pd.DataFrame([file[:len(file) - 4]], columns=['clipID'])
            action_units = helper.calc_action_units(df)
            gaze = helper.calc_gaze_angle(df)
            blink_stuff = helper.get_blinks(df)  # exclude since too many outliers
            mouth_open_frame_len = helper.get_frames_mouthopen(df)
            mouth_stuff = helper.get_mouthopen(df)
            head_distance = helper.calc_head_distance(df)
            head_pose = helper.calc_head_pose(df)

            # create Dataframe for all features
            feature_frame = pd.concat([clipID_avi, clipID, action_units, gaze, blink_stuff, mouth_open_frame_len,
                                       mouth_stuff, head_distance, head_pose], sort=False, axis=1)
            feature_extraction_user = feature_extraction_user.append(feature_frame)
            feature_extraction_user = feature_extraction_user.sort_values(by=['clipID'])

        features = features.append(feature_extraction_user)

    # data will be saved as single CSV file for Test/Train/Validation Set
    features.to_csv(path_2save + '\\' + ttv + 'set.csv', index=False)
    print(path_2save + '\\' + ttv + 'set.csv')
