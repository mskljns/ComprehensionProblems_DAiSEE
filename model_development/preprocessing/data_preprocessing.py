"""
Feature Preprocessing:
"""

import os
import pandas as pd
import warnings
from model_development.preprocessing import utils as util
from time import time

def main():

    warnings.simplefilter(action='ignore', category=FutureWarning)
    cwd = os.path.dirname(os.path.abspath(__file__))
    head = os.path.abspath(os.path.join(cwd, "../.."))
    path = os.path.join(head, 'data\\interim\\')
    figures2save = os.path.join(cwd, 'figures\\data_distribution\\')
    if not os.path.isdir(figures2save):
        os.makedirs(figures2save)

    dataset = ['Test', 'Train', 'Validation']

    for ttv in dataset:
        print(ttv)
        path2save = os.path.join(path + 'data_preprocessed')
        if not os.path.isdir(path2save):
            os.mkdir(path2save)

        features = pd.read_csv(path + 'data_labelled\\' + ttv + 'set.csv')

        # Remove empty spaces in column names.
        features.columns = [col.replace(" ", "") for col in features.columns]

        data = features.copy()
        data_norm = features.copy()
        data_cap = features.copy()
        data_drop = features.copy()

        '------------------------------------------------'
        # Action Units bis 45 mean
        aus = ['AU01_c_mean', 'AU02_c_mean', 'AU04_c_mean', 'AU05_c_mean', 'AU06_c_mean', 'AU07_c_mean', 'AU09_c_mean',
               'AU10_c_mean', 'AU12_c_mean', 'AU14_c_mean', 'AU15_c_mean', 'AU17_c_mean', 'AU20_c_mean', 'AU23_c_mean',
               'AU25_c_mean', 'AU26_c_mean', 'AU28_c_mean', 'AU45_c_mean']

        # normalized data
        data_norm = util.normalize(data_norm, aus)
        util.plot_data_dist(data, data_norm, aus, figures2save, ttv)

        # capped data
        data_cap = util.normalize(data_cap, aus)

        # dropped data
        data_drop = util.normalize(data_drop, aus)
        '------------------------------------------------'

        # Gaze stuff
        gaze = ['gaze_angle_x_std', 'gaze_angle_y_std', 'gaze_angle_x_diff_std', 'gaze_angle_y_diff_std']

        # normalized data
        data_norm = util.log_transform(data_norm, gaze)
        data_norm = util.normalize(data_norm, gaze)
        util.plot_data_dist(data, data_norm, gaze, figures2save, ttv)

        # capped data
        data_cap = util.drop_outlier(data_cap, gaze)
        data_cap = util.log_transform(data_cap, gaze)
        data_cap = util.normalize(data_cap, gaze)

        # dropped data
        data_drop = util.drop_replace(data_drop, gaze)
        data_drop = util.log_transform(data_drop, gaze)
        data_drop = util.normalize(data_drop, gaze)

        '------------------------------------------------'

        blinks = ['gaze_fixation_count', 'blinks', 'blink_interval_mean', 'blink_len_mean']

        for fcol in blinks:

            if fcol == 'blinks':
                # normalized data
                data_norm = util.normalize(data_norm, fcol)
                util.plot_data_dist(data, data_norm, blinks, figures2save, ttv)

                # capped data
                data_cap = util.normalize(data_cap, fcol)

                # dropped data
                data_drop = util.normalize(data_drop, fcol)
                continue

            # normalized data
            data_norm = util.log_transform(data_norm, fcol)
            data_norm = util.normalize(data_norm, fcol)
            util.plot_data_dist(data, data_norm, blinks, figures2save, ttv)

            # capped data
            data_cap = util.drop_outlier(data_cap, fcol)
            data_cap = util.log_transform(data_cap, fcol)
            data_cap = util.normalize(data_cap, fcol)

            # dropped data
            data_drop = util.drop_replace(data_drop, fcol)
            data_drop = util.log_transform(data_drop, fcol)
            data_drop = util.normalize(data_drop, fcol)

        '------------------------------------------------'

        mouth = ['mouth_open_frame_len', 'AU25_mouth_open', 'AU25_mouth_open_interval_mean',
                 'AU25_mouth_open_frame_len_mean']

        for fcol in mouth:

            if fcol == 'AU25_mouth_open':
                # normalized data
                data_norm = util.normalize(data_norm, fcol)
                util.plot_data_dist(data, data_norm, mouth, figures2save, ttv)

                # capped data
                data_cap = util.normalize(data_cap, fcol)

                # dropped data
                data_drop = util.normalize(data_drop, fcol)

            else:
                # normalized data
                data_norm = util.log_transform(data_norm, fcol)
                data_norm = util.normalize(data_norm, fcol)
                util.plot_data_dist(data, data_norm, mouth, figures2save, ttv)

                # capped data
                data_cap = util.drop_outlier(data_cap, fcol)
                data_cap = util.log_transform(data_cap, fcol)
                data_cap = util.normalize(data_cap, fcol)

                # dropped data
                data_drop = util.drop_replace(data_drop, fcol)
                data_drop = util.log_transform(data_drop, fcol)
                data_drop = util.normalize(data_drop, fcol)

        '------------------------------------------------'

        head = ['pose_Tz_avgdist', 'pose_Tz_max_away', 'pose_Tz_min_away', 'left_rot_avg', 'left_rot_max', 'right_rot_mean',
                'right_rot_max', 'skew_left_avg', 'skew_left_max', 'skew_right_avg', 'skew_right_max']

        for fcol in head:

            if fcol in ['pose_Tz_avgdist']:
                # normalized data
                data_norm = util.normalize(data_norm, fcol)
                util.plot_data_dist(data, data_norm, fcol, figures2save, ttv)
                # capped data
                data_cap = util.normalize(data_cap, fcol)
                # dropped data
                data_drop = util.normalize(data_drop, fcol)
                continue

            elif fcol in ['right_rot_max', 'skew_right_max']:
                # normalized data
                data_norm = util.log_transform(data_norm, fcol)
                data_norm = util.normalize(data_norm, fcol)
                util.plot_data_dist(data, data_norm, fcol, figures2save, ttv)
                # capped data
                data_cap = util.drop_outlier(data_cap, fcol)
                data_cap = util.log_transform(data_cap, fcol)
                data_cap = util.normalize(data_cap, fcol)
                # dropped data
                data_drop = util.normalize(data_drop, fcol)
                continue

            elif fcol in ['pose_Tz_max_away', 'right_rot_mean']:
                # normalized data
                data_norm = util.log_transform(data_norm, fcol)
                data_norm = util.normalize(data_norm, fcol)
                util.plot_data_dist(data, data_norm, fcol, figures2save, ttv)
                # capped data
                data_cap = util.drop_outlier(data_cap, fcol)
                data_cap = util.log_transform(data_cap, fcol)
                data_cap = util.normalize(data_cap, fcol)
                # dropped data
                data_drop = util.drop_replace(data_drop, fcol)
                data_drop = util.normalize(data_drop, fcol)
                continue

            elif fcol in ['left_rot_max', 'skew_left_avg', 'skew_left_max']:
                # normalized data
                data_norm[fcol] = (data_norm[fcol] * (-1))
                data_norm = util.log_transform(data_norm, fcol)
                data_norm = util.normalize(data_norm, fcol)
                util.plot_data_dist(data, data_norm, fcol, figures2save, ttv)
                # capped data
                data_cap[fcol] = (data_cap[fcol] * (-1))
                data_cap = util.drop_outlier(data_cap, fcol)
                data_cap = util.log_transform(data_cap, fcol)
                data_cap = util.normalize(data_cap, fcol)
                # dropped data
                data_drop[fcol] = (data_drop[fcol] * (-1))
                data_drop = util.normalize(data_drop, fcol)
                continue

            elif fcol in ['pose_Tz_min_away', 'left_rot_avg']:
                # normalized data
                data_norm[fcol] = (data_norm[fcol] * (-1))
                data_norm = util.log_transform(data_norm, fcol)
                data_norm = util.normalize(data_norm, fcol)
                util.plot_data_dist(data, data_norm, fcol, figures2save, ttv)
                # capped data
                data_cap[fcol] = (data_cap[fcol] * (-1))
                data_cap = util.drop_outlier(data_cap, fcol)
                data_cap = util.log_transform(data_cap, fcol)
                data_cap = util.normalize(data_cap, fcol)
                # dropped data
                data_drop[fcol] = (data_drop[fcol] * (-1))
                data_drop = util.drop_replace(data_drop, fcol)
                data_drop = util.log_transform(data_drop, fcol)
                data_drop = util.normalize(data_drop, fcol)
                continue

            elif fcol in ['skew_right_avg']:
                # normalized data
                data_norm = util.log_transform(data_norm, fcol)
                data_norm = util.normalize(data_norm, fcol)
                util.plot_data_dist(data, data_norm, fcol, figures2save, ttv)
                # capped data
                data_cap = util.drop_outlier(data_cap, fcol)
                data_cap = util.log_transform(data_cap, fcol)
                data_cap = util.normalize(data_cap, fcol)
                # dropped data
                data_drop[fcol] = (data_drop[fcol] * (-1))
                data_drop = util.drop_replace(data_drop, fcol)
                data_drop = util.log_transform(data_drop, fcol)
                data_drop = util.normalize(data_drop, fcol)
                continue

        data_norm.to_csv(path2save + '\\' + ttv + 'set_normalized.csv', index=False)
        data_cap.to_csv(path2save + '\\' + ttv + 'set_capped.csv', index=False)
        data_drop.to_csv(path2save + '\\' + ttv + 'set_dropped.csv', index=False)
    print('done')


if __name__ == "__main__":
    main()