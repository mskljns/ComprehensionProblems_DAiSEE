"""
Helper functions to calculate the features.
"""
import pandas as pd
import numpy as np
import math


# ACTION UNITS
# calculate average AU presence in 10 sec video for feature selection
# AU [696:714]
def calc_action_units(dataframe):
    # action_units_name = df.columns[696:714]
    action_units_name = ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c',
                         'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c',
                         'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

    action_units_orig = dataframe[action_units_name]
    action_units_mean = pd.DataFrame(action_units_orig.mean(axis=0)).transpose()
    action_units_mean = action_units_mean.add_suffix('_mean')

    action_units_sum = pd.DataFrame(action_units_orig.sum(axis=0)).transpose()
    action_units_sum = action_units_sum.add_suffix('_sum')

    action_units = pd.concat([action_units_mean, action_units_sum], axis=1)

    return action_units


'----------------------------------------------------------------------------------------------------------------------'


# GAZE ANGLE in radians for left-right and up-down movements
def calc_gaze_angle(dataframe):
    gaze_name = ['gaze_angle_x', 'gaze_angle_y']
    gaze_angle_xy = dataframe[gaze_name]

    gaze_angle_x_diff = [list(gaze_angle_xy.iloc[:, 0])[n] - list(gaze_angle_xy.iloc[:, 0])[n - 1] for n in
                         range(1, len(list(gaze_angle_xy.iloc[:, 0])))]
    gaze_angle_y_diff = [list(gaze_angle_xy.iloc[:, 1])[n] - list(gaze_angle_xy.iloc[:, 1])[n - 1] for n in
                         range(1, len(list(gaze_angle_xy.iloc[:, 1])))]

    gaze_angle_x_diff_std = pd.DataFrame([np.std(gaze_angle_x_diff)], columns=['gaze_angle_x_diff_std'])
    gaze_angle_y_diff_std = pd.DataFrame([np.std(gaze_angle_y_diff)], columns=['gaze_angle_y_diff_std'])

    gaze_fixation_count_x = gaze_angle_x_diff.count(0)
    gaze_fixation_count_y = gaze_angle_y_diff.count(0)
    gaze_fixation_count = pd.DataFrame([(gaze_fixation_count_x + gaze_fixation_count_y) / 2],
                                       columns=['gaze_fixation_count'])

    # std since the more often the eyes were moved away from the camera the higher the std should be
    gaze_angle_xy_std = pd.DataFrame(gaze_angle_xy.std(axis=0)).transpose()
    gaze_angle_xy_std = gaze_angle_xy_std.add_suffix('_std')

    gaze = pd.concat([gaze_angle_xy_std, gaze_angle_x_diff_std, gaze_angle_y_diff_std, gaze_fixation_count], axis=1)

    return gaze


'----------------------------------------------------------------------------------------------------------------------'


# HEAD DISTANCE with respect to the camera in mm
def calc_head_distance(dataframe):
    # distance between camera and face (+z away from camera)
    head_distance_z = dataframe[['pose_Tz']]
    head_distance_z_avg = pd.DataFrame(head_distance_z.mean(axis=0)).transpose()
    head_distance_z_avg = head_distance_z_avg.add_suffix('_avgdist')
    # print(head_distance_z_avg)

    # max head distance from avg away from camera
    max_away_head_distance = pd.DataFrame(head_distance_z.max() - head_distance_z.mean()).transpose()
    max_away_head_distance = max_away_head_distance.add_suffix('_max_away')

    # min head distance from avg towards camera
    min_away_head_distance = pd.DataFrame(head_distance_z.min() - head_distance_z.mean()).transpose()
    min_away_head_distance = min_away_head_distance.add_suffix('_min_away')

    head_distance = pd.concat([head_distance_z_avg, max_away_head_distance, min_away_head_distance], axis=1)

    return head_distance


'----------------------------------------------------------------------------------------------------------------------'


# HEAD POSE
def calc_head_pose(dataframe):
    head_pose_name = dataframe.columns[296:299]
    head_pose_rot_xyz = dataframe[head_pose_name]

    # left(-)/right(+) head pose
    yaw_rot = dataframe[head_pose_name[1]]
    # print(yaw_rot)

    # check if left and right orientation exist
    # the higher the mean of left or roght rotation values the longer their head was rotation into a specific direction
    left_rot = [x for x in yaw_rot if x <= 0]  # negativ value
    right_rot = [x for x in yaw_rot if x >= 0]

    if not left_rot:
        left_rot_avg = 0
        left_rot_max = 0
    else:
        left_rot_avg = np.mean(left_rot)
        left_rot_max = min(left_rot)

    if not right_rot:
        right_rot_mean = 0
        right_rot_max = 0
    else:
        right_rot_mean = np.mean(right_rot)
        right_rot_max = max(right_rot)

    roll_rot = dataframe[head_pose_name[2]]
    skew_left = [x for x in roll_rot if x <= 0]  # negative
    skew_right = [x for x in roll_rot if x >= 0]

    if not skew_left:
        skew_left_avg = 0
        skew_left_max = 0
    else:
        skew_left_avg = np.mean(skew_left)
        skew_left_max = min(skew_left)

    if not skew_right:
        skew_right_avg = 0
        skew_right_max = 0
    else:
        skew_right_avg = np.mean(skew_right)
        skew_right_max = max(skew_right)

    head_pose_list = [
        [left_rot_avg, left_rot_max, right_rot_mean, right_rot_max, skew_left_avg, skew_left_max, skew_right_avg,
         skew_right_max]]
    head_pose = pd.DataFrame(head_pose_list, columns=['left_rot_avg', 'left_rot_max', 'right_rot_mean', 'right_rot_max',
                                                      'skew_left_avg', 'skew_left_max', 'skew_right_avg',
                                                      'skew_right_max'])

    return head_pose


'----------------------------------------------------------------------------------------------------------------------'


# AU45 - blinks
# frame length of blinks=1 and frame interval between blinks
def get_frame_interval_len(frame_diff):
    inter_interval = []
    frame_len = []
    length = 0
    for item in frame_diff:
        if item == 1:
            length += 1
        else:
            frame_len.append(length)
            inter_interval.append(item)
            blink_len = 0
    frame_len.append(length)
    return frame_len, inter_interval


# count how many blinks: frame blink length >=3 frames and frame interval between blinks >=5
# a blink last for at least 100 ms --> 3 frames// eyes open between blinks at around 150 ms --> 5 frames
def get_counts(frame_len, inter_interval):
    j = 0
    counts = 0
    for i in range(0, len(frame_len)):
        if i == len(frame_len) - 1:
            if frame_len[i] >= 5:
                counts += 1
        else:
            if frame_len[i] >= 3 and inter_interval[j] >= 5:
                counts += 1
                i += 1
                j += 1
            else:
                continue
    return counts


def get_blinks(dataframe):
    # only frame number of blinks==1
    all_blinks_frame = dataframe.loc[dataframe['AU45_c'] == 1, 'frame']

    # for every frame a blink is stated as 0 or 1, all frame numbers where a blink=1 are saved in order to calculate how
    # long and how many blinks exist (consecutive frame numbers = ongoing blink, long difference between frames = open)
    all_blinks_frame_diff = [list(all_blinks_frame)[n] - list(all_blinks_frame)[n - 1] for n in
                             range(1, len(list(all_blinks_frame)))]

    blink_frame_len, inter_blink_interval = get_frame_interval_len(all_blinks_frame_diff)

    if not blink_frame_len:
        blink_frame_len = [0]
    if not inter_blink_interval:
        inter_blink_interval = [0]

    blink_len_mean = pd.DataFrame([np.mean(blink_frame_len)], columns=['blink_len_mean'])
    blinks = pd.DataFrame([get_counts(blink_frame_len, inter_blink_interval)], columns=['blinks'])
    inter_blink_intervall_mean = pd.DataFrame([np.mean(inter_blink_interval)], columns=['blink_interval_mean'])

    blink_stuff = pd.concat([blinks, inter_blink_intervall_mean, blink_len_mean], axis=1)

    return blink_stuff


'----------------------------------------------------------------------------------------------------------------------'


# mouth open count for breathing activity as suggested by Peter Xie in
# https://towardsdatascience.com/how-to-detect-mouth-open-for-face-login-84ca834dff3b

# features_validationset_file = 'C:\\Users\\Tatjana\\Google Drive\\Engagement_Detection\\DAiSEE\\FeatureEx\\Validation\\400022\\4000221010.csv'
# dataframe = pd.read_csv(features_validationset_file)
# dataframe.columns = [col.replace(" ", "") for col in dataframe.columns]


def get_lip_height(lip):
    summ = 0
    for i in [2, 3, 4]:
        # distance between two near points up and down
        distance = math.sqrt((lip[i][0] - lip[12 - i][0]) ** 2 + (lip[i][1] - lip[12 - i][1]) ** 2)
        summ += distance
    return summ / 3


def get_mouth_height(top_lip, bottom_lip):
    summ = 0
    for i in [8, 9, 10]:
        # distance between two near points up and down
        distance = math.sqrt(
            (top_lip[i][0] - bottom_lip[18 - i][0]) ** 2 + (top_lip[i][1] - bottom_lip[18 - i][1]) ** 2)
        summ += distance
    return summ / 3


def check_mouth_open(top_lip, bottom_lip):
    top_lip_height = get_lip_height(top_lip)
    bottom_lip_height = get_lip_height(bottom_lip)
    mouth_height = get_mouth_height(top_lip, bottom_lip)

    # if mouth is open more than lip height * ratio, return true.
    ratio = 0.5
    if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
        return 1
    else:
        return 0


def get_frames_mouthopen(dataframe):
    top_lip_name_xy = ['x_48', 'y_48', 'x_49', 'y_49', 'x_50', 'y_50', 'x_51', 'y_51', 'x_52', 'y_52', 'x_53', 'y_53',
                       'x_54', 'y_54', 'x_64', 'y_64', 'x_63', 'y_63', 'x_62', 'y_62', 'x_61', 'y_61', 'x_60', 'y_60']
    top_lip_xy = dataframe[top_lip_name_xy]
    records_top = top_lip_xy.to_records(index=False)

    bottom_lip_name_xy = ['x_54', 'y_54', 'x_55', 'y_55', 'x_56', 'y_56', 'x_57', 'y_57', 'x_58', 'y_58', 'x_59',
                          'y_59', 'x_48', 'y_48', 'x_60', 'y_60', 'x_67', 'y_67', 'x_66', 'y_66', 'x_65', 'y_65',
                          'x_64', 'y_64']
    bottom_lip_xy = dataframe[bottom_lip_name_xy]
    records_bottom = bottom_lip_xy.to_records(index=False)

    mouth_open_frame_len = 0
    for i in range(0, len(top_lip_xy)):
        iter_top = iter(records_top[i])
        top_lip = list(zip(iter_top, iter_top))
        iter_bottom = iter(records_bottom[i])
        bottom_lip = list(zip(iter_bottom, iter_bottom))

        mouth_open = check_mouth_open(top_lip, bottom_lip)
        mouth_open_frame_len = mouth_open_frame_len + mouth_open

    mouth_open_frame_len = pd.DataFrame([mouth_open_frame_len], columns=['mouth_open_frame_len'])

    return mouth_open_frame_len


'-----------------------------------------------'


def get_mouthopen(dataframe):
    # mouth open AU25
    # only frame number of AU25==1
    all_mouthopen_frame = dataframe.loc[dataframe['AU25_c'] == 1, 'frame']

    # for every frame a blink is stated as 0 or 1, all frame numbers where a blink=1 are saved in order to calculate how
    # long and how many blinks exist (consecutive frame numbers = ongoing blink, long difference between frames = open)
    all_mouthopen_frame_diff = [list(all_mouthopen_frame)[n] - list(all_mouthopen_frame)[n - 1] for n in
                                range(1, len(list(all_mouthopen_frame)))]

    mouthopen_frame_len, inter_mouthopen_interval = get_frame_interval_len(all_mouthopen_frame_diff)

    if not mouthopen_frame_len:
        mouthopen_frame_len = [0]
    if not inter_mouthopen_interval:
        inter_mouthopen_interval = [0]

    mouthopen_len_mean = pd.DataFrame([np.mean(mouthopen_frame_len)],
                                      columns=['AU25_mouth_open_frame_len_mean'])  # mean frame length
    mouthopen = pd.DataFrame([get_counts(mouthopen_frame_len, inter_mouthopen_interval)], columns=['AU25_mouth_open'])
    inter_mouthopen_interval_mean = pd.DataFrame([np.mean(inter_mouthopen_interval)],
                                                 columns=['AU25_mouth_open_interval_mean'])

    mouth_stuff = pd.concat((mouthopen, inter_mouthopen_interval_mean, mouthopen_len_mean), axis=1)

    return mouth_stuff


'----------------------------------------------------------------------------------------------------------------------'
# # eye blink rate
# # eye aspect ratio by Soukupová and Čech (2016)
#
# def calc_blinks(dataframe):
#     # ['x_36', 'x_37', 'x_38', 'x_39', 'x_40', 'x_41']
#     # ['y_36', 'y_37', 'y_38', 'y_39', 'y_40', 'y_41']
#     # ['x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47']
#     # ['y_42', 'y_43', 'y_44', 'y_45', 'y_46', 'y_47']
#
#     l_36_41_x = dataframe.columns[335:341]
#     l_36_41_y = dataframe.columns[403:409]
#
#     r_42_47_x = dataframe.columns[341:347]
#     r_42_47_y = dataframe.columns[409:415]
#
#     left_eye_x = dataframe[l_36_41_x]  # 300,6
#     left_eye_y = dataframe[l_36_41_y]
#     right_eye_x = dataframe[r_42_47_x]
#     right_eye_y = dataframe[r_42_47_y]
#
#     # xy_36
#     left_eye = np.transpose(
#         [left_eye_x.iloc[:, 0].to_numpy(), left_eye_y.iloc[:, 0].to_numpy(), left_eye_x.iloc[:, 1].to_numpy(),
#          left_eye_y.iloc[:, 1].to_numpy(), left_eye_x.iloc[:, 2].to_numpy(), left_eye_y.iloc[:, 2].to_numpy(),
#          left_eye_x.iloc[:, 3].to_numpy(), left_eye_y.iloc[:, 3].to_numpy(), left_eye_x.iloc[:, 4].to_numpy(),
#          left_eye_y.iloc[:, 4].to_numpy(), left_eye_x.iloc[:, 5].to_numpy(), left_eye_y.iloc[:, 5].to_numpy()])
#     right_eye = np.transpose(
#         [right_eye_x.iloc[:, 0].to_numpy(), right_eye_y.iloc[:, 0].to_numpy(), right_eye_x.iloc[:, 1].to_numpy(),
#          right_eye_y.iloc[:, 1].to_numpy(), right_eye_x.iloc[:, 2].to_numpy(), right_eye_y.iloc[:, 2].to_numpy(),
#          right_eye_x.iloc[:, 3].to_numpy(), right_eye_y.iloc[:, 3].to_numpy(), right_eye_x.iloc[:, 4].to_numpy(),
#          right_eye_y.iloc[:, 4].to_numpy(), right_eye_x.iloc[:, 5].to_numpy(), right_eye_y.iloc[:, 5].to_numpy()])
#
#     p1_l = left_eye[:, 0:2]
#     p2_l = left_eye[:, 2:4]
#     p3_l = left_eye[:, 4:6]
#     p4_l = left_eye[:, 6:8]
#     p5_l = left_eye[:, 8:10]
#     p6_l = left_eye[:, 10:12]
#
#     p1_r = right_eye[:, 0:2]
#     p2_r = right_eye[:, 2:4]
#     p3_r = right_eye[:, 4:6]
#     p4_r = right_eye[:, 6:8]
#     p5_r = right_eye[:, 8:10]
#     p6_r = right_eye[:, 10:12]
#
#     A_l = np.sqrt((p2_l[:, 0] - p6_l[:, 0]) ** 2 + ((p2_l[:, 1] - p6_l[:, 1]) ** 2))
#     B_l = np.sqrt((p3_l[:, 0] - p5_l[:, 0]) ** 2 + ((p3_l[:, 1] - p5_l[:, 1]) ** 2))
#     C_l = np.sqrt((p1_l[:, 0] - p4_l[:, 0]) ** 2 + ((p1_l[:, 1] - p4_l[:, 1]) ** 2))
#
#     A_r = np.sqrt((p2_r[:, 0] - p6_r[:, 0]) ** 2 + ((p2_r[:, 1] - p6_r[:, 1]) ** 2))
#     B_r = np.sqrt((p3_r[:, 0] - p5_r[:, 0]) ** 2 + ((p3_r[:, 1] - p5_r[:, 1]) ** 2))
#     C_r = np.sqrt((p1_r[:, 0] - p4_r[:, 0]) ** 2 + ((p1_r[:, 1] - p4_r[:, 1]) ** 2))
#
#     # compute the eye aspect ratio
#     ear_l = (A_l + B_l) / (2.0 * C_l)
#     ear_r = (A_r + B_r) / (2.0 * C_r)
#
#     ear = (ear_l + ear_r) / 2
#     ear_n = ear * -1
#
#     # plot ear and look for neg peaks
#     # plt.plot(ear_n)
#     # plt.show()
#
#     # above = np.mean(ear_n)+ 2*np.std(ear_n)
#     # peaks, _ = find_peaks(ear_n, height= above)
#     # blinks_count = [len(peaks)]
#
#     # maxidistance = max(ear_n)- min(ear_n)
#     # print(maxidistance)
#     # print(np.mean(ear_n))
#     # plt.plot(ear_n)
#     # plt.plot(peaks, ear_n[peaks], "x")
#     # plt.hlines(np.mean(ear_n)+ 2*np.std(ear_n),0,300)
#     # plt.show()
#
#     peaks, _ = find_peaks(ear_n, height=-0.30)
#     blinks_count = [len(peaks)]
#
#     blinks = pd.DataFrame(blinks_count, columns=['blinks'])
#
#     return blinks
