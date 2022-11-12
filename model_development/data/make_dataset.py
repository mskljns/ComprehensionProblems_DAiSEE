"""
Extracting face parameters with OpenFace for the whole DAiSEE Dataset.

When using PyCharm, openface for Windows should be in python.exe directory and install as stated here:
https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation
"""

import os
import subprocess
import time
import sys

def make_dirs():
    test_path = os.path.join(data_path_raw, 'Test')
    if not os.path.isdir(test_path):
        os.mkdir(test_path)
    train_path = os.path.join(data_path_raw, 'Train')
    print(train_path)
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    validation_path = os.path.join(data_path_raw, 'Validation')
    if not os.path.isdir(validation_path):
        os.mkdir(validation_path)


def main():
    python_path = os.path.dirname(os.path.abspath(sys.executable))
    openface_path = os.path.abspath(os.path.join(python_path, 'openface'))
    cwd = os.path.dirname(os.path.abspath(__file__))
    head = os.path.abspath(os.path.join(cwd, "../.."))
    data_path_daisee = os.path.join(head, 'data\\raw\\DAiSEE\\DataSet\\')
    data_path_raw = os.path.join(head, 'data\\raw\\')
    make_dirs()
    data_set = ['Test', 'Train', 'Validation']

    for ttv in data_set:
        subjects = os.listdir(data_path_daisee + ttv + '\\')

        for subject in subjects:
            output_dir =  data_path_raw + ttv + '\\' + subject
            print(output_dir)
            if not os.path.isdir(output_dir):
                print('not')
                os.mkdir(output_dir)

            curr_subject = os.listdir(data_path_daisee + ttv + '\\' + subject + '\\')

            for video in curr_subject:
                clip = os.listdir(data_path_daisee + ttv + '\\' + subject + '\\' + video + '\\')
                clip_path = data_path_daisee + ttv + '\\' + subject + '\\' + video + '\\' + clip[0]
                print(clip_path)

                args = 'FeatureExtraction.exe -f "{clip_path}" -2Dfp -3Dfp -pdmparams -pose -aus -gaze -out_dir "{output}"'.format(
                    clip_path=clip_path, output=output_dir)

                start = time.time()

                # add OpenFace path in your python dir
                subprocess.call(args, shell=True, cwd=openface_path)
                end = time.time()
                print(end - start)


if __name__ == "__main__":
    main()