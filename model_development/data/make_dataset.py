# """
# Extracting face parameters with OpenFace for the whole DAiSEE Dataset.
#
# When using PyCharm, openface for Windows should be in python.exe directory and install as stated here:
# https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation
# """
#
# import os
# import subprocess
# import time
# import sys
#
# python_path = os.path.dirname(os.path.abspath(sys.executable))
# openface_path = os.path.abspath(os.path.join(python_path, 'openface'))
#
# cwd = os.path.dirname(os.path.abspath(__file__))
# head = os.path.abspath(os.path.join(cwd, os.pardir))
#
# # data path to the DAiSEE Dataset
# data_path_raw = os.path.join(head, 'data\\raw\\DAiSEE\\DataSet\\')
# data_path = os.path.join(head, 'data')
#
#
# def make_dirs():
#     interim_path = os.path.join(data_path, 'interim')
#     if not os.path.isdir(interim_path):
#         os.mkdir(interim_path)
#     processed_path = os.path.join(data_path, 'processed')
#     if not os.path.isdir(processed_path):
#         os.mkdir(processed_path)
#
#
# def main():
#     make_dirs()
#     data_set = ['Train', 'Test', 'Validation']
#
#     #clips_list = []
#
#     for ttv in data_set:
#         subjects = os.listdir(data_path_raw + ttv + '\\')
#
#         for subject in subjects:
#             # output_dir = '...\\DAiSEE\\Feature_Extraction_PyCh\\' + ttv + '/' + subject + '/'  # <--- output path
#             #output_dir = os.path.join(head, 'data\\interim\\') + ttv + '/' + subject + '/'
#             output_dir = 'C:\\Users\\Tatjana\\PycharmProjects\\ComProbProject\\data\\interim\\' + ttv + '\\' + subject + '\\'
#             print(output_dir)
#             curr_subject = os.listdir(data_path_raw + ttv + '\\' + subject + '\\')
#
#             for video in curr_subject:
#                 clip = os.listdir(data_path_raw + ttv + '\\' + subject + '\\' + video + '\\')
#                 clip_path = data_path_raw + ttv + '\\' + subject + '\\' + video + '\\' + clip[0]
#
#                 args = 'FeatureExtraction.exe -fdir "{clip_path}" -2Dfp -3Dfp -pdmparams -pose -aus -gaze -out_dir "{output}"'.format(
#                     clip_path=clip_path, output=output_dir)
#
#                 start = time.time()
#
#                 # add OpenFace path in your python dir
#                 subprocess.call(args, shell=True, cwd=openface_path)
#                 end = time.time()
#                 print(end - start)
#
#
# if __name__ == "__main__":
#     main()
#


"""
Extracting face parameters with OpenFace
when using PyCharm, openface for Windows should be in python.exe directory and install as stated here:
https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation
"""


import os
import subprocess
# import time


# openface_path = 'C:\\Users\\Tatjana\\AppData\\Local\\Programs\\Python\\Python37\\openface\\'

#
#data_path = 'C:\\Users\\Tatjana\\Google Drive\\Engagement_Detection\\DAiSEE\\DataSet\\'
data_path = 'C:\\Users\\Tatjana\\PycharmProjects\\ComProbProject\\data\\raw\\DAiSEE\\DataSet\\'

data_set = ['Test', 'Train', 'Validation']

clips_list = []

for ttv in data_set:
    subjects = os.listdir(data_path + ttv + '\\')

    for subject in subjects:
        output_dir = 'C:\\Users\\Tatjana\\PycharmProjects\\ComProbProject\\data\\interim\\'+ttv+'\\' + subject+'\\'
        curr_subject = os.listdir(data_path + ttv +'\\' + subject + '\\')

        for video in curr_subject:
            clip = os.listdir(data_path + ttv + '\\' + subject + '\\' + video + '\\')
            clip_path = data_path + ttv + '\\' + subject + '\\' + video + '\\' + clip[0]
            print()
            print(clip_path)
            args = 'FeatureExtraction.exe -fdir "{clip_path}" -2Dfp -3Dfp -pdmparams -pose -aus -gaze -out_dir "{output}"'.format(clip_path=clip_path, output=output_dir)
            print(output_dir)
            print()
            # call command line
            #start = time.time()
            subprocess.call(args, shell=True,
                            cwd='C:\\Users\\Tatjana\\AppData\\Local\\Programs\\Python\\Python37\\openface\\')
            #end = time.time()
            #print(end - start)
