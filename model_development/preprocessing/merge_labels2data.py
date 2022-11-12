"""
Labels and the test, train and validation sets are not equally ordered, also some videos are not labelled.
Hence some feature of respective videos must be deleted and the labels correctly assigned.
Explanation for the labels can be found in the description or project report.
"""

import pandas as pd
import os

cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, "../.."))

data_path = os.path.join(head, 'data\\interim\\')
data_path_labels = os.path.join(head, 'data\\external\\')

data_path2save = os.path.join(data_path, 'data_labelled\\')
if not os.path.isdir(data_path2save):
    os.mkdir(data_path2save)

path_labels2save = os.path.join(data_path_labels, 'merged_Labels')
if not os.path.isdir(path_labels2save):
    os.mkdir(path_labels2save)

# TEST SET
testset_file = os.path.join(data_path, 'calculated_features\\Testset.csv')
testset = pd.read_csv(testset_file)

# Remove empty spaces in column names.
testset.columns = [col.replace(" ", "") for col in testset.columns]
testset = testset.drop(columns=['clipID_avi'])
testset.rename(columns={'clipID': 'ClipID'}, inplace=True)

# Test Set Labels
data_labels = os.listdir(data_path_labels)
test_label_2class = data_path_labels + 'TestLabels_2Class.CSV'
test_label_2class_lessEng = data_path_labels + 'TestLabels_2Class_lessEng.CSV'
test_label_2class_onlyCon = data_path_labels + 'TestLabels_2Class_onlyCon.CSV'
test_label_2class_onlyFrus = data_path_labels + 'TestLabels_2Class_onlyFrus.CSV'
test_label_2class_onlyBor = data_path_labels + 'TestLabels_2Class_onlyBor.CSV'
test_label_3class = data_path_labels + 'TestLabels_3Class.CSV'

testLabels_2class = pd.read_csv(test_label_2class, sep=';')
# Remove empty spaces in column names.
testLabels_2class.columns = [col.replace(" ", "") for col in testLabels_2class.columns]
testLabels_2class = testLabels_2class.apply(pd.to_numeric)

testLabels_2class_lessEng = pd.read_csv(test_label_2class_lessEng, sep=';')
# Remove empty spaces in column names.
testLabels_2class_lessEng.columns = [col.replace(" ", "") for col in testLabels_2class_lessEng.columns]
testLabels_2class_lessEng = testLabels_2class_lessEng.apply(pd.to_numeric)

testLabels_2class_onlyCon = pd.read_csv(test_label_2class_onlyCon, sep=';')
# Remove empty spaces in column names.
testLabels_2class_onlyCon.columns = [col.replace(" ", "") for col in testLabels_2class_onlyCon.columns]
testLabels_2class_onlyCon = testLabels_2class_onlyCon.apply(pd.to_numeric)

testLabels_2class_onlyFrus = pd.read_csv(test_label_2class_onlyFrus, sep=';')
# Remove empty spaces in column names.
testLabels_2class_onlyFrus.columns = [col.replace(" ", "") for col in testLabels_2class_onlyFrus.columns]
testLabels_2class_onlyFrus = testLabels_2class_onlyFrus.apply(pd.to_numeric)

testLabels_2class_onlyBor = pd.read_csv(test_label_2class_onlyBor, sep=';')
# Remove empty spaces in column names.
testLabels_2class_onlyBor.columns = [col.replace(" ", "") for col in testLabels_2class_onlyBor.columns]
testLabels_2class_onlyBor = testLabels_2class_onlyBor.apply(pd.to_numeric)

testLabels_3class = pd.read_csv(test_label_3class, sep=';')
# Remove empty spaces in column names.
testLabels_3class.columns = [col.replace(" ", "") for col in testLabels_3class.columns]
testLabels_3class = testLabels_3class.apply(pd.to_numeric)

# Rearrange Labels of TestSet2_Class
testLabels_2class_names = testLabels_2class.columns
testLabels_2class_ordered = pd.DataFrame(columns=testLabels_2class_names)

# Rearrange Labels of TestSet2_Class_lessEng
testLabels_2class_lessEng_names = testLabels_2class_lessEng.columns
testLabels_2class_lessEng_ordered = pd.DataFrame(columns=testLabels_2class_lessEng_names)

# Rearrange Labels of TestSet2_Class_onlyCon
testLabels_2class_onlyCon_names = testLabels_2class_onlyCon.columns
testLabels_2class_onlyCon_ordered = pd.DataFrame(columns=testLabels_2class_onlyCon_names)

# Rearrange Labels of TestSet2_Class_onlyFrus
testLabels_2class_onlyFrus_names = testLabels_2class_onlyFrus.columns
testLabels_2class_onlyFrus_ordered = pd.DataFrame(columns=testLabels_2class_onlyFrus_names)

# Rearrange Labels of TestSet2_Class_onlyBor
testLabels_2class_onlyBor_names = testLabels_2class_onlyBor.columns
testLabels_2class_onlyBor_ordered = pd.DataFrame(columns=testLabels_2class_onlyBor_names)

# Rearrange Labels of TestSet3_Class
testLabels_3class_names = testLabels_3class.columns
testLabels_3class_ordered = pd.DataFrame(columns=testLabels_3class_names)

for iD in testset['ClipID']:
    # print(iD)
    if iD in testLabels_2class['ClipID'].values:
        # trainLabels_2class.loc[trainLabels_2class['ClipID'] == iD]
        testLabels_2class_ordered = testLabels_2class_ordered.append(
            testLabels_2class.loc[testLabels_2class['ClipID'] == iD])
        testLabels_2class_lessEng_ordered = testLabels_2class_lessEng_ordered.append(
            testLabels_2class_lessEng.loc[testLabels_2class_lessEng['ClipID'] == iD])
        testLabels_2class_onlyCon_ordered = testLabels_2class_onlyCon_ordered.append(
            testLabels_2class_onlyCon.loc[testLabels_2class_onlyCon['ClipID'] == iD])
        testLabels_2class_onlyFrus_ordered = testLabels_2class_onlyFrus_ordered.append(
            testLabels_2class_onlyFrus.loc[testLabels_2class_onlyFrus['ClipID'] == iD])
        testLabels_2class_onlyBor_ordered = testLabels_2class_onlyBor_ordered.append(
            testLabels_2class_onlyBor.loc[testLabels_2class_onlyBor['ClipID'] == iD])
        testLabels_3class_ordered = testLabels_3class_ordered.append(
            testLabels_3class.loc[testLabels_3class['ClipID'] == iD])
    else:
        testset = testset.drop(testset.loc[testset['ClipID'] == iD].index)
testLabels_2class_ordered.reset_index(drop=True)
testLabels_2class_lessEng_ordered.reset_index(drop=True)
testLabels_2class_onlyCon_ordered.reset_index(drop=True)
testLabels_2class_onlyFrus_ordered.reset_index(drop=True)
testLabels_2class_onlyBor_ordered.reset_index(drop=True)
testLabels_3class_ordered.reset_index(drop=True)
testset = testset.reset_index(drop=True)
testset = pd.concat(
    [testset, testLabels_2class_ordered['0=EngBor//1=C+F'], testLabels_2class_lessEng_ordered['0=EngBorLess//1=C+F'],
     testLabels_2class_onlyCon_ordered['OnlyCon'], testLabels_2class_onlyFrus_ordered['OnlyFrus'],
     testLabels_2class_onlyBor_ordered['OnlyBor'], testLabels_3class_ordered['0=C+F//1=EngOnly//2=BorEng']], sort=False,
    axis=1)

testLabels_2class_ordered.to_csv(path_labels2save + '\\TestLabels_2class.csv', index=False)
testLabels_2class_lessEng_ordered.to_csv(path_labels2save + '\\TestLabels_2class_lessEng.csv', index=False)
testLabels_2class_onlyCon_ordered.to_csv(path_labels2save + '\\TestLabels_2class_onlyCon.csv', index=False)
testLabels_2class_onlyFrus_ordered.to_csv(path_labels2save + '\\TestLabels_2class_onlyFrus.csv', index=False)
testLabels_2class_onlyBor_ordered.to_csv(path_labels2save + '\\TestLabels_2class_onlyBor.csv', index=False)
testLabels_3class_ordered.to_csv(path_labels2save + '\\TestLabels_3class.csv', index=False)
testset.to_csv(data_path2save + 'Testset.csv', index=False)

########################################################################################################################
# TRAIN SET
data_path2save = os.path.join(data_path, 'data_labelled\\')
if not os.path.isdir(data_path2save):
    os.mkdir(data_path2save)

trainset_file = os.path.join(data_path, 'calculated_features\\Trainset.csv')
trainset = pd.read_csv(trainset_file)

# Remove empty spaces in column names.
trainset.columns = [col.replace(" ", "") for col in trainset.columns]
trainset = trainset.drop(columns=['clipID_avi'])
trainset.rename(columns={'clipID': 'ClipID'}, inplace=True)

# TRAIN SET LABELS
train_label_2class = data_path_labels + 'TrainLabels_2Class.CSV'
train_label_2class_lessEng = data_path_labels + 'TrainLabels_2Class_lessEng.CSV'
train_label_2class_onlyCon = data_path_labels + 'TrainLabels_2Class_onlyCon.CSV'
train_label_2class_onlyFrus = data_path_labels + 'TrainLabels_2Class_onlyFrus.CSV'
train_label_2class_onlyBor = data_path_labels + 'TrainLabels_2Class_onlyBor.CSV'
train_label_3class = data_path_labels + 'TrainLabels_3Class.csv'

trainLabels_2class = pd.read_csv(train_label_2class, sep=';')
# Remove empty spaces in column names.
trainLabels_2class.columns = [col.replace(" ", "") for col in trainLabels_2class.columns]
trainLabels_2class = trainLabels_2class.apply(pd.to_numeric)

trainLabels_2class_lessEng = pd.read_csv(train_label_2class_lessEng, sep=';')
# Remove empty spaces in column names.
trainLabels_2class_lessEng.columns = [col.replace(" ", "") for col in trainLabels_2class_lessEng.columns]
trainLabels_2class_lessEng = trainLabels_2class_lessEng.apply(pd.to_numeric)

trainLabels_2class_onlyCon = pd.read_csv(train_label_2class_onlyCon, sep=';')
# Remove empty spaces in column names.
trainLabels_2class_onlyCon.columns = [col.replace(" ", "") for col in trainLabels_2class_onlyCon.columns]
trainLabels_2class_onlyCon = trainLabels_2class_onlyCon.apply(pd.to_numeric)

trainLabels_2class_onlyFrus = pd.read_csv(train_label_2class_onlyFrus, sep=';')
# Remove empty spaces in column names.
trainLabels_2class_onlyFrus.columns = [col.replace(" ", "") for col in trainLabels_2class_onlyFrus.columns]
trainLabels_2class_onlyFrus = trainLabels_2class_onlyFrus.apply(pd.to_numeric)

trainLabels_2class_onlyBor = pd.read_csv(train_label_2class_onlyBor, sep=';')
# Remove empty spaces in column names.
trainLabels_2class_onlyBor.columns = [col.replace(" ", "") for col in trainLabels_2class_onlyBor.columns]
trainLabels_2class_onlyBor = trainLabels_2class_onlyBor.apply(pd.to_numeric)

trainLabels_3class = pd.read_csv(train_label_3class, sep=';')
# Remove empty spaces in column names.
trainLabels_3class.columns = [col.replace(" ", "") for col in trainLabels_3class.columns]
trainLabels_3class = trainLabels_3class.apply(pd.to_numeric)

# Rearrange Labels of TrainSet2_Class
trainLabels_2class_names = trainLabels_2class.columns
trainLabels_2class_ordered = pd.DataFrame(columns=trainLabels_2class_names)

# Rearrange Labels of TrainSet2_Class_lessEng
trainLabels_2class_lessEng_names = trainLabels_2class_lessEng.columns
trainLabels_2class_lessEng_ordered = pd.DataFrame(columns=trainLabels_2class_lessEng_names)

# Rearrange Labels of TrainSet2_Class_onlyCon
trainLabels_2class_onlyCon_names = trainLabels_2class_onlyCon.columns
trainLabels_2class_onlyCon_ordered = pd.DataFrame(columns=trainLabels_2class_onlyCon_names)

# Rearrange Labels of TrainSet2_Class_onlyFrus
trainLabels_2class_onlyFrus_names = trainLabels_2class_onlyFrus.columns
trainLabels_2class_onlyFrus_ordered = pd.DataFrame(columns=trainLabels_2class_onlyFrus_names)

# Rearrange Labels of TrainSet2_Class_onlyBor
trainLabels_2class_onlyBor_names = trainLabels_2class_onlyBor.columns
trainLabels_2class_onlyBor_ordered = pd.DataFrame(columns=trainLabels_2class_onlyBor_names)

# Rearrange Labels of TrainSet3_Class
trainLabels_3class_names = trainLabels_3class.columns
trainLabels_3class_ordered = pd.DataFrame(columns=trainLabels_3class_names)

for iD in trainset['ClipID']:
    # print(iD)
    if iD in trainLabels_2class['ClipID'].values:
        # trainLabels_2class.loc[trainLabels_2class['ClipID'] == iD]
        trainLabels_2class_ordered = trainLabels_2class_ordered.append(
            trainLabels_2class.loc[trainLabels_2class['ClipID'] == iD])
        trainLabels_2class_lessEng_ordered = trainLabels_2class_lessEng_ordered.append(
            trainLabels_2class_lessEng.loc[trainLabels_2class_lessEng['ClipID'] == iD])
        trainLabels_2class_onlyCon_ordered = trainLabels_2class_onlyCon_ordered.append(
            trainLabels_2class_onlyCon.loc[trainLabels_2class_onlyCon['ClipID'] == iD])
        trainLabels_2class_onlyFrus_ordered = trainLabels_2class_onlyFrus_ordered.append(
            trainLabels_2class_onlyFrus.loc[trainLabels_2class_onlyFrus['ClipID'] == iD])
        trainLabels_2class_onlyBor_ordered = trainLabels_2class_onlyBor_ordered.append(
            trainLabels_2class_onlyBor.loc[trainLabels_2class_onlyBor['ClipID'] == iD])
        trainLabels_3class_ordered = trainLabels_3class_ordered.append(
            trainLabels_3class.loc[trainLabels_3class['ClipID'] == iD])
    else:
        trainset = trainset.drop(trainset.loc[trainset['ClipID'] == iD].index)

# trainLabels_2class_ordered.reset_index(drop=True)
trainset = trainset.reset_index(drop=True)
trainLabels_2class_ordered = trainLabels_2class_ordered.reset_index(drop=True)
trainLabels_2class_lessEng_ordered = trainLabels_2class_lessEng_ordered.reset_index(drop=True)
trainLabels_2class_onlyCon_ordered = trainLabels_2class_onlyCon_ordered.reset_index(drop=True)
trainLabels_2class_onlyFrus_ordered = trainLabels_2class_onlyFrus_ordered.reset_index(drop=True)
trainLabels_2class_onlyBor_ordered = trainLabels_2class_onlyBor_ordered.reset_index(drop=True)
trainLabels_3class_ordered = trainLabels_3class_ordered.reset_index(drop=True)
labels = pd.concat(
    [trainLabels_2class_ordered['0=EngBor//1=C+F'], trainLabels_2class_lessEng_ordered['0=EngBorLess//1=C+F'],
     trainLabels_2class_onlyCon_ordered['OnlyCon'], trainLabels_2class_onlyFrus_ordered['OnlyFrus'],
     trainLabels_2class_onlyBor_ordered['OnlyBor'], trainLabels_3class_ordered['0=C+F//1=EngOnly//2=BorEng']],
    sort=False, axis=1)
trainset = pd.concat((trainset, labels), sort=False, axis=1, join='inner')

trainLabels_2class_ordered.to_csv(path_labels2save + '\\TrainLabels_2class.csv', index=False)
trainLabels_2class_lessEng_ordered.to_csv(path_labels2save + '\\TrainLabels_2class_lessEng.csv', index=False)
trainLabels_2class_onlyCon_ordered.to_csv(path_labels2save + '\\TrainLabels_2class_onlyCon.csv', index=False)
trainLabels_2class_onlyFrus_ordered.to_csv(path_labels2save + '\\TrainLabels_2class_onlyFrus.csv', index=False)
trainLabels_2class_onlyBor_ordered.to_csv(path_labels2save + '\\TrainLabels_2class_onlyBor.csv', index=False)
trainLabels_3class_ordered.to_csv(path_labels2save + '\\TrainLabels_3class.csv', index=False)
trainset.to_csv(data_path2save + 'Trainset.csv', index=False)

########################################################################################################################

# VALIDATION SET
data_path2save = os.path.join(data_path, 'data_labelled\\')
if not os.path.isdir(data_path2save):
    os.mkdir(data_path2save)

validationset_file = os.path.join(data_path, 'calculated_features\\Validationset.csv')

validationset = pd.read_csv(validationset_file)

# Remove empty spaces in column names.
validationset.columns = [col.replace(" ", "") for col in validationset.columns]
validationset = validationset.drop(columns=['clipID_avi'])
validationset.rename(columns={'clipID': 'ClipID'}, inplace=True)

# VALIDATION SET LABELS

validation_label_2class = data_path_labels + 'ValidationLabels_2Class.CSV'
validation_label_2class_lessEng = data_path_labels + 'ValidationLabels_2Class_lessEng.CSV'
validation_label_2class_onlyCon = data_path_labels + 'ValidationLabels_2Class_onlyCon.CSV'
validation_label_2class_onlyFrus = data_path_labels + 'ValidationLabels_2Class_onlyFrus.CSV'
validation_label_2class_onlyBor = data_path_labels + 'ValidationLabels_2Class_onlyBor.CSV'
validation_label_3class = data_path_labels + 'ValidationLabels_3Class.csv'

validationLabels_2class = pd.read_csv(validation_label_2class, sep=';')
# Remove empty spaces in column names.
validationLabels_2class.columns = [col.replace(" ", "") for col in validationLabels_2class.columns]
validationLabels_2class = validationLabels_2class.apply(pd.to_numeric)

validationLabels_2class_lessEng = pd.read_csv(validation_label_2class_lessEng, sep=';')
# Remove empty spaces in column names.
validationLabels_2class_lessEng.columns = [col.replace(" ", "") for col in validationLabels_2class_lessEng.columns]
validationLabels_2class_lessEng = validationLabels_2class_lessEng.apply(pd.to_numeric)

validationLabels_2class_onlyCon = pd.read_csv(validation_label_2class_onlyCon, sep=';')
# Remove empty spaces in column names.
validationLabels_2class_onlyCon.columns = [col.replace(" ", "") for col in validationLabels_2class_onlyCon.columns]
validationLabels_2class_onlyCon = validationLabels_2class_onlyCon.apply(pd.to_numeric)

validationLabels_2class_onlyFrus = pd.read_csv(validation_label_2class_onlyFrus, sep=';')
# Remove empty spaces in column names.
validationLabels_2class_onlyFrus.columns = [col.replace(" ", "") for col in validationLabels_2class_onlyFrus.columns]
validationLabels_2class_onlyFrus = validationLabels_2class_onlyFrus.apply(pd.to_numeric)

validationLabels_2class_onlyBor = pd.read_csv(validation_label_2class_onlyBor, sep=';')
# Remove empty spaces in column names.
validationLabels_2class_onlyBor.columns = [col.replace(" ", "") for col in validationLabels_2class_onlyBor.columns]
validationLabels_2class_onlyBor = validationLabels_2class_onlyBor.apply(pd.to_numeric)

validationLabels_3class = pd.read_csv(validation_label_3class, sep=';')
# Remove empty spaces in column names.
validationLabels_3class.columns = [col.replace(" ", "") for col in validationLabels_3class.columns]
validationLabels_3class = validationLabels_3class.apply(pd.to_numeric)

# Rearrange Labels of ValidationSet2_Class
validationLabels_2class_names = validationLabels_2class.columns
validationLabels_2class_ordered = pd.DataFrame(columns=validationLabels_2class_names)

# Rearrange Labels of ValidationSet2_Class_lessEng
validationLabels_2class_lessEng_names = validationLabels_2class_lessEng.columns
validationLabels_2class_lessEng_ordered = pd.DataFrame(columns=validationLabels_2class_lessEng_names)

# Rearrange Labels of ValidationSet2_Class_onlyCon
validationLabels_2class_onlyCon_names = validationLabels_2class_onlyCon.columns
validationLabels_2class_onlyCon_ordered = pd.DataFrame(columns=validationLabels_2class_onlyCon_names)

# Rearrange Labels of ValidationSet2_Class_onlyFrus
validationLabels_2class_onlyFrus_names = validationLabels_2class_onlyFrus.columns
validationLabels_2class_onlyFrus_ordered = pd.DataFrame(columns=validationLabels_2class_onlyFrus_names)

# Rearrange Labels of ValidationSet2_Class_onlyBor
validationLabels_2class_onlyBor_names = validationLabels_2class_onlyBor.columns
validationLabels_2class_onlyBor_ordered = pd.DataFrame(columns=validationLabels_2class_onlyBor_names)

# Rearrange Labels of ValidationSet3_Class
validationLabels_3class_names = validationLabels_3class.columns
validationLabels_3class_ordered = pd.DataFrame(columns=validationLabels_3class_names)

for iD in validationset['ClipID']:
    # print(iD)
    if iD in validationLabels_2class['ClipID'].values:
        # trainLabels_2class.loc[trainLabels_2class['ClipID'] == iD]
        validationLabels_2class_ordered = validationLabels_2class_ordered.append(
            validationLabels_2class.loc[validationLabels_2class['ClipID'] == iD])
        validationLabels_2class_lessEng_ordered = validationLabels_2class_lessEng_ordered.append(
            validationLabels_2class_lessEng.loc[validationLabels_2class_lessEng['ClipID'] == iD])
        validationLabels_2class_onlyCon_ordered = validationLabels_2class_onlyCon_ordered.append(
            validationLabels_2class_onlyCon.loc[validationLabels_2class_onlyCon['ClipID'] == iD])
        validationLabels_2class_onlyFrus_ordered = validationLabels_2class_onlyFrus_ordered.append(
            validationLabels_2class_onlyFrus.loc[validationLabels_2class_onlyFrus['ClipID'] == iD])
        validationLabels_2class_onlyBor_ordered = validationLabels_2class_onlyBor_ordered.append(
            validationLabels_2class_onlyBor.loc[validationLabels_2class_onlyBor['ClipID'] == iD])
        validationLabels_3class_ordered = validationLabels_3class_ordered.append(
            validationLabels_3class.loc[validationLabels_3class['ClipID'] == iD])
    else:
        validationset = validationset.drop(validationset.loc[validationset['ClipID'] == iD].index)

validationLabels_2class_ordered.reset_index(drop=True)
validationLabels_2class_lessEng_ordered.reset_index(drop=True)
validationLabels_2class_onlyCon_ordered.reset_index(drop=True)
validationLabels_2class_onlyFrus_ordered.reset_index(drop=True)
validationLabels_2class_onlyBor_ordered.reset_index(drop=True)
validationLabels_3class_ordered.reset_index(drop=True)
validationset = validationset.reset_index(drop=True)
validationset = pd.concat([validationset, validationLabels_2class_ordered['0=EngBor//1=C+F'],
                           validationLabels_2class_lessEng_ordered['0=EngBorLess//1=C+F'],
                           validationLabels_2class_onlyCon_ordered['OnlyCon'],
                           validationLabels_2class_onlyFrus_ordered['OnlyFrus'],
                           validationLabels_2class_onlyBor_ordered['OnlyBor'],
                           validationLabels_3class_ordered['0=C+F//1=EngOnly//2=BorEng']], sort=False, axis=1)

validationLabels_2class_ordered.to_csv(path_labels2save + '\\ValidationLabels_2class.csv', index=False)
validationLabels_2class_lessEng_ordered.to_csv(path_labels2save + '\\ValidationLabels_2class_lessEng.csv', index=False)
validationLabels_2class_onlyCon_ordered.to_csv(path_labels2save + '\\ValidationLabels_2class_onlyCon.csv', index=False)
validationLabels_2class_onlyFrus_ordered.to_csv(path_labels2save + '\\ValidationLabels_2class_onlyFrus.csv', index=False)
validationLabels_2class_onlyBor_ordered.to_csv(path_labels2save + '\\ValidationLabels_2class_onlyBor.csv', index=False)
validationLabels_3class_ordered.to_csv(path_labels2save + '\\ValidationLabels_3class.csv', index=False)
validationset.to_csv(data_path2save + 'Validationset.csv', index=False)
