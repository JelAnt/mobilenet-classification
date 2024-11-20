import os
import random
import shutil

splitsize = 0.85
categories = []

source_folder = "C:/Users/marko/Desktop/computer vision project/garbage_classification"
folders = os.listdir(source_folder)


for folder in folders:
    if os.path.isdir(source_folder + "/" + folder):
        categories.append(folder)


categories.sort()
print(categories)


# create target folder

target_folder = "C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model"
existDataSetPath = os.path.exists(target_folder)
if existDataSetPath == False:
    os.makedirs(target_folder)


# split data for train and validation
def split_data(SOURCE, training, validation, split_size):
    files = []

    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        print(file)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignore it.")

    training_length = int(len(files) * split_size)
    shuffleSet = random.sample(files, len(files))
    trainingSet = shuffleSet[0:training_length]
    validSet = shuffleSet[training_length:]

    # copy each image from the source to the target
    for filename in trainingSet:
        thisFile = SOURCE + filename
        destination = training + filename
        shutil.copyfile(thisFile, destination)

    # copy each image from the source to the target
    for filename in validSet:
        thisFile = SOURCE + filename
        destination = validation + filename
        shutil.copyfile(thisFile, destination)


trainPath = target_folder + "/train/"
validatePath = target_folder + "/validate/"

# create the target folders:

existsDataSetPath = os.path.exists(trainPath)
if existDataSetPath == False:
    os.makedirs(trainPath)

existsDataSetPath = os.path.exists(validatePath)
if existDataSetPath == False:
    os.makedirs(validatePath)


# run function

for category in categories:
    trainDestPath = trainPath + "/" + category
    validateDestPath = validatePath + "/" + category

    if os.path.exists(trainDestPath) == False:
        os.makedirs(trainDestPath)

    if os.path.exists(validateDestPath) == False:
        os.makedirs(validateDestPath)

    sourcePath = source_folder + "/" + category + "/"
    trainDestPath = trainDestPath + "/"
    validateDestPath = validateDestPath + "/"

    print(
        "Copy from: "
        + sourcePath
        + " to: "
        + trainDestPath
        + " and: "
        + validateDestPath
    )

    split_data(sourcePath, trainDestPath, validateDestPath, splitsize)
