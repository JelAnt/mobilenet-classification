import os
import random
import shutil

splitsize = (
    0.85  # variable used to specify that 85% of images will be used for training
)
categories = []  # list that will hold possible categories of images

source_folder = "C:/Users/marko/Desktop/computer vision project/garbage_classification"  # location of the dataset folder
folders = os.listdir(
    source_folder
)  # creating a list of folders inside of the dataset folder


for (
    folder
) in (
    folders
):  # for each folder inside the dataset folder append the folder name to the category list
    if os.path.isdir(source_folder + "/" + folder):
        categories.append(folder)


categories.sort()  # sort the category list
print(categories)  # print to check if its correct


target_folder = "C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model"  # specify the location of where the modified dataset folder will be stored
existDataSetPath = os.path.exists(target_folder)  # check uf it already exists
if existDataSetPath == False:
    os.makedirs(target_folder)  # if it doesn't exist it gets created


def split_data(
    SOURCE, training, validation, split_size
):  # function that will split the dataset into training and validation data
    files = []  # list to hold file names

    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        print(file)
        if os.path.getsize(file) > 0:
            files.append(
                filename
            )  # if the file doesn't already exist in the list add it to it
        else:
            print(filename + " empty")

    training_length = int(
        len(files) * split_size
    )  # specify the length of the training data
    shuffleSet = random.sample(files, len(files))  # shuffle all the files
    trainingSet = shuffleSet[
        0:training_length
    ]  # specifies the files that belong to the training set
    validSet = shuffleSet[
        training_length:
    ]  # the rest of the files belong to the validation set

    for filename in trainingSet:  # copies training set files to the destination folder
        thisFile = SOURCE + filename
        destination = training + filename
        shutil.copyfile(thisFile, destination)

    for filename in validSet:  # copies validset files to the destination folder
        thisFile = SOURCE + filename
        destination = validation + filename
        shutil.copyfile(thisFile, destination)


trainPath = target_folder + "/train/"
validatePath = (
    target_folder + "/validate/"
)  # specifying paths of training and validation folders


# creates training and validation folder if they don't already exist
existsDataSetPath = os.path.exists(trainPath)
if existDataSetPath == False:
    os.makedirs(trainPath)

existsDataSetPath = os.path.exists(validatePath)
if existDataSetPath == False:
    os.makedirs(validatePath)


for (
    category
) in categories:  # create an appropriate folder for each category in the list
    trainDestPath = trainPath + "/" + category
    validateDestPath = validatePath + "/" + category

    if os.path.exists(trainDestPath) == False:  # if it doesn't already exist create it
        os.makedirs(trainDestPath)

    if os.path.exists(validateDestPath) == False:
        os.makedirs(validateDestPath)

    sourcePath = source_folder + "/" + category + "/"  # specify the source path
    trainDestPath = trainDestPath + "/"  # specify the trainign destination
    validateDestPath = validateDestPath + "/"  # specify the validation destination

    split_data(
        sourcePath, trainDestPath, validateDestPath, splitsize
    )  # run the splitdata function. (finish preparing the dataset)
