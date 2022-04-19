import sys
import os
import cv2
import numpy as np
import glob

# Buiuld class for loading data
class DatasetLoad:
    # Inintialize the class
    def __init__ (self, width = 64, height = 64, pre_type = 'Resize'):
        # give default preprocessing type is resize
        self.width  = width
        self.height = height
        self.pre_type = pre_type

    # load dataset
    def load (self, pathes, verbose = -1):
        # verbose is used to show the loading status on screen
        # initial empty datas and labels
        datas = []
        labels = []

        # initial the path (main pathes, eg. datasets/animals)
        mainfolder = os.listdir(pathes)
        # this os.listdir of that pathes will get everything in side the folder
        # eg. inside folder animals, it has cats, dogs and panda
        # then folders = [cats, dogs, panda]

        for folder in mainfolder:
            # loop inside mainfolders to read each folder
            # os.path.join use to join pathes with folder inside it
            # eg. pathes = datasets/animals
            # folder = cats
            # then fullpath will  get datasets/animals/cats
            fullpath =os.path.join(pathes, folder)
            print(fullpath)
            #fullpath = "data_1/data/Train/animals/"
            #print(fullpath)
            # list all files that has in full path
            # in this eg. those file is image in each folder
            listfiles = os.listdir(fullpath)
            #listfiles = glob.glob(fullpath)
            # Print on screen
            if verbose > 0:
                print('[INFO] loading ', folder, ' ...')

            for (i, imagefile) in enumerate(listfiles):
                # Define full path of image 
                imagepath = pathes+'/'+folder+'/'+imagefile
                # Read image from imagepath
                image = cv2.imread(imagepath)
                # Give label image based on folder
                label = folder

                # Because of input image is not the same size
                # we have to resize it to have same size
                # That is why the default of pre_type is "Resize"
                if (self.pre_type == 'Resize'):
                    image = cv2.resize(image, (self.width, self.height), interpolation = cv2.INTER_AREA)

                datas.append(image)
                labels.append(label)

                if verbose > 0 and i > 0 and (i+1)%verbose == 0 :
                    print('[INFO] processed {}/{}'.format(i+1, len(listfiles)))

        return (np.array(datas), np.array(labels))