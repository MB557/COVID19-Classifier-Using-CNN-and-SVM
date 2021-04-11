# Basic libraries
import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle as pkl

labels = [0, 1] # 0 = noncovid, 1 = covid

def resize(dirs, S, dest):
    '''
        1) Making a list of images
        2) Resizing the images to SxSx3
        3) Converting the lists into numpy arrays

        params : 
            dirs -> COVID or Non-COVID
            S    -> Resultant Size after Resizing

        return : 
            numpy_array -> contains the images (n, S, S, 3)
    '''
    path = dest + '/'+ dirs + '/'
    array_of_imgs = []
    path_to_each_image = []    
    for images in sorted(os.listdir(path)):
        path_to_each_image.append(path + images)

    for sub_path in tqdm(path_to_each_image):
        img = cv2.imread(sub_path)

        try:
            # resize to SxSx3
            new_img = cv2.resize(img, (S,S), interpolation = cv2.INTER_AREA)
        except:
            continue
        
        # inserting into the array
        array_of_imgs.append(new_img)


    # convert lists to numpy arrays
    numpy_array = np.array(array_of_imgs)

    return numpy_array

def dump_into_pkl(data, name):
    '''
        dumps the data into a pkl file
        
        params : 
            data -> whatever you want to dump
            name -> name of the file
    '''
    outfile = open(name,'wb')
    pkl.dump(data, outfile)
    outfile.close()
    
    print(name + " dumped")


def load_from_pkl(name):
    '''
        loads pkl data from the .pkl file
        
        params : 
            name -> name of the file

        return :
            X    -> whatever data was inside the pkl file
    '''
    infile = open(name,'rb')
    X = pkl.load(infile)
    infile.close()

    return X