'''
    Project Title:      COVID-19 Classification on X-ray images using Multimodal CNN's
    
    Group Members:      Milan Ashvinbhai Bhuva  -   IIT2018176
                        Manav Kamlesh Agrawal   -   IIT2018178
                        Mohammed Aadil          -   IIT2018179
'''

#   --------------------------------- Importing Header Files ----------------------------------------

import os
import cv2
import numpy as np
import pickle
import io

# GUI
import PySimpleGUI as sg
from PIL import Image

# Sklearn
from sklearn.decomposition import PCA
from sklearn import svm

# tf and keras
from tensorflow.keras.applications import VGG16, EfficientNetB3, ResNet50
from keras.models import Sequential
from tensorflow.keras.models import Model
from keras.models import load_model


labels = [0, 1]         # 0 --> Non-COVID, 1 --> COVID.


'''---------------------------------------- PRE-TRAINED MODELS ----------------------------------------'''


# Three pre-trained models: VGG16, ResNet50, EffNet B3.

def pass_through_VGG(img):
    vgg_features = vgg16.predict(img)
    PCA_features = vgg_pca.transform(vgg_features)
    probs = vgg_svm.predict_proba(PCA_features)
    
    return probs


def pass_through_ResNet(img):
    resnet_features = resnet.predict(img)
    PCA_features = resnet_pca.transform(resnet_features)
    probs = resnet_svm.predict_proba(PCA_features)
    
    return probs


def pass_through_EffNet(img):
    effnet_features = effnet.predict(img)
    PCA_features = effnet_pca.transform(effnet_features)
    probs = effnet_svm.predict_proba(PCA_features)
    
    return probs



# Loading VGG16, ResNet50 and EffecientNetB3 pretrained models
vgg16 = load_model('H5/vgg_model.h5', compile=False)
resnet = load_model('H5/resnet_model.h5', compile=False)
effnet = load_model('H5/effnet_model.h5', compile=False)

# Loading PCA models for VGG16, ResNet50 and EffecientNetB3 [PCA is used for dimensionality reduction.]
vgg_pca = pickle.load(open('models/vgg_pca.pkl', 'rb'))
resnet_pca = pickle.load(open('models/resnet_pca.pkl', 'rb'))
effnet_pca = pickle.load(open('models/effnet_pca.pkl', 'rb'))

# Loading SVM models for VGG16, ResNet50 and EffecientNetB3 [SVM for classification.]
vgg_svm = pickle.load(open('models/vgg_svm.pkl', 'rb'))
resnet_svm = pickle.load(open('models/resnet_svm.pkl', 'rb'))
effnet_svm = pickle.load(open('models/effnet_svm.pkl', 'rb'))


# ----------------------------------------- GUI ---------------------------------------------------

# image_viewer.py

sg.theme('Light Blue')

file_types = [("All files (*.*)", "*.*")]
def main():

    # Left column of the GUI will contain filechooser and image-viewer.

    left_col = [
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
        ],
        [sg.Image(key="-IMAGE-")],
    ]

    def predict(imagepath):
        # importing the test image
        img = cv2.imread(imagepath)

        # resize and reshape for VGG16, ResNet50 and EffecientNetB3
        vgg16_img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
        vgg16_img = vgg16_img.reshape(1,224, 224,3)
        resnet50_img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
        resnet50_img = resnet50_img.reshape(1,512,512,3)
        effnetb3_img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
        effnetb3_img = effnetb3_img.reshape(1,300,300,3)

        # passing through the 3 layres of each pretrained model
        vgg_probs = pass_through_VGG(vgg16_img)
        resnet_probs = pass_through_ResNet(resnet50_img)
        effnet_probs = pass_through_EffNet(effnetb3_img)

        # fusion using voting
        merged = (effnet_probs + resnet_probs + vgg_probs)/3
        merged_preds = np.argmax(merged, axis=1)

        data = make_table(merged)

        # printing the class
        return data

    # Make the table (3 x 3).

    def make_table(merged):
        data = [[j for j in range(3)] for i in range(3)]
        data[1] = ["NON-COVID", '%.3f' % (merged[0][0]*100)]
        data[2] = ['COVID', '%.3f' % (merged[0][1]*100)]

        return data


    layout = [[sg.Column(left_col, justification='c'),]]

    window = sg.Window("Image Viewer", layout)
    while True:

        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "Load Image":
            filename = values["-FILE-"]

            if os.path.exists(filename):
                print(filename)
                image = Image.open(values["-FILE-"])
                image.thumbnail((600, 600))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())

                result = predict(filename)

                # ------ Make the Table Data ------
                headings = ['    Label    ', 'Accuracy (%)']

                tableau = [[sg.Table(values=result[1:][:], headings=headings, max_col_width=30,
                                background_color='lightblue',
                                auto_size_columns=True,
                                justification='center',
                                num_rows=3,
                                hide_vertical_scroll=True,
                                alternating_row_color='lightblue',
                                key='-TABLE-',
                                row_height=40,)]]

                layout2 = [[sg.Column(tableau, key='_UN_'),]]

                window2 = sg.Window("Table", layout2)

                while True:
                    event2, values2 = window2.read()
                    if event2 == "Exit" or event2 == sg.WIN_CLOSED:
                        break
                window2.close()
    window.close()

if __name__ == "__main__":
    main()