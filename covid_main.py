'''
    Project Title:      Application Of Deep Learning For Indoor Scenes Understanding By Robot
    
    Group Members:      Kislay Kishore          -   IIT2018079
                        Milan Ashvinbhai Bhuva  -   IIT2018176
                        Manav Kamlesh Agrawal   -   IIT2018178
                        Mohammed Aadil          -   IIT2018179
                        Ankit Rauniyar          -   IIT2018202
'''
#   ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   ------------------------------------------------------------------- Importing Header Files --------------------------------------------------------------------------

print("Importing headers...")

# basic ML
import io
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

# GUI
import PySimpleGUI as sg
from PIL import Image

# Sklearn
from sklearn.decomposition import PCA
from sklearn import svm

os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # simply supress the warnings

# tf and keras
from tensorflow.keras.applications import VGG16, EfficientNetB3, ResNet50
from keras.models import Sequential
from tensorflow.keras.models import Model
from keras.models import load_model

# suppress warnings
import warnings
warnings.filterwarnings("ignore")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------- PRE-TRAINED MODELS ----------------------------------------------------------------------------

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

print("Loading Hybrids...")

# loading VGG16, ResNet50 and EffecientNetB3 pretrained models
vgg16 = load_model('H5/vgg_model.h5', compile=False)
resnet = load_model('H5/resnet_model.h5', compile=False)
effnet = load_model('H5/effnet_model.h5', compile=False)

print("Loading PCA...")

# loading PCA models for VGG16, ResNet50 and EffecientNetB3
vgg_pca = pickle.load(open('models/vgg_pca.pkl', 'rb'))
resnet_pca = pickle.load(open('models/resnet_pca.pkl', 'rb'))
effnet_pca = pickle.load(open('models/effnet_pca.pkl', 'rb'))

print("Loading SVM...")

# loading SVM models for VGG16, ResNet50 and EffecientNetB3
vgg_svm = pickle.load(open('models/vgg_svm.pkl', 'rb'))
resnet_svm = pickle.load(open('models/resnet_svm.pkl', 'rb'))
effnet_svm = pickle.load(open('models/effnet_svm.pkl', 'rb'))


labels = [0, 1]   #define labels



#   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#   ------------------------------------------------------------------------------- GUI -----------------------------------------------------------------------------------

# define theme of GUI.

sg.theme('Dark Grey 3')

# file extensions that can be allowed.

file_types = [("All files (*.*)", "*.*")]

# declare "Image", "Video", and "Exit" buttons for main layout.

main_left_col = [[sg.Button("Image", size = (25, 3), pad = (10, 30)),]]
#main_right_col = [[sg.Button("Video", size = (25, 3), pad = (10, 30)),]]
main_horiz_row = [[sg.Button("Exit", size = (50, 2), pad = (20, 35)),]]


#   -------------------------------------------------------------------------- Main Layout ------------------------------------------------------------------------------

# defining the main layout.

layout = [
    [
        sg.Column(main_left_col, justification='c'),
        #sg.VSeparator(),
        #sg.Column(main_right_col, justification = 'c'),
    ],
    [
        sg.HSeparator(),
    ],
    [
        sg.Column(main_horiz_row, justification = 'c'),
    ],
    ]

# making the window.

main_window = sg.Window("COVID-19 Classifier", layout, size=(500, 250))

while True:

    main_event, main_values = main_window.read()

    # exit condition.

    if main_event == "Exit" or main_event == sg.WIN_CLOSED:
        break


    #   ----------------------------------------------------------------------- Image Input ---------------------------------------------------------------------------


    if main_event == "Image":   # i.e. if Image button is clicked.

        # define left column.

        image_left_col = [
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
        ],
        
        [
            sg.Image(key="-IMAGE-")
        ],
        ]

        image_viewer_column = [
        [sg.Text("Choose an image from list on left:")],
        ]

        # ------ Make the Table Data ------

        data1 = [['' for row in range(2)]for col in range(3)]
        headings = ['    Label    ', 'Accuracy (%)']

        image_tableau = [[sg.Table(values=data1[1:][:], headings=headings, max_col_width=25,
                background_color='grey',
                auto_size_columns=True,
                justification='center',
                text_color = 'black',
                num_rows=2,
                hide_vertical_scroll=True,
                alternating_row_color='snow3',
                key='-TABLE-',
                row_height=40,)]]

        # ----------------- Browsing and Loading Images -------------------

        image_horiz_row = [[sg.Button("Exit"),]]

        layout = [[
            sg.Column(image_left_col, justification='c'),
            sg.VSeparator(),
            sg.Column(image_tableau, key='_UN_'),
        ],
        [
            sg.HSeparator(),
        ],
        [
            sg.Column(image_horiz_row, justification = 'c'),
        ]
        ]

        window = sg.Window("Image Viewer", layout)
        while True:

            event, values = window.read()

            # Exit condition.
            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            # If "load image" button is clicked.
            if event == "Load Image":
                filename = values["-FILE-"]

                # Select image and get file path.
                if os.path.exists(filename):
                    image = Image.open(values["-FILE-"])
                    image.thumbnail((700, 700))
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    window["-IMAGE-"].update(data=bio.getvalue())

                # Results are tabulated.
                def make_table(merged):
                    data = [[j for j in range(2)] for i in range(3)]
                    data[1] = ["NON-COVID", '%.3f' % (merged[0][0]*100)]
                    data[2] = ['COVID', '%.3f' % (merged[0][1]*100)]

                    return data



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

                    # passing through the 3 layers of each pretrained model
                    vgg_probs = pass_through_VGG(vgg16_img)
                    resnet_probs = pass_through_ResNet(resnet50_img)
                    effnet_probs = pass_through_EffNet(effnetb3_img)


                    # fusion using voting
                    merged = (effnet_probs + resnet_probs + vgg_probs)/3
                    #merged_preds = np.argmax(merged, axis=1)
                    #print(merged)
                    data = make_table(merged)

                    # printing the class
                    return data

                data = predict(filename)

                # Update the table once the data is received.
                window.FindElement('-TABLE-').update(data[1:][:])

        window.close()
    
# Close main window, terminate execution.
main_window.close()
