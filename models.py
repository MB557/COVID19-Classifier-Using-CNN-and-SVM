# Keras and Tensorflow
import keras

from tensorflow import keras, nn
from tensorflow.keras.applications import VGG16, EfficientNetB3, ResNet50

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import LeakyReLU, Input, Dense, BatchNormalization
from tensorflow.keras.models import Model, load_model

'''VGG model'''
class VGG(keras.Model):
    def __init__(self):
        super().__init__()

        # creating the VGG model
        vgg_conv = VGG16(weights='imagenet', input_shape=(224,224,3))
        self.VGG_model = Sequential()
        for layer in vgg_conv.layers[:-1]: # excluding last layer from copying
            self.VGG_model.add(layer)
                
    def freeze(self):
        self.VGG_model.trainable = False

    def predict(self, images):
        features = self.VGG_model.predict(images, verbose=1)

        return features



'''EfficientNet B3 model'''
class EffNetB3(keras.Model):
    def __init__(self):
        super().__init__()

        # creating the Inception V3 model
        self.EfficientNetB3_model = EfficientNetB3(weights='imagenet',
                                include_top = False, 
                                input_shape=(300, 300, 3),
                                pooling = 'avg')
                
    def freeze(self):
        self.EfficientNetB3_model.trainable = False

    def predict(self, images):
        features = self.EfficientNetB3_model.predict(images, verbose=1)

        return features



'''ResNet50 Model'''
class ResNet(keras.Model):
    def __init__(self):
        super().__init__()

        # creating the Inception V3 model
        self.ResNet_model = ResNet50(weights='imagenet', 
                                    input_shape=(512,512,3), 
                                    include_top = False, 
                                    pooling = 'avg')
                
    def freeze(self):
        self.ResNet_model.trainable = False

    def predict(self, images):
        features = self.ResNet_model.predict(images, verbose=1)

        return features