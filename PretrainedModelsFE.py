import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet101V2, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.applications import ResNet50V2, ResNet152V2, MobileNetV3Large, MobileNetV3Small
from getdata import get_image_data
import pickle
import numpy as np
from config import config

data, cls = get_image_data()
print(len(cls))
cls = np.reshape(np.array(cls), (-1, 1))
print(cls.shape)
baseModelsResNet50V2 = ResNet50V2(weights="imagenet", include_top=False,input_shape = (config.image_size[0], config.image_size[1] ,3))
baseModelsResNet152V2 = ResNet152V2(weights="imagenet", include_top=False,input_shape = (config.image_size[0], config.image_size[1] ,3))
baseModelsMNV3L = MobileNetV3Large(weights="imagenet", include_top=False,input_shape = (config.image_size[0], config.image_size[1] ,3))
baseModelsMNV3S = MobileNetV3Small(weights="imagenet", include_top=False,input_shape = (config.image_size[0], config.image_size[1] ,3))

baseModels = [baseModelsResNet50V2, baseModelsResNet152V2, baseModelsMNV3L, baseModelsMNV3S]
modelNames = ['MNV3S.pickle', 'MNV3L.pickle', 'ResNet152V2.pickle', 'ResNet50V2.pickle']

for baseModel in baseModels:
    for layer in baseModel.layers:
        layer.trainable = False

    # creating the output layers that merges with the ResNet101v2
    x = baseModel.output
    x = tf.keras.layers.Flatten()(x)
    # merging the 2 model together as one
    model = Model(inputs= baseModel.input , outputs = x)

    features = model.predict(data)
    features = np.append(features, cls, axis=1)
    print(features.shape)

    fileName = modelNames.pop()
    print(fileName)
    fp = open(fileName, "wb")
    pickle.dump(features, fp)
    fp.close()