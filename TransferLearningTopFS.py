from tensorflow.keras.applications import EfficientNetB1, EfficientNetB2, \
    EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

from Decompose import TransferLearningFeatureSelection
from config import config

#result_file = 'Transfer Learning Results Covid_pne_normal.csv'
image_data_pickle_file = "Corona_Pne_normal_image_data.pickle"
image_cls_pickle_file = "Corona_Pne_normal_class_data.pickle"


baseModelsEfficientNetB1 = EfficientNetB1(weights="imagenet", include_top=False,input_shape = (config.image_size[0], config.image_size[1] ,3))
baseModelsEfficientNetB2 = EfficientNetB2(weights="imagenet", include_top=False,input_shape = (config.image_size[0], config.image_size[1] ,3))
baseModelsEfficientNetB3 = EfficientNetB3(weights="imagenet", include_top=False,input_shape = (config.image_size[0], config.image_size[1] ,3))
baseModelsEfficientNetB4 = EfficientNetB4(weights="imagenet", include_top=False,input_shape = (config.image_size[0], config.image_size[1] ,3))
baseModelsEfficientNetB5 = EfficientNetB5(weights="imagenet", include_top=False,input_shape = (config.image_size[0], config.image_size[1] ,3))
baseModelsEfficientNetB6 = EfficientNetB6(weights="imagenet", include_top=False,input_shape = (config.image_size[0], config.image_size[1] ,3))
baseModelsEfficientNetB7 = EfficientNetB7(weights="imagenet", include_top=False,input_shape = (config.image_size[0], config.image_size[1] ,3))

baseModels = [baseModelsEfficientNetB1, baseModelsEfficientNetB2, baseModelsEfficientNetB3, baseModelsEfficientNetB4,
              baseModelsEfficientNetB5, baseModelsEfficientNetB6, baseModelsEfficientNetB7]
modelNames = ['EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']

for i in range(len(baseModels)):
    model_name = modelNames[i]
    base_model = baseModels[i]
    fs = TransferLearningFeatureSelection(model_name, base_model, image_data_pickle_file, image_cls_pickle_file)
    fs.create_model_features_pickle()
    fs.create_top_model_features_pickle()
    print('Done')