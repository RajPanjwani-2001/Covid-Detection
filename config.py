class config:
    folderPath = 'D:/Research/Medical/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset/'
    metaDataFile = 'metadata.csv'
    image_size = (128, 128)  # Width, Height
    folders = ['covid/', 'normal/', 'pneumonia/']
    cls_dict = {folders[0]: 0, folders[1]: 1, folders[2]: 2}
