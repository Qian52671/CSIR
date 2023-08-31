from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import tensorflow as tf
import pandas as pd


def VGG16_getFeatures(file_list,feature_number,label_type):

    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

    if feature_number == '4096':
        model = tf.keras.Model(inputs = model.inputs, outputs = model.layers[-2].output)

    filenames_list = []
    vgg16_feature_list = []

   # Loop through the file_list and extract VGG16 features
    for file in file_list:
        img = load_img(file, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        vgg16_feature = model.predict(x)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten()) 
        feature_array = np.array(vgg16_feature_list)
        filenames_list.append(file)
    
    filenames_array = np.array(filenames_list)
    vgg16_feature_array = np.array(vgg16_feature_list)
    
    df_O = pd.DataFrame({'filename': filenames_array, 'image_feature': vgg16_feature_array.tolist()})
    df = manage_df(df_O,label_type)
    print(df)
    return df


    

    

def manage_df(df,label_type):
    df['filename'] = df['filename'].str.split('/').str[-1]
    if label_type == 'city':
        df['file_type'] = df['filename'].str.split('_day').str[0]

    elif label_type == 'weekend_weekday':
           df['file_type'] = df['filename'].str.split('_building').str[0]
        
    df['Day'] = df['filename'].str.extract(r'day(\d+)').astype(int)

    file_types = df['file_type'].unique()
    print(file_types)

    def find_file_type(x):
        for file_type in file_types:
            if file_type in x:
               return file_type
        return None
        
    df['Type'] = df['filename'].apply(find_file_type)

        
    df = df.sort_values(by=['Type', 'Day'], ascending=[True, True])
    df = df.drop(columns=['Type', 'Day'])

    def assign_label(file_name):
            for idx, file_type in enumerate(file_types):
                if file_type in file_name:
                 return idx
            return None
        
    df['label_true'] = df['filename'].apply(assign_label)
    df = df.drop(columns=['file_type'])

    return df
    
   








    
        
    


    
    
    




#     model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling='avg',
#                   classes=1000)
#     # model.compile()

#     vgg16_feature_list = []
#     img = load_img(file, target_size=(224, 224))
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)

#     vgg16_feature = model.predict(x)
#     vgg16_feature_np = np.array(vgg16_feature)
#     vgg16_feature_list.append(vgg16_feature_np.flatten())

#     feature_array = np.array(vgg16_feature_list)
#     return feature_array

# def VGG16_get_images_feature(file_list):
#     data = {}
#     for feature in file_list:
#                  feat = getFeatures(feature)
#                  data[feature] = feat
#     return data

#     filenames = np.array(list(feature_data.keys()))


# # 假设data.values()返回的是一个列表，每个元素都是一个一维数组
# data_arrays = feature_data.values()

# # 将数组列表转换为一个NumPy数组
# data_matrix = np.vstack(data_arrays)




# model = VGG16()
# model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

# def extract_features(file, model):
#     # load the image as a 224x224 array
#     img = load_img(file, target_size=(224,224))
#     # convert from 'PIL.Image.Image' to numpy array
#     img = np.array(img) 
#     # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
#     reshaped_img = img.reshape(1,224,224,3) 
#     # prepare image for model
#     imgx = preprocess_input(reshaped_img)
#     # get the feature vector
#     features = model.predict(imgx, use_multiprocessing=True)
#     return features

# data = {}

# for GASF_feature in GASF_feature_list:
#                  feat = extract_features(GASF_feature,model)
#                  data[GASF_feature] = feat


# # get a list of the filenames
# filenames = np.array(list(data.keys()))

# # get a list of just the features
# feat = np.array(list(data.values()))

# # reshape so that there are 210 samples of 4096 vectors
# feat = feat.reshape(-1,4096)
    
        



