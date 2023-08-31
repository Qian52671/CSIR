import pywt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import pandas as pd
from image_extraction.feature_extraction import *
import pandas as pd
import re

def custom_sort_key(filename):
    match = re.match(r'building_(\d+)_day_(\d+)', filename)
    if match:
        building_number = int(match.group(1))
        day_number = int(match.group(2))
        return building_number, day_number
    return (0, 0)


image_dir = "/Users/liuqianyi/Documents/Customer Segmentation Clustering Based on Image Representation for Energy Consumption/heatmap_generation/15_building_184days_plots"

def enhance_heatmaps(image_dir):
    png_files = glob.glob(os.path.join(image_dir, "*.png"))
    transformed_images_list = []
    for i in range(len(png_files)):
        file = png_files[i]
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (112, 112))
        transformed_image = np.empty_like(image)

        for channel in range(3):
            coeffs = pywt.wavedec2(image[:, :, channel], 'haar', level=2)
            for i in range(len(coeffs)):
                if i == 0:
                    coeffs[i][coeffs[i] > 250] *= 1.25
                    coeffs[i][coeffs[i] < 250] *= 1.25
                else:
                    LH, HL, HH = coeffs[i]
                    LH[LH < 150] *= 0
                    HL[HL < 150] *= 0
                    HH[HH < 150] *= 0
                    LH[LH > 150] *= 1.25
                    HL[HL > 150] *= 1.25
                    HH[HH > 150] *= 1.25
                    coeffs[i] = LH, HL, HH
            transformed_image[:, :, channel] = pywt.waverec2(coeffs, 'haar')
        transformed_images_list.append(transformed_image.astype(int))

    df = pd.DataFrame({'filename': png_files})
    df['image_feature'] = transformed_images_list
    df['filename'] = df['filename'].str.split('/').str[-1]
    df['filename'] = sorted(df['filename'], key=custom_sort_key)
    feature = df['image_feature'].values
    feature = np.array(feature.tolist())
    x = np.divide(feature, 255.)
    print(x.shape)

    return x, df


