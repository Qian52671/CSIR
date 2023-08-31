import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from preprocess.generate_time_series import*

import seaborn as sns


house_list = generate_all_1day_15min_list('15min_newyork.h5')

def create_heatmap(house_list):
    for i in range(len(house_list)):
        all_list = house_list[i] 
        for j in range(len(all_list)):
            building = all_list[j]
            hourly_data_array = np.vstack(building)
            hourly_data_zscore = zscore(hourly_data_array)
            main_2d = hourly_data_zscore.reshape(-1, 96)
            reshaped_data = main_2d.reshape(-1, 4)
            plt.figure()
            ax = sns.heatmap(reshaped_data, cmap="jet", cbar=False)
            ax.axis('off')
            plt.savefig(f"./15_building_184days_plots/building_{i+1}_day_{j+1}_heatmap.png", bbox_inches='tight', pad_inches=0)
            plt.close()

    
    

