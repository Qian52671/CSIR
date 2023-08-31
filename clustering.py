import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from proper_k import*
from enhance_heatmaps import *
from datetime import datetime
from preprocess.generate_time_series import*


import re

n_clusters =15
show_cluster = 13

def kmean_clustering(n_clusters, df, d_x, image_folder, show_cluster, if_show):
    
    kmeans = KMeans(n_clusters = n_clusters,random_state=24)
    kmeans.fit(d_x)
    pred = kmeans.predict(d_x)
    frame = pd.DataFrame(d_x)
    frame['cluster'] = pred
    frame['cluster'].value_counts()

    extracted_filenames = df['filename']
    new_data = {'filename': extracted_filenames}
    new_df = pd.DataFrame(new_data)
    filename_to_label = dict(zip(new_df['filename'], pred))

    for index, row in new_df.iterrows():
        filename = row['filename']
        if filename in filename_to_label:
            new_df.at[index, 'cluster_label'] = filename_to_label[filename]

    selected_rows = new_df[new_df['cluster_label'] == show_cluster]

    if if_show == 'yes':
        for filename in selected_rows['filename']:
            image_path = os.path.join(image_folder, filename)
            if os.path.exists(image_path):
                img = imread(image_path)
                plt.imshow(img)
                plt.title(filename)
                plt.show()
            else:
                print(f"Image not found for filename: {filename}")

    return new_df,selected_rows




def extract_building_id(input_string):
    match = re.search(r'\d+', input_string)
    if match:
        return int(match.group())
    else:
        return None
    
def extract_day_id(input_string):
    match = re.search(r'day_(\d+)', input_string)
    if match:
        extracted_number = int(match.group(1))
        return extracted_number
    else:
        return None   

def assign_month(day):
    if 1 <= day <= 31:
        return 5
    elif 32 <= day <= 62:
        return 6
    elif 63 <= day <= 94:
        return 7
    elif 95 <= day <= 126:
        return 8
    elif 127 <= day <= 157:
        return 9
    elif 158 <= day <= 184:
        return 10  
    else:
        return None
    
def assign_warm_cold(month):
    if month in [5, 9, 10]:
        return 0
    else:
        return 1
    


def cluster_check_avg(new_df,show_cluster):
    new_df['building_id'] = new_df['filename'].apply(extract_building_id)
    new_df['day'] = new_df['filename'].apply(extract_day_id)
    new_df['month'] = new_df['day'].apply(assign_month)
    new_df['season'] = new_df['month'].apply(assign_warm_cold)

    selected_rows = new_df[new_df['cluster_label'] == show_cluster]
    print(selected_rows)

    building_id_counts = selected_rows['building_id'].value_counts()
    building_id_counts_sorted = building_id_counts.sort_index()
    # for building_id, count in building_id_counts_sorted.items():
        # print(f"Building ID: {building_id}, Count: {count}")

    building_ids = []
    days = []

    for value in selected_rows['filename']:
        print(value)
        match = re.search(r'\d+', str(value))
        if match:
            building_id = int(match.group())
            building_ids.append(building_id)

        day_match = re.search(r'day_(\d+)', value)
        if day_match:
            day_number = int(day_match.group(1))
            days.append(day_number)

        
    daily_total_values = {}
    monthly_totals = {}

    for building_id, day in zip(building_ids,days):
        all_buildings_list = generate_all_1day_15min_list('15min_newyork.h5')
        data_list = all_buildings_list[building_id-1][day-1]
        for date_str, value in zip(data_list.index.strftime('%Y-%m-%d'), data_list.values):
            if pd.notna(value):  # This checks if 'value' is not NaN
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
                if date in daily_total_values:
                    daily_total_values[date] += value
                else:
                    daily_total_values[date] = value


    for date, total_value in daily_total_values.items():
        month = date.month
        if month not in monthly_totals:
            monthly_totals[month] = {
            'total': 0,
            'count': 0
        }
            
        monthly_totals[month]['total'] += total_value
        monthly_totals[month]['count'] += 1

    
    monthly_averages = {}
    for month, data in monthly_totals.items():
        total = data['total']
        count = data['count']
        average = total / count
        if month not in monthly_averages:
            monthly_averages[month] = []
        monthly_averages[month].append(average)



    six_daily_total_values = {}

    for building_id, day in zip(building_ids,days):
         data_list = all_buildings_list[building_id-1][day-1]
         for date_str, value in zip(data_list.index.strftime('%Y-%m-%d %H:%M:%S'), data_list.values):
             if pd.notna(value):  # This checks if 'value' is not NaN
                 date_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                 time_group = f"{date_time.hour // 6 * 6:02d}:00:00 - {(date_time.hour // 6 + 1) * 6 - 1:02d}:59:59"
                 date_time_group = f"{date_time.date()} {time_group}"
                 if date_time_group not in six_daily_total_values:
                     six_daily_total_values[date_time_group] = 0
                     six_daily_total_values[date_time_group] += value

    time_group_totals = {}
    for date_time_group, total_value in six_daily_total_values.items():
            time_group = date_time_group.split(' - ')[1]
            if time_group not in time_group_totals:
                        time_group_totals[time_group] = {
                            'total': 0,
                            'count': 0
                            }
            time_group_totals[time_group]['total'] += total_value
            time_group_totals[time_group]['count'] += 1
    # print(time_group_totals) 

    time_group_averages = {}
    for time_group, data in time_group_totals.items():
           total = data['total']
           count = data['count']
           average = total / count
           if time_group not in time_group_averages:
                time_group_averages[time_group] = []
           time_group_averages[time_group].append(average)

    for time_group, averages in time_group_averages.items():
        print(f"period:{time_group}, average power:{sum(averages) / len(averages)}")

    