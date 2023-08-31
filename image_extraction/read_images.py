import os


# Function to get all PNG files from a given directory and append them to GASF_feature_list
def append_png_files(directory):
    file_list = []

    with os.scandir(directory) as files:
        for file in files:
            if file.name.endswith('.png') and file.is_file():
                file_list.append(os.path.join(directory, file.name))
        return file_list




def get_images_file_list(path1,path2 = False):

    if path2:
        file_list1 = append_png_files(path1)
        file_list2 = append_png_files(path2)
        return file_list1 + file_list2

    else:
       return append_png_files(path1)










