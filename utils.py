import os

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
