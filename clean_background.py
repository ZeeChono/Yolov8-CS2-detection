# This file removes all the unannotated labels out
import os
import shutil

def filter_non_empty_txt_files(directory):
    # Get the list of text files in the directory
    txt_files = [file for file in os.listdir(directory) if file.endswith('.txt')]

    # Filter out files without any content
    non_empty_txt_files = []
    for file_name in txt_files:
        file_path = os.path.join(directory, file_name)
        if os.path.getsize(file_path) > 0:
            non_empty_txt_files.append(file_name)

    return non_empty_txt_files



if __name__ == '__main__':
    # Clean the data
    directory_path = 'obj_train_data/'
    non_empty_files = filter_non_empty_txt_files(directory_path)
    print("Non-empty text files:", len(non_empty_files))
    print(non_empty_files[0])


    output_directory_path = 'non_empty_file'
    for file in non_empty_files:
        shutil.copy(directory_path + "\\" + file, output_directory_path + "\\" + file)