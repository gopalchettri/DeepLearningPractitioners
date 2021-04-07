# contains logic to import data dynamically
import os
import os.path
import glob
import shutil

parent_directory = "1. DataAugmentation/dataset/flowers17/image"
sub_directory = "1. DataAugmentation/dataset/flowers17/image/"
files_path = "1. DataAugmentation/dataset/17flowers/jpg/"

def create_dir(dir_path):
    if(os.path.exists(dir_path) == False):
        # directory not found.create the directory
         os.makedirs(dir_path)
         print("[INFO] ", dir_path , " created.")

def add_file(dir_path, file_path, sub_directory_count=18):
    # create sub directory
    if(len(dir_path)!=0):
        if(sub_directory_count != 0):
            # create subdirctory
            for i in range(1,sub_directory_count):
                # print("sub directory path : ", dir_path + str(i))
                sub_dir_path = dir_path + str(i)
                create_dir(sub_dir_path)

                # move file - logic
                file_count = len(list(glob.iglob(file_path + '/**/*.jpg', recursive=True)))
                image_list = [f for f in glob.glob(file_path + "*.jpg")]
                for i, image_path in enumerate(image_list):
                    # move image from one location to another location
                    if i == 80: # i need 80 pictures per folder so this condition
                        break
                    else:
                        if(str(os.path.exists(image_path))):
                            print(image_path)
                            dest = shutil.move(image_path, sub_dir_path)
                        # else:
                        #     print("file already present")
                        # print("info destination path : ", dest)              


def main():
    create_dir(parent_directory)
    # file_path = list(paths.list_images(files_path))
    add_file(sub_directory, files_path, sub_directory_count=18)

if __name__ == '__main__':
    main()