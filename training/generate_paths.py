from glob import glob

def generate_paths_for_subset(subset_path, txt_filepath):
    image_paths = list(glob(subset_path))
    with open(txt_filepath, 'w') as file:
        for path in image_paths:
            file.write(path + '\n')
    file.close()

def generate_all_paths():
    subsets = ['train\\*', 'test\\*', 'val\\*']
    txt_filenames = ['train.txt', 'test.txt', 'val.txt']
    image_dirpath = 'datasets\\trafficSigns\\images\\'
    txt_dirpath = 'datasets\\trafficSigns\\'

    for subset, txt_file in zip(subsets, txt_filenames):
        generate_paths_for_subset(image_dirpath + subset, txt_dirpath + txt_file)