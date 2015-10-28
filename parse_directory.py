from os import walk
from os.path import splitext, join

def select_files(root, files):
    """
    simple logic here to filter out interesting files
    .py files in this example
    """

    selected_files = []

    for file in files:
        #do concatenation here to get full path
        full_path = join(root, file)
        ext = splitext(file)[1]


        selected_files.append(full_path)

    return selected_files

def build_recursive_dir_tree(path):
    """
    path    -    where to begin folder scan
    """
    selected_files = []

    for root, dirs, files in walk(path):
        selected_files += select_files(root, files)

    return selected_files
