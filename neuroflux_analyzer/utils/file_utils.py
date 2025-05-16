import os
from glob import glob

def get_label(image_path):
    return os.path.basename(os.path.dirname(image_path))

def get_images_and_labels(data_dir):
    image_paths = []
    labels = []

    all_image_paths = glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True)
    for image_path in all_image_paths:
        label = get_label(image_path)
        image_paths.append(image_path)
        labels.append(label)

    return image_paths, labels

