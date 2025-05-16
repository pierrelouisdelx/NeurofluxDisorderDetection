from torch.utils.data import Dataset
from PIL import Image

class NeurofluxDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, class_names=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or ["PTE", "LO", "EO", "IPTE", "IO"]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("L")
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            raise
        except Exception as e:
            print(f"Error opening or converting image {img_path}: {e}")
            raise

        label = self.label_to_idx[self.labels[idx]]

        if self.transform:
            image = self.transform(image)
        
        return image, label