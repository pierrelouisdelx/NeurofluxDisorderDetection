from torch.utils.data import Dataset
from PIL import Image

class NeurofluxDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            raise
        except Exception as e:
            print(f"Error opening or converting image {img_path}: {e}")
            raise

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label