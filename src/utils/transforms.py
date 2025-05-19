from torchvision import transforms


def get_train_transforms(image_size=[224, 224]):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=(-10, 10), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def get_val_test_transforms(image_size=[224, 224]):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
