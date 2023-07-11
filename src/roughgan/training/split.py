from torch.utils.data import DataLoader, random_split


def train_test_dataloaders(dataset, train_ratio=0.8, **kwargs):
    """Construct the training and testing `DataLoader`s."""
    train_dataset, test_dataset = train_test_split(dataset, train_ratio=train_ratio)

    train_dataloader = DataLoader(train_dataset, **kwargs)
    test_dataloader = DataLoader(test_dataset, **kwargs)

    return train_dataloader, test_dataloader


def train_test_split(dataset, train_ratio=0.8):
    """Split the original dataset into training and testing subsets."""
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset
