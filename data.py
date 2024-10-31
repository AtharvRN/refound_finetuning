import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader, random_split,Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class OCTDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Filter out rows with bad quality and non-existent images
        self.data_frame = self.data_frame[
            (self.data_frame['OverallQuality'] != 'Bad') &
            (self.data_frame['Image Name'].apply(lambda x: os.path.exists(os.path.join(self.root_dir, x))))
        ]

        # self.class_counts = self.data_frame.iloc[:, 12:20].sum(axis=0)  # Assuming labels start from column 12
        
        #  # Print class-wise positive example counts
        # print("Class-wise positive example counts:")
        # for label, count in self.class_counts.items():
        #     print(f"\t- {label}: {count}")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx, 9]  # Assuming 'Image Name' is the column containing image names
        image_path = os.path.join(self.root_dir, img_name)
        image = Image.open(image_path)

        # Select the desired columns from the DataFrame
        foveal_scan = self.data_frame.iloc[idx, 12]
        healthy = self.data_frame.iloc[idx, 13]
        srf = self.data_frame.iloc[idx, 14]
        irf = self.data_frame.iloc[idx, 15]
        drusen = self.data_frame.iloc[idx, 16]
        ped = self.data_frame.iloc[idx, 17]
        hdots = self.data_frame.iloc[idx, 18]
        hfoci = self.data_frame.iloc[idx, 19]

        # Convert 'Yes' to 1 and 'No' to 0
        labels = {
            'foveal_scan': 1 if foveal_scan == 'Yes' else 0,
            'healthy': 1 if healthy == 'Yes' else 0,
        
            # 'srf': 1 if srf == 'Yes' else 0,
            # 'irf': 1 if irf == 'Yes' else 0,
            'drusen': 1 if drusen == 'Yes' else 0,
            
            'ped': 0 if ped == 'No PED' else 1,
            'hdots': 1 if hdots == 'Yes' else 0,
            # 'hfoci': 1 if hfoci == 'Yes' else 0,
            

        }

        # Convert labels to torch tensors
        labels = {key: torch.tensor(value) for key, value in labels.items()}

        if self.transform:
            image = self.transform(image)

        return image, labels

    def print_positive_counts(self):
        # Initialize dictionary to store positive counts for each task
        positive_counts = {
            'foveal_scan': 0,
            'healthy': 0,
            # 'srf': 0,
            # 'irf': 0,
            'drusen': 0,
            'ped': 0,
            'hdots': 0,
            # 'hfoci': 0
        }

        # Iterate through the dataset to count positive samples
        for idx in range(len(self)):
            # print(self[idx])
            labels = self[idx][1]  # Assuming labels are returned as the second item
            for task, label in labels.items():
                positive_counts[task] += label.item()

        # Print the positive counts for each task
        print("Number of positive samples in each task across the entire dataset:")
        size = len(self)
        for task, count in positive_counts.items():
            print(f"{task}: {count}/{size}")

def create_data_loaders(train_csv_file, train_root_dir, test_csv_file, test_root_dir, batch_size=32, val_split=0.1, image_size=224, seed=42):
    """
    This function creates and returns data loaders for training, validation, and testing.

    Args:
        train_csv_file (str): Path to the CSV file containing training data annotations.
        train_root_dir (str): Directory containing the training images.
        test_csv_file (str): Path to the CSV file containing testing data annotations.
        test_root_dir (str): Directory containing the testing images.
        batch_size (int, optional): The number of samples in a batch. Defaults to 32.
        val_split (float, optional): Proportion of the training data to be used for validation. Defaults to 0.1 (10%).
        image_size (int, optional): The target size for resizing images. Defaults to 224.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple of three data loaders - train_loader, val_loader, test_loader.
    """

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define data transformations
    print("Defining data transformations...")
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize images to the desired size
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add color jitter for training data
        transforms.Grayscale(num_output_channels=3),  # Convert single-channel images to 3 channels
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize pixel values
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize images to the desired size
        transforms.Grayscale(num_output_channels=3),  # Convert single-channel images to 3 channels
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize pixel values
    ])

    # Create OCTDatasets for training and testing
    train_dataset = OCTDataset(csv_file=train_csv_file, root_dir=train_root_dir, transform=train_transform)
    test_dataset = OCTDataset(csv_file=test_csv_file, root_dir=test_root_dir, transform=test_transform)
    
    train_dataset.print_positive_counts()
    test_dataset.print_positive_counts()
    # Calculate sizes of splits based on validation ratio
    dataset_size = len(train_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    test_size = len(test_dataset)

    # Split the training dataset into training and validation sets
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader




def create_k_fold_data_loaders(csv_file, root_dir, batch_size=32, validation_split=0.1, test_split=0.1, shuffle_dataset=True, seed=42):
    # Define transforms for training and validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images if needed
        transforms.Grayscale(num_output_channels=3),  # Convert single-channel image to 3 channels
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create dataset
    dataset = OCTDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

    # Calculate sizes of splits
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size - test_size

    # Create 10-fold cross-validation data loaders
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    cross_val_loaders = []
    for train_index, test_index in kf.split(dataset):

        train_fold, val_fold = random_split(Subset(dataset, train_index), [train_size, val_size])
        train_loader = DataLoader(train_fold, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_fold, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(Subset(dataset, test_index), batch_size=batch_size, shuffle=False)
        cross_val_loaders.append((train_loader, val_loader, test_loader))

    return cross_val_loaders