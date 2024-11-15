from torchvision import datasets
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from CapsuleNetwork import CapsuleNetwork
from CapsuleLoss import CapsuleLoss
from train import train


# Number of subprocesses to use for data loading
num_workers = 0
# How many samples per batch to load
batch_size = 32

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()])

# Paths to training and testing data
train_dir = 'cat_and_dog_dataset/training_set/training_set'
test_dir = 'cat_and_dog_dataset/test_set/test_set'

# Load the datasets
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)

# Verify class mapping
print(train_data.classes)       # ['cat', 'dog']
print(train_data.class_to_idx)  # {'cat': 0, 'dog': 1}

# ------------------------ #
#  Adjust Training Data    #
# ------------------------ #

# Number of samples per class for training
num_train_samples_per_class = 250

# Get all indices for each class in the training data
train_targets = np.array(train_data.targets)
train_class0_indices = np.where(train_targets == 0)[0]  # Indices of class 'cat'
train_class1_indices = np.where(train_targets == 1)[0]  # Indices of class 'dog'

# Shuffle indices
np.random.shuffle(train_class0_indices)
np.random.shuffle(train_class1_indices)

# Check if there are enough samples
if len(train_class0_indices) < num_train_samples_per_class or len(train_class1_indices) < num_train_samples_per_class:
    raise ValueError("Not enough samples in one of the classes in the training data.")

# Select the first 500 samples from each class
train_class0_samples = train_class0_indices[:num_train_samples_per_class]
train_class1_samples = train_class1_indices[:num_train_samples_per_class]

# Combine and shuffle the selected indices
train_indices = np.concatenate((train_class0_samples, train_class1_samples))
np.random.shuffle(train_indices)

# Create a sampler using the selected indices
train_sampler = SubsetRandomSampler(train_indices)

# Prepare the training data loader
train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=batch_size, 
                                           sampler=train_sampler,
                                           num_workers=num_workers)

# ------------------------ #
#    Adjust Testing Data   #
# ------------------------ #

# Number of samples per class for testing
num_test_samples_per_class = 50  # Adjusted to get 100 samples total

# Get all indices for each class in the testing data
test_targets = np.array(test_data.targets)
test_class0_indices = np.where(test_targets == 0)[0]  # Indices of class 'cat'
test_class1_indices = np.where(test_targets == 1)[0]  # Indices of class 'dog'

# Shuffle indices
np.random.shuffle(test_class0_indices)
np.random.shuffle(test_class1_indices)

# Check if there are enough samples
if len(test_class0_indices) < num_test_samples_per_class or len(test_class1_indices) < num_test_samples_per_class:
    raise ValueError("Not enough samples in one of the classes in the test data.")

# Select the first 50 samples from each class
test_class0_samples = test_class0_indices[:num_test_samples_per_class]
test_class1_samples = test_class1_indices[:num_test_samples_per_class]

# Combine and shuffle the selected indices
test_indices = np.concatenate((test_class0_samples, test_class1_samples))
np.random.shuffle(test_indices)

# Create a sampler using the selected indices
test_sampler = SubsetRandomSampler(test_indices)

# Prepare the testing data loader
test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=batch_size, 
                                          sampler=test_sampler,
                                          num_workers=num_workers)
print('Number of training samples:', len(train_indices))
print('Number of testing samples:', len(test_indices))


capsule_net = CapsuleNetwork()
print(capsule_net)


# custom loss
criterion = CapsuleLoss()

# Adam optimizer with default params
optimizer = optim.Adam(capsule_net.parameters(),lr=0.1)


# Move model to GPU, if available
if torch.cuda.is_available():
    capsule_net = capsule_net.cuda()

# Training
n_epochs = 10
losses = train(capsule_net, criterion, optimizer, train_loader, n_epochs=n_epochs)
print(losses)

