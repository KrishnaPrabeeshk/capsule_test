import torch
import torch.nn.functional as F

def train(capsule_net, criterion, optimizer, train_loader, n_epochs):
    '''Trains a capsule network and prints out training loss statistics per epoch.
       param capsule_net: capsule network model
       param criterion: capsule loss function
       param optimizer: optimizer for updating network weights
       param train_loader: DataLoader for the training data
       param n_epochs: number of epochs to train for
       return: list of recorded training losses
    '''

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    capsule_net.to(device)
    criterion.to(device)

    # Track training loss over time
    losses = []

    # One epoch = one pass over all training data
    for epoch in range(1, n_epochs + 1):

        # Initialize training loss
        train_loss = 0.0
        
        capsule_net.train()  # Set to train mode
    
        # Get batches of training image data and targets
        for images, labels in train_loader:

            # Move data to device
            images = images.to(device)    # Shape: [batch_size, 1, 128, 128]
            labels = labels.to(device)    # Shape: [batch_size]

            # Convert labels to one-hot encoding for binary classes
            labels_one_hot = F.one_hot(labels, num_classes=2).float()  # Shape: [batch_size, 2]

            # Zero out gradients
            optimizer.zero_grad()

            # Get model outputs
            caps_output, reconstructions, y = capsule_net(images)

            # Calculate loss
            loss = criterion(caps_output, labels_one_hot, images, reconstructions)

            # Perform backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            train_loss += loss.item()
        
        # Calculate and print average loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        losses.append(avg_train_loss)
        print(f'Epoch: {epoch} \tAverage Training Loss: {avg_train_loss:.8f}')

    return losses
