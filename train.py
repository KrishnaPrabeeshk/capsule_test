import os
import torch
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt

def train(capsule_net, decoder_loss_fn, optimizer, train_loader, n_epochs, recon_loss_weight=1):
    """
    Trains a capsule network, saves the model, and tests on training samples.
    :param capsule_net: Capsule Network model.
    :param capsule_loss_fn: Loss function for the capsule layer (e.g., margin loss).
    :param reconstruction_loss_fn: Loss function for the decoder (e.g., MSE).
    :param optimizer: Optimizer for training.
    :param train_loader: DataLoader for training data.
    :param n_epochs: Number of epochs to train for.
    :param recon_loss_weight: Weight for reconstruction loss in the total loss.
    :return: List of training losses and the path to the saved model.
    """
    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training started on : ", device)
    capsule_net.to(device.type)

    # Track training loss over time
    losses = []
    train_samples = []  # To store images for testing later

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        capsule_net.train()  # Set to train mode

        for images, _ in train_loader:
            # Store images for visualization (only for the first epoch)
            if epoch == 1:
                train_samples.append(images.cpu().clone())

            # Move images to the device
            images = images.to(device)

            # Zero out gradients
            optimizer.zero_grad()

            # Get model outputs
            digit_caps_output, reconstructions, _ = capsule_net(images)

            # Compute capsule loss
            # capsule_loss = capsule_loss_fn(digit_caps_output)

            # Compute reconstruction loss
            reconstruction_loss = decoder_loss_fn(reconstructions, images)

            # Combine losses with the reconstruction weight
            total_loss = recon_loss_weight * reconstruction_loss

            # Perform backpropagation and optimization
            total_loss.backward()
            optimizer.step()

            # Accumulate training loss
            train_loss += total_loss.item()

        # Average loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        losses.append(avg_train_loss)
        print(f"Epoch {epoch}/{n_epochs}, Average Loss: {avg_train_loss:.8f}")

    # Save the model with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join('models', f'capsule_net_{timestamp}.pth')
    os.makedirs('models', exist_ok=True)
    torch.save(capsule_net, model_path)
    print(f"Model saved to {model_path}")

    # Visualization: Reconstructed Images from the First Batch
    capsule_net.eval()
    print("Testing on training samples...")
    with torch.no_grad():
        batch_images = train_samples[0]  # Get the first batch of training samples
        batch_images = batch_images.to(device)

        for i in range(10):  # Limit visualization to the first 10 images
            image = batch_images[i].unsqueeze(0)  # Add batch dimension for single image
            _, reconstruction, _ = capsule_net(image)

            # Convert to numpy for visualization
            original_image = image.cpu().squeeze(0).numpy()
            reconstructed_image = reconstruction.cpu().squeeze(0).numpy()

            # Display original and reconstructed images
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(original_image[0], cmap='gray')
            axes[0].set_title("Original")
            axes[0].axis('off')

            axes[1].imshow(reconstructed_image[0], cmap='gray')
            axes[1].set_title("Reconstruction")
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()

    return losses, model_path
