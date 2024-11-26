import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os

def test(model_path, capsule_net, test_images_folder):
    '''Loads a trained model, processes test images, and shows reconstruction outputs.'''
    
    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    capsule_net.to(device)
    
    # Load the saved model
    capsule_net = torch.load(model_path)
    capsule_net.eval()
    print(f'Model loaded from {model_path}')

    # Define image transformation (resize to 64x64 and grayscale)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # Get test image paths
    test_image_paths = [os.path.join(test_images_folder, fname) for fname in os.listdir(test_images_folder)]
    test_image_paths = [p for p in test_image_paths if os.path.isfile(p)]  # Ensure only files are selected

    # Process each test image and show the output
    for i, image_path in enumerate(test_image_paths):
        print(f"Processing {image_path}...")

        # Open and transform the image
        image = Image.open(image_path)
        image = transform(image)  # Apply transformations (grayscale, resize, tensor)
        image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Run the model on the image
        with torch.no_grad():
            _, reconstruction, _ = capsule_net(image)

        # Convert image tensors to numpy arrays for display
        original_image = image.cpu().squeeze(0).numpy()  # Remove batch dimension
        reconstructed_image = reconstruction.cpu().squeeze(0).numpy()

        # Display the original and reconstructed image
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Original image
        axes[0].imshow(original_image[0], cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        # Reconstructed image
        axes[1].imshow(reconstructed_image[0], cmap='gray')
        axes[1].set_title('Reconstruction')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        # Optionally, break after a few images
        if i >= 9:  # Show only the first 10 images
            break


if __name__ == '__main__':
    from CapsuleNetwork import CapsuleNetwork  # Import your model class

    # Specify the model path and test images folder
    model_path = 'models/capsule_net_20241121_194332.pth'  # Path to the saved model
    test_images_folder = 'test_images'  # Path to the folder containing test images

    # Instantiate the CapsuleNetwork
    capsule_net = CapsuleNetwork()

    # Run the test script
    test(model_path, capsule_net, test_images_folder)
