import os
import time
from PIL import Image
import numpy as np
import torch

# Function to load images from a directory and convert them into sets of points
def load_images_to_points(directory):
    # List all files in the directory that end with .png or .jpg
    image_files = [f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]
    sets_of_points = []

    # Iterate over each image file
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path).convert('L')  # Convert image to grayscale
        image_np = np.array(image)
        
        # Get coordinates of white points (assuming a threshold of 128 for white pixels)
        points = np.column_stack(np.where(image_np > 128))
        
        # Filter to take only one out of every two points
        points = points[::2]
        
        sets_of_points.append(points)

    print("Done processing points")
    return sets_of_points

# Function to calculate the Modified Hausdorff Distance (MHD) between two batches of points
def calculate_mhd_batch(tensor1, tensor2):
    # Convert tensor to a set of points
    def tensor_to_points(tensor):
        return torch.nonzero(tensor.squeeze() > 0.1, as_tuple=False).float()
    
    # Convert the input tensors to points and move them to GPU
    A = tensor_to_points(tensor1).cuda()
    B = tensor_to_points(tensor2).cuda()
    if A.size(1) != B.size(1):
        raise ValueError("Both sets of points have different dimensions.")
    
    # Compute the pairwise distance matrix between the points
    dist_matrix = torch.cdist(A.unsqueeze(0), B.unsqueeze(0)).squeeze(0)
    fhd = torch.mean(torch.min(dist_matrix, dim=1)[0])  # Forward Hausdorff Distance
    rhd = torch.mean(torch.min(dist_matrix, dim=0)[0])  # Reverse Hausdorff Distance
    mhd = torch.max(fhd, rhd)  # Modified Hausdorff Distance
    
    return mhd.item()

# Function to compute and save a block of the distance matrix
def compute_and_save_distance_matrix_block(sets_of_points, block_start, block_size, save_dir, device):
    n = len(sets_of_points)
    end = min(block_start + block_size, n)
    block_height = end - block_start
    
    # Convert sets of points to tensors and move them to the GPU
    tensors = [torch.tensor(points, dtype=torch.float32).to(device) for points in sets_of_points]

    # Create a block of the distance matrix with NaN in the lower triangular part
    distance_matrix_block = np.full((block_height, n), np.nan)

    for i in range(block_height):
        tensor1 = tensors[block_start + i]
        distances = []
        for j in range(block_start + i, n):
            tensor2 = tensors[j]
            mhd = calculate_mhd_batch(tensor1, tensor2)
            distances.append(mhd)
        
        distance_matrix_block[i, block_start + i:n] = distances

        # Print message every five processed lines
        if (i + 1) % 5 == 0:
            print(f"Processed lines {block_start + i - 3} to {block_start + i + 1}")

    # Save the distance matrix block to disk
    np.save(os.path.join(save_dir, f'distance_matrix_block_{block_start}_{end}_02.npy'), distance_matrix_block)

# Main function to manage the overall processing
def main(directory, start_block=0):
    start_time = time.time()

    # Load images and convert them to sets of points
    sets_of_points = load_images_to_points(directory)
    n = len(sets_of_points)
    block_size = 100  # Adjust the block size according to available memory

    # Create directory to save the blocks of the distance matrix
    save_dir = 'G:\\distance_matrix_blocks'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Select device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process the images in blocks
    for block_start in range(start_block, n, block_size):
        compute_and_save_distance_matrix_block(sets_of_points, block_start, block_size, save_dir, device)
        print(f"Processed block {block_start} to {min(block_start + block_size, n)}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"All blocks of the distance matrix have been saved in '{save_dir}'")
    print(f"Execution time: {elapsed_time:.2f} seconds")

# Entry point of the script
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py <image_directory> [<start_block>]")
    else:
        directory = sys.argv[1]
        start_block = int(sys.argv[2]) if len(sys.argv) == 3 else 0
        main(directory, start_block)
