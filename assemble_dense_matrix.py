import numpy as np
import os
import re

# Function to extract the start and end indices from the filename
def extract_indices(filename):
    """Extracts the start and end indices of the row block from the filename."""
    match = re.search(r'_(\d+)_(\d+)_02\.npy', filename)
    if match:
        start_index = int(match.group(1))
        end_index = int(match.group(2))
        return start_index, end_index
    return None, None

# Main function to reassemble the distance matrix from blocks
def main(directory):
    # Get the list of files in the specified directory
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
    # Filter out files that do not match the expected pattern
    files = [f for f in files if extract_indices(f)[0] is not None]

    # Sort the files based on the start index of the row block
    files.sort(key=lambda f: extract_indices(f)[0])

    # List to store the parts of the matrix
    matrix_parts = []

    # Load and add the matrix parts to the list
    for file in files:
        filepath = os.path.join(directory, file)
        print(f"Reading and adding to the matrix: {file}")  # Print the filename
        matrix_part = np.load(filepath).astype(np.float32)  # Convert to float32 immediately
        matrix_parts.append(matrix_part)

    # Reconstruct the original square matrix by concatenating the parts vertically
    final_matrix = np.vstack(matrix_parts)

    # Copy the upper triangular part to the lower triangular part, block by block
    n = final_matrix.shape[0]
    block_size = 1000  # Block size, adjustable to balance between memory usage and performance

    for i in range(0, n, block_size):
        for j in range(i + 1, n, block_size):
            i_end = min(i + block_size, n)
            j_end = min(j + block_size, n)
            # Copy the transpose of the upper block to the corresponding lower block
            final_matrix[j:j_end, i:i_end] = final_matrix[i:i_end, j:j_end].T

    # Check if the matrix is square
    if final_matrix.shape[0] == final_matrix.shape[1]:
        # Save the reconstructed matrix in the same directory
        output_path = os.path.join(directory, "final_dense_matrix.npy")
        np.save(output_path, final_matrix)
        print(f"Reconstructed matrix saved at: {output_path}")
    else:
        print("Error: The reconstructed matrix is not square.")

# Entry point of the script
if __name__ == "__main__":
    import sys
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python assemble_dense_matrix.py <directory>")
    else:
        main(sys.argv[1])
