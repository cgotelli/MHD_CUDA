# Image Point Set Distance Calculator

This Python project processes images, converts them into sets of points based on white pixels, and calculates the Modified Hausdorff Distance (MHD) between each set of points. The results are saved in blocks as distance matrices. This is particularly useful for comparing image similarity in large datasets.

## Features

- **Image Processing:** Converts images to grayscale and identifies white pixel locations.
- **Point Filtering:** Filters points to reduce computational load.
- **Distance Calculation:** Computes the Modified Hausdorff Distance between sets of points.
- **Block-wise Processing:** Saves results in blocks for efficient handling of large datasets.
- **GPU Support:** Leverages GPU acceleration if available.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.6+
- NumPy
- Pillow (PIL)
- PyTorch

You can install the required Python packages using:

```sh
pip install numpy pillow torch
```

## Usage

To run the script, use the following command:

```sh
python script.py /path/to/your/images
```

This command will process all `.png` and `.jpg` images in the specified directory.

### Optional Parameters

- `start_block` (optional): Specify the starting block index for processing. This is useful if you want to resume processing from a specific block.

```sh
python script.py /path/to/your/images 200
```

In this example, processing starts from the 200th image.

## How It Works

1. **Image Loading:** The script loads all `.png` and `.jpg` images from the specified directory.
2. **Point Extraction:** Each image is converted to grayscale, and the coordinates of white pixels are extracted.
3. **Distance Matrix Calculation:** The Modified Hausdorff Distance between each pair of point sets is calculated.
4. **Block Processing:** The distance matrix is processed in blocks to handle large datasets efficiently.
5. **Result Storage:** Each block of the distance matrix is saved to disk in the specified directory.

## Example

If you have a folder of images located at `/data/images`, you can run the script as follows:

```sh
python script.py /data/images
```

This will process all the images in the folder, calculate the distance matrix, and save the blocks of the matrix in `G:\distance_matrix_blocks`.

## Output

- The script saves the distance matrix blocks as `.npy` files in the specified save directory. 
- You will find files named like `distance_matrix_block_<start>_<end>_02.npy`.

## Performance Considerations

- **Block Size:** The `block_size` parameter in the script controls how many images are processed in a single block. You may need to adjust this value based on your system's memory capacity.
- **GPU Utilization:** The script automatically detects if a CUDA-compatible GPU is available and utilizes it for faster computations. If no GPU is available, the CPU will be used instead.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

If you have suggestions for improving this script, feel free to open an issue or submit a pull request.
