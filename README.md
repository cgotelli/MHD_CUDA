
# Image Analysis Project with Modified Hausdorff Distance, UMAP, and Clustering

This repository contains the source code for a project that performs image analysis in three main steps using Jupyter notebooks:

1. **Image Processing**: Preprocessing and preparing images for analysis.
2. **Modified Hausdorff Distance Calculation**: Calculating the distance between all processed images.
3. **Result Analysis**: Using UMAP for dimensionality reduction and applying clustering techniques to interpret the data.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Image Processing](#1-image-processing)
  - [2. Modified Hausdorff Distance Calculation](#2-modified-hausdorff-distance-calculation)
  - [3. Result Analysis](#3-result-analysis)
- [Project Structure](#project-structure)
- [Contributions](#contributions)
- [License](#license)
- [Contact](#contact)

## Requirements

- Python 3.x
- Jupyter Notebook
- Required libraries (included in `requirements.txt`):
  - NumPy
  - OpenCV
  - SciPy
  - scikit-learn
  - UMAP-learn
  - Matplotlib
  - HDBSCAN

## Usage

### 1. Image Processing

Open and execute the notebook `01-ImagePreparation.ipynb` to process the input images. This notebook reads the images, preprocesses them by converting them to grayscale, resizing them, and normalizing them for analysis.

### 2. Modified Hausdorff Distance Calculation

Open and execute the notebook `02-MHD-matrixBuilder.ipynb` to calculate the Modified Hausdorff Distance (MHD) between the processed images. The result will be saved in a CSV file.

### 3. Result Analysis

Open and execute the notebook `03-MatrixAnalysis.ipynb` to perform UMAP dimensionality reduction and clustering using HDBSCAN and DBSCAN. This notebook generates clustering metrics and saves images of the cluster centroids.

## Project Structure

```
├── images/                        # Original images
├── processed/                     # Processed images
├── results/                       # Analysis results
├── 01-ImagePreparation.ipynb      # Notebook for image processing
├── 02-MHD-matrixBuilder.ipynb     # Notebook for calculating Modified Hausdorff Distance
├── 03-MatrixAnalysis.ipynb        # Notebook for UMAP and clustering analysis
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or support, please contact:

- **Name**: Your Name
- **Email**: [your_email@domain.com](mailto:your_email@domain.com)

---
*This README was generated to provide clear guidance on how to use and contribute to the project. Feel free to modify it according to your project’s specific needs.*
