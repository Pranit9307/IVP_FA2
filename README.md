# IVP5 - Image and Video Processing Application Documentation

## Overview
IVP5 is a comprehensive image and video processing application built using Streamlit, OpenCV, and various image processing libraries. The application provides a wide range of image processing techniques, compression methods, and astronomical image analysis tools.

## Table of Contents
1. [Installation](#installation)
2. [Features](#features)
3. [Usage](#usage)
4. [Image Processing Functions](#image-processing-functions)
5. [Compression Techniques](#compression-techniques)
6. [Astronomical Image Processing](#astronomical-image-processing)
7. [Video Frame Processing](#video-frame-processing)

## Installation

### Prerequisites
- Python 3.7+
- Required Python packages:
  ```
  streamlit
  numpy
  opencv-python
  matplotlib
  pillow
  pywavelets
  scipy
  pandas
  ```

### Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run ivp5.py
   ```

## Features

### Main Components
1. Image Compression Techniques
   - Run Length Encoding (RLE)
   - Huffman Coding
   - JPEG Compression
   - Orthogonal Transforms (DCT & Haar)

2. Astronomical Image Processing
   - Star Detection & Counting
   - Enhancing Faint Objects
   - Noise Reduction
   - Combined Analysis

3. Video Frame Processing
   - Frame Extraction
   - Multiple Processing Techniques
   - Real-time Processing

## Usage

### Navigation
The application provides a sidebar navigation menu with the following options:
- About
- Image Compression Techniques
- Astronomical Image Processing
- Video Frame Processing

### Image Processing Functions

#### Basic Image Processing
1. **Thresholding**
   - Converts image to binary based on threshold value
   - Parameters: threshold value (0-255)

2. **Gray Level Slicing**
   - Highlights specific intensity ranges
   - Preserves details in selected gray levels

3. **Bit Plane Slicing**
   - Extracts specific bit planes
   - Useful for image analysis and compression

#### Edge Detection
1. **Sobel Edge Detection**
   - Detects edges using Sobel operators
   - Combines horizontal and vertical gradients

2. **Prewitt Edge Detection**
   - Uses Prewitt operators for edge detection
   - Good for noise reduction

3. **Robert Edge Detection**
   - Uses Robert cross operators
   - Fast edge detection method

4. **Laplacian**
   - Second-order derivative edge detection
   - Sensitive to noise

5. **LoG (Laplacian of Gaussian)**
   - Combines Gaussian smoothing with Laplacian
   - Reduces noise sensitivity

#### Filtering
1. **Low Pass Filter**
   - Smooths image
   - Reduces noise and high-frequency components

2. **High Pass Filter**
   - Enhances edges and details
   - Removes low-frequency components

3. **High Boost Filter**
   - Enhances edges while preserving image details
   - Combines original image with high-pass filtered version

4. **Gaussian Blur**
   - Smooths image using Gaussian kernel
   - Parameters: kernel size, sigma

## Compression Techniques

### Run Length Encoding (RLE)
- Lossless compression
- Effective for images with large uniform areas
- Simple and fast implementation

### Huffman Coding
- Lossless compression
- Variable-length coding based on frequency
- Good for skewed distributions

### JPEG Compression
- Lossy compression
- Uses DCT transform
- Quality factor adjustable
- Custom quantization tables

### Orthogonal Transforms
1. **DCT (Discrete Cosine Transform)**
   - Converts spatial to frequency domain
   - Used in JPEG compression
   - Energy compaction property

2. **Haar Wavelet Transform**
   - Multi-resolution analysis
   - Good for preserving structures
   - Used in specialized astronomical compression

## Astronomical Image Processing

### Star Detection & Counting
- Identifies stars in astronomical images
- Uses thresholding and contour detection
- Provides star count and visualization

### Enhancing Faint Objects
1. **Histogram Equalization**
   - Improves overall contrast
   - Enhances visibility of faint features

2. **Adaptive Histogram Equalization (CLAHE)**
   - Local contrast enhancement
   - Preserves details in different regions

3. **Log Transform**
   - Compresses dynamic range
   - Enhances faint objects

4. **Gamma Correction**
   - Adjusts mid-tone contrast
   - Customizable gamma value

### Noise Reduction
1. **Gaussian Blur**
   - Reduces random noise
   - Adjustable kernel size and sigma

2. **Median Filter**
   - Removes salt-and-pepper noise
   - Preserves edges

3. **Non-Local Means Denoising**
   - Advanced noise reduction
   - Preserves fine details

4. **Wavelet-based Denoising**
   - Multi-scale noise reduction
   - Preserves structures at different scales

## Video Frame Processing

### Frame Extraction
1. **Uniform Extraction**
   - Extracts frames at regular intervals
   - Good for general analysis

2. **Key Frame Extraction**
   - Detects significant changes
   - Extracts important frames

### Processing Options
- All image processing techniques available
- Real-time processing
- Multiple frame display
- Time-based frame selection

## Best Practices

### Image Processing
1. Start with noise reduction for noisy images
2. Use appropriate edge detection based on image characteristics
3. Adjust parameters based on image content
4. Combine multiple techniques for optimal results

### Compression
1. Use lossless compression for scientific data
2. Adjust JPEG quality based on intended use
3. Consider wavelet transforms for astronomical images
4. Test different compression methods for optimal results

### Astronomical Processing
1. Use CLAHE for enhancing faint objects
2. Combine multiple techniques for best results
3. Preserve scientific integrity in processing
4. Document all processing steps

## Troubleshooting

### Common Issues
1. **Memory Issues**
   - Reduce image size
   - Process in smaller batches
   - Use appropriate compression

2. **Processing Speed**
   - Optimize algorithm parameters
   - Use appropriate processing techniques
   - Consider hardware acceleration

3. **Quality Issues**
   - Adjust processing parameters
   - Try different techniques
   - Check input image quality

## Support
For issues and feature requests, please contact the development team.

## License
[Specify your license here]

## Version History
- v1.0.0: Initial release
  - Basic image processing functions
  - Compression techniques
  - Astronomical image processing
  - Video frame processing 
