import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import pywt
from scipy.fftpack import dct, idct
import heapq
import collections
from skimage import exposure
import pandas as pd
import tempfile
import os

# =========================
# Image Processing Functions
# =========================

def process_thresholding(roi, threshold=127):
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def process_gray_level_slicing(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    mask = (gray > 50) & (gray < 200)
    sliced = np.zeros_like(gray)
    sliced[mask] = 255
    return cv2.cvtColor(sliced, cv2.COLOR_GRAY2RGB)

def process_bit_plane_slicing(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    bit_plane = (gray & 128).astype(np.uint8) * 255
    return cv2.cvtColor(bit_plane, cv2.COLOR_GRAY2RGB)

def process_low_pass(roi):
    return cv2.GaussianBlur(roi, (5, 5), 0)

def process_high_pass(roi):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(roi, -1, kernel)

def process_high_boost(roi):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(roi, -1, kernel)

def process_sobel(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    # Sobel X and Y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Combine X and Y
    sobel = np.sqrt(sobelx**2 + sobely**2)
    # Normalize
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(sobel.astype(np.uint8), cv2.COLOR_GRAY2RGB)

def process_prewitt(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    # Prewitt kernels
    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])
    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]])
    # Apply kernels
    prewittx = cv2.filter2D(gray, -1, kernelx)
    prewitty = cv2.filter2D(gray, -1, kernely)
    # Combine X and Y
    prewitt = np.sqrt(prewittx**2 + prewitty**2)
    # Normalize
    prewitt = cv2.normalize(prewitt, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(prewitt.astype(np.uint8), cv2.COLOR_GRAY2RGB)

def process_robert(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    # Robert kernels
    kernelx = np.array([[1, 0],
                        [0, -1]])
    kernely = np.array([[0, 1],
                        [-1, 0]])
    # Apply kernels
    robertx = cv2.filter2D(gray, -1, kernelx)
    roberty = cv2.filter2D(gray, -1, kernely)
    # Combine X and Y
    robert = np.sqrt(robertx**2 + roberty**2)
    # Normalize
    robert = cv2.normalize(robert, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(robert.astype(np.uint8), cv2.COLOR_GRAY2RGB)

def process_laplacian(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)

def process_log(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def process_gaussian(roi):
    return cv2.GaussianBlur(roi, (5, 5), 0)

# Set page config
st.set_page_config(page_title="Astronomical Image Compression", layout="wide")

# Helper functions for compression techniques
def run_length_encoding(image_array):
    """Perform basic run-length encoding on a flattened image array"""
    if len(image_array.shape) > 2:
        # For RGB images, convert to grayscale for RLE
        if image_array.shape[2] == 3:
            flat_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY).flatten()
        else:
            flat_array = image_array[:,:,0].flatten()  # Just use first channel
    else:
        flat_array = image_array.flatten()
    
    rle_result = []
    count = 1
    current = flat_array[0]
    
    for i in range(1, len(flat_array)):
        if flat_array[i] == current:
            count += 1
        else:
            rle_result.append((current, count))
            current = flat_array[i]
            count = 1
    
    # Add the last run
    rle_result.append((current, count))
    
    return rle_result, len(rle_result), len(flat_array)



def build_huffman_tree(data):
    """Build a Huffman tree from data"""
    # Count frequency of each value
    frequency = collections.Counter(data)
    
    # Create leaf nodes for each value
    heap = [[weight, [value, ""]] for value, weight in frequency.items()]
    heapq.heapify(heap)
    
    # Build Huffman tree
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Extract codes
    huffman_codes = {}
    for pair in heap[0][1:]:
        huffman_codes[pair[0]] = pair[1]
    
    return huffman_codes



def apply_huffman_coding(image_array):
    """Apply Huffman coding to an image array"""
    if len(image_array.shape) > 2:
        # For RGB images, process each channel separately
        if image_array.shape[2] == 3:
            # Process Y channel only for demo purposes
            ycrcb = cv2.cvtColor(image_array, cv2.COLOR_RGB2YCrCb)
            y_channel = ycrcb[:,:,0].flatten()
            huffman_codes = build_huffman_tree(y_channel)
            
            # Calculate average code length
            code_lengths = [len(huffman_codes[value]) for value in y_channel if value in huffman_codes]
            avg_length = sum(code_lengths) / len(code_lengths) if code_lengths else 0
            
            # Original vs compressed bit size
            original_bits = y_channel.size * 8
            compressed_bits = sum(code_lengths) if code_lengths else 0
            
            return huffman_codes, len(huffman_codes), avg_length, original_bits, compressed_bits
    else:
        flat_array = image_array.flatten()
        huffman_codes = build_huffman_tree(flat_array)
        
        code_lengths = [len(huffman_codes[value]) for value in flat_array if value in huffman_codes]
        avg_length = sum(code_lengths) / len(code_lengths) if code_lengths else 0
        
        original_bits = flat_array.size * 8
        compressed_bits = sum(code_lengths) if code_lengths else 0
        
        return huffman_codes, len(huffman_codes), avg_length, original_bits, compressed_bits

def display_dct_transform(image):
    """Apply and display DCT transform"""
    # Convert to grayscale if RGB
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply DCT
    dct_image = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    
    # Log transform for better visualization
    log_dct = np.log(np.abs(dct_image) + 1)
    
    # Normalize for display
    log_dct_normalized = (log_dct - np.min(log_dct)) / (np.max(log_dct) - np.min(log_dct)) * 255
    return log_dct_normalized.astype(np.uint8)

def haar_transform(image):
    """Apply and display Haar wavelet transform"""
    # Convert to grayscale if RGB
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply 2D Haar wavelet transform
    coeffs = pywt.dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs
    
    # Prepare for visualization
    # Normalize each component
    cA_norm = (cA - np.min(cA)) / (np.max(cA) - np.min(cA) + 1e-10) * 255
    cH_norm = (cH - np.min(cH)) / (np.max(cH) - np.min(cH) + 1e-10) * 255
    cV_norm = (cV - np.min(cV)) / (np.max(cV) - np.min(cV) + 1e-10) * 255
    cD_norm = (cD - np.min(cD)) / (np.max(cD) - np.min(cD) + 1e-10) * 255
    
    # Create composite image
    h, w = gray.shape
    h2, w2 = h//2, w//2
    haar_image = np.zeros((h, w), dtype=np.uint8)
    haar_image[:h2, :w2] = cA_norm
    haar_image[:h2, w2:] = cH_norm
    haar_image[h2:, :w2] = cV_norm
    haar_image[h2:, w2:] = cD_norm
    
    return haar_image

def quantize_dct(dct_image, quality_factor):
    """Apply quantization to DCT coefficients"""
    # Standard JPEG quantization matrix (luminance)
    q_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # Scale quantization matrix based on quality factor
    if quality_factor < 50:
        scale = 5000 / quality_factor
    else:
        scale = 200 - 2 * quality_factor
    
    q_scaled = np.floor((q_table * scale + 50) / 100)
    q_scaled[q_scaled < 1] = 1  # Ensure minimum value is 1
    
    # Create full-sized quantization matrix
    h, w = dct_image.shape
    q_full = np.zeros((h, w))
    
    # Tile the quantization matrix
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block_h = min(8, h - i)
            block_w = min(8, w - j)
            q_full[i:i+block_h, j:j+block_w] = q_scaled[:block_h, :block_w]
    
    # Apply quantization
    quantized = np.round(dct_image / q_full)
    
    # Count non-zero coefficients
    non_zero_count = np.count_nonzero(quantized)
    
    # Dequantize
    dequantized = quantized * q_full
    
    return quantized, dequantized, non_zero_count

# Astronomical image enhancement functions
def adaptive_histogram_equalization(image):
    """Apply adaptive histogram equalization for enhancing faint stars"""
    if len(image.shape) > 2:
        # For color images, only enhance the luminance channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y_channel = ycrcb[:,:,0]
        
        # Apply CLAHE to luminance channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        y_enhanced = clahe.apply(y_channel)
        
        ycrcb[:,:,0] = y_enhanced
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        # For grayscale images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
    
    return enhanced

def star_detection(image):
    """Detect stars in an astronomical image"""
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to identify bright spots
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours of bright spots
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw circles around detected stars
    result = image.copy() if len(image.shape) > 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for contour in contours:
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw circle
            cv2.circle(result, (cX, cY), 10, (0, 255, 0), 2)
    
    return result, len(contours)



def noise_reduction(image):
    """Apply noise reduction techniques"""
    if len(image.shape) > 2:
        # For color images
        result = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        # For grayscale images
        result = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    return result

# Main App
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a mode", [
        "About",
        "Image Compression Techniques",
        "Astronomical Image Processing",
        "Video Frame Processing"
    ])
    
    if app_mode == "About":
        st.title("Astronomical Image Compression Application")
        
        st.markdown("""
        ## 🌌 About This Application
        
        This application demonstrates various image compression and processing techniques specifically optimized for astronomical imagery. Astronomical images present unique challenges for compression algorithms:
        
        1. **High Dynamic Range**: Need to preserve both bright stars and faint nebulae
        2. **Faint Objects**: Critical to preserve low-intensity features
        3. **Scientific Integrity**: Compression must not introduce artifacts that could be mistaken for astronomical phenomena
        4. **Large File Sizes**: Telescope surveys generate massive amounts of data
        
        ## Features
        
        ### Compression Techniques
        - **Run Length Encoding (RLE)**: Lossless compression for star field images with large dark areas
        - **Huffman Coding**: Variable-length encoding based on pixel value frequency
        - **JPEG Compression**: Optimized for preserving faint celestial objects
        - **Orthogonal Transforms**: Explore how DCT and Haar transforms represent astronomical data
        
        ### Specialized Processing
        - **Star Detection**: Identify and count stars in images
        - **Adaptive Histogram Equalization**: Enhance visibility of faint objects
        - **Noise Reduction**: Remove sensor noise while preserving astronomical details
        
        """)
        
        st.image("https://apod.nasa.gov/apod/image/2304/M31_HubbleSpitzerSubaru_960.jpg", 
                caption="Example: Andromeda Galaxy (M31) - This is the type of astronomical image this application is designed to process", 
                use_column_width=True)
    
    elif app_mode == "Image Compression Techniques":
        st.title("Astronomical Image Compression Techniques")
        
        option = st.selectbox("Choose a compression technique", [
            "Types of Redundancies in Astronomical Images",
            "Run Length Encoding (RLE)",
            "Huffman Coding",
            "JPEG Compression for Astronomy",
            "Orthogonal Transforms (DCT & Haar)",
            "Compression Comparison"
        ])
        
        if option == "Types of Redundancies in Astronomical Images":
            st.markdown("""
            ## Types of Redundancies in Astronomical Images
            
            ### Coding Redundancy
            - Many astronomical images have large dark areas with similar pixel values
            - In deep space images, the distribution of pixel values is often highly skewed
            - Solution: Variable length coding (Huffman) optimized for astronomical histograms
            
            ### Spatial Redundancy
            - Star fields often have large areas of uniform background
            - Nebulae and galaxies have smoothly varying regions
            - Solution: Run length encoding, predictive coding, and transform-based methods
            
            ### Psychovisual Redundancy
            - Human eyes are less sensitive to small variations in very dark areas
            - Important to preserve bright point sources (stars) and faint extended sources (nebulae)
            - Solution: Customized quantization tables that preserve scientifically important features
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image("https://apod.nasa.gov/apod/image/2010/OrionNebula_HubbleSerrano_960.jpg", 
                        caption="Nebula with smooth gradients (spatial redundancy)", 
                        use_column_width=True)
            with col2:
                st.image("https://apod.nasa.gov/apod/image/2203/NGC4945_HaLRGBpugh_960.jpg", 
                        caption="Galaxy with both point sources and extended structure", 
                        use_column_width=True)
        
        elif option == "Run Length Encoding (RLE)":
            st.markdown("""
            ## Run Length Encoding for Astronomical Images
            
            Run Length Encoding (RLE) is particularly effective for astronomical images because:
            
            - Star fields have large areas of dark, uniform background
            - Binary masks for object detection can be highly compressible
            - Simple and fast for real-time telescope data processing
            
            ### How RLE Works
            RLE encodes consecutive repeated pixel values as a pair: (value, count)
            
            For example, a row of pixels:
            ```
            0,0,0,0,0,255,255,0,0,0,0,0,0,0,0,100,100,100
            ```
            
            Would be encoded as:
            ```
            (0,5),(255,2),(0,8),(100,3)
            ```
            
            This works well for astronomical images with large areas of similar values.
            """)
            
            uploaded_image = st.file_uploader("Upload an astronomical image for RLE demonstration", type=["jpg", "png", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                image_array = np.array(image)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_column_width=True)
                
                with col2:
                    # Create a binary version to show RLE potential
                    if len(image_array.shape) > 2:
                        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = image_array
                    
                    # Threshold to create binary star field
                    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                    st.image(binary, caption="Binary Star Field (Better for RLE)", use_column_width=True)
                
                # Run RLE on both original and binary
                rle_result, encoded_size, original_size = run_length_encoding(image_array)
                rle_binary, encoded_binary_size, original_binary_size = run_length_encoding(binary)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    ### Original Image RLE Results
                    - Original size: {original_size} pixels
                    - Encoded size: {encoded_size} count-value pairs
                    - Compression ratio: {original_size / encoded_size:.2f}:1
                    """)
                
                with col2:
                    st.markdown(f"""
                    ### Binary Image RLE Results
                    - Original size: {original_binary_size} pixels
                    - Encoded size: {encoded_binary_size} count-value pairs
                    - Compression ratio: {original_binary_size / encoded_binary_size:.2f}:1
                    """)
                
                st.markdown("### Analysis")
                st.markdown(f"""
                RLE is {original_binary_size / encoded_binary_size / (original_size / encoded_size):.2f}x more effective on the binary star field than on the original image.
                
                This demonstrates why astronomers often use RLE for:
                - Star catalogs (binary representations)
                - Object masks
                - Regions of interest
                """)
        
        elif option == "Huffman Coding":
            st.markdown("""
            ## Huffman Coding for Astronomical Data
            
            Huffman coding creates variable-length codes based on pixel value frequencies:
            - More frequent values get shorter codes
            - Less frequent values get longer codes
            
            This is particularly useful for astronomical images because:
            - Dark sky pixels often dominate the histogram
            - Bright stars and features are relatively rare
            - The pixel value distribution is typically highly skewed
            
            ### Process:
            1. Calculate frequency of each pixel value
            2. Build a Huffman tree from these frequencies
            3. Assign codes based on the tree
            4. Replace each pixel value with its code
            """)
            
            uploaded_image = st.file_uploader("Upload an astronomical image for Huffman coding demonstration", type=["jpg", "png", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                image_array = np.array(image)
                
                # Display original image
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Apply Huffman coding
                huffman_codes, unique_values, avg_bits, original_bits, compressed_bits = apply_huffman_coding(image_array)
                compression_ratio = original_bits / compressed_bits if compressed_bits else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    ### Huffman Coding Results
                    - Original bits: {original_bits} (8 bits per pixel)
                    - Compressed bits: {compressed_bits}
                    - Compression ratio: {compression_ratio:.2f}:1
                    - Average bits per pixel: {avg_bits:.2f}
                    - Unique values found: {unique_values}
                    """)
                
                with col2:
                    # Create histogram for visualization
                    if len(image_array.shape) > 2:
                        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = image_array
                    
                    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(range(256), hist.flatten(), width=1)
                    ax.set_xlim([0, 256])
                    ax.set_title('Pixel Value Histogram')
                    ax.set_xlabel('Pixel Value')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
                
                # Display a few Huffman codes
                st.markdown("### Sample Huffman Codes")
                sample_codes = {k: v for k, v in list(huffman_codes.items())[:10]}
                
                code_data = [{"Value": k, "Code": v, "Length": len(v)} for k, v in sample_codes.items()]
                st.table(code_data)
                
                st.markdown("""
                ### Astronomical Significance
                
                Huffman coding is especially efficient for astronomical images because:
                - Background sky pixels (usually dark values) receive very short codes
                - Rare bright stars and features receive longer codes
                - The overall file size is significantly reduced while preserving all data
                
                This lossless compression is critical for scientific analysis, as no information is lost.
                """)
        
        elif option == "JPEG Compression for Astronomy":
            st.markdown("""
            ## JPEG Compression for Astronomical Images
            
            Standard JPEG compression can be problematic for astronomical images because:
            - It can eliminate faint objects
            - It may introduce artifacts that look like celestial features
            - It often discards important high-frequency information
            
            ### Astronomical JPEG Optimization
            
            For astronomy, JPEG compression should be modified with:
            1. Custom quantization tables that preserve faint features
            2. Higher quality factors for scientific integrity
            3. Special handling of dark areas where faint objects may exist
            """)
            
            uploaded_image = st.file_uploader("Upload an astronomical image for JPEG demonstration", type=["jpg", "png", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                image_array = np.array(image)
                
                # Display original image
                st.image(image, caption="Original Image", use_column_width=True)
                
                # DCT transform visualization
                if len(image_array.shape) > 2:
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image_array
                
                # Apply DCT
                dct_image = dct(dct(gray.T, norm='ortho').T, norm='ortho')
                
                # Quality slider
                quality_factor = st.slider("JPEG Quality Factor", 1, 100, 80)
                
                # Apply quantization
                quantized, dequantized, non_zero_count = quantize_dct(dct_image, quality_factor)
                
                # Apply inverse DCT to get compressed image
                compressed = idct(idct(dequantized.T, norm='ortho').T, norm='ortho')
                compressed = np.clip(compressed, 0, 255).astype(np.uint8)
                
                col1, col2 = st.columns(2)
                with col1:
                    # Display DCT coefficients
                    log_dct = np.log(np.abs(dct_image) + 1)
                    log_dct_norm = (log_dct - np.min(log_dct)) / (np.max(log_dct) - np.min(log_dct)) * 255
                    st.image(log_dct_norm.astype(np.uint8), caption="DCT Coefficients (Log Scale)", use_column_width=True)
                
                with col2:
                    # Display quantized DCT coefficients
                    log_quantized = np.log(np.abs(quantized) + 1)
                    log_quantized_norm = (log_quantized - np.min(log_quantized)) / (np.max(log_quantized) - np.min(log_quantized)) * 255
                    st.image(log_quantized_norm.astype(np.uint8), caption="Quantized DCT (Log Scale)", use_column_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(gray, caption="Original Grayscale", use_column_width=True)
                
                with col2:
                    st.image(compressed, caption=f"Compressed (Quality: {quality_factor})", use_column_width=True)
                
                # Calculate compression ratio
                original_pixels = gray.size
                non_zero_ratio = non_zero_count / original_pixels
                compression_ratio = 1 / non_zero_ratio if non_zero_ratio > 0 else 0
                
                st.markdown(f"""
                ### Compression Analysis
                
                - Original coefficients: {original_pixels}
                - Non-zero coefficients after quantization: {non_zero_count} ({non_zero_ratio*100:.2f}%)
                - Approximate compression ratio: {compression_ratio:.2f}:1
                
                #### Impact on Astronomical Features
                
                At quality factor {quality_factor}:
                - Faint objects: {"Preserved" if quality_factor > 85 else "Partially preserved" if quality_factor > 70 else "May be lost"}
                - Star details: {"Fully preserved" if quality_factor > 90 else "Mostly preserved" if quality_factor > 75 else "Degraded"}
                - Background noise: {"Preserved" if quality_factor > 95 else "Reduced" if quality_factor > 60 else "Significantly reduced"}
                
                For scientific analysis, quality factors above 85 are generally recommended.
                """)
                
                # Show difference image
                difference = cv2.absdiff(gray, compressed)
                # Enhance difference to make it visible
                difference_enhanced = cv2.equalizeHist(difference)
                
                st.image(difference_enhanced, caption="Enhanced Difference Image (Shows what information was lost)", use_column_width=True)
        
        elif option == "Orthogonal Transforms (DCT & Haar)":
            st.markdown("""
            ## Orthogonal Transforms for Astronomical Images
            
            Transforms convert spatial domain data to frequency domain, which can reveal different properties of astronomical images:
            
            ### Discrete Cosine Transform (DCT)
            - Decomposes image into cosine functions at different frequencies
            - Low frequencies represent large-scale structures (galaxies, nebulae)
            - High frequencies represent fine details (stars, filaments)
            - Core transform used in JPEG compression
            
            ### Haar Wavelet Transform
            - Decomposes image into averages and differences at multiple scales
            - Provides multi-resolution analysis capability
            - Can effectively separate stars from extended sources
            - Used in specialized astronomical image compression
            """)
            
            uploaded_image = st.file_uploader("Upload astronomical image for transform visualization", type=["jpg", "jpeg", "png"])
            if uploaded_image:
                image = Image.open(uploaded_image).convert("RGB")
                image_array = np.array(image)
                
                # Display original image
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Choose transform type
                transform_type = st.radio("Select Transform Type", ["DCT", "Haar Wavelet", "Both Side-by-Side"])
                
                if transform_type == "DCT" or transform_type == "Both Side-by-Side":
                    dct_image = display_dct_transform(image_array)
                    if transform_type == "DCT":
                        st.image(dct_image, caption="DCT Transform (Log Scale)", use_column_width=True)
                        
                        st.markdown("""
                        ### DCT Transform Analysis
                        
                        In astronomical images, the DCT transform reveals:
                        
                        - **Bright center point**: Average brightness (DC component)
                        - **Near center**: Large structures like galaxies and nebulae
                        - **Mid frequencies**: Medium-scale structures
                        - **High frequencies (corners)**: Stars, noise, and fine details
                        
                        By preserving low and mid frequencies while selectively keeping high frequencies, 
                        compression can maintain scientific value while reducing file size.
                        """)
                
                if transform_type == "Haar Wavelet" or transform_type == "Both Side-by-Side":
                    haar_image = haar_transform(image_array)
                    if transform_type == "Haar Wavelet":
                        st.image(haar_image, caption="Haar Wavelet Transform", use_column_width=True)
                        
                        st.markdown("""
                        ### Haar Wavelet Analysis
                        
                        The Haar transform divides the image into:
                        
                        - **Top-left (LL)**: Low resolution approximation of the image
                        - **Top-right (LH)**: Horizontal edges and features
                        - **Bottom-left (HL)**: Vertical edges and features
                        - **Bottom-right (HH)**: Diagonal details (often contains stars)
                        
                        For astronomical images, this separation helps in:
                        - Isolating point sources (stars) from extended sources (nebulae)
                        - Multi-resolution analysis of structures at different scales
                        - Progressive transmission of large survey images
                        """)
                
                if transform_type == "Both Side-by-Side":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(dct_image, caption="DCT Transform", use_column_width=True)
                    
                    with col2:
                        st.image(haar_image, caption="Haar Wavelet Transform", use_column_width=True)
                    
                    st.markdown("""
                    ### Comparison for Astronomical Images
                    ### Comparison for Astronomical Images
                    
                    **DCT (Left) vs. Haar (Right) for Astronomy:**
                    
                    - **For nebulae and galaxies**: DCT often provides better energy compaction
                    - **For star fields**: Haar wavelets can better isolate point sources
                    - **For mixed scenes**: Combining both approaches can be optimal
                    
                    Many specialized astronomical image compression formats use wavelet transforms 
                    (e.g., FITS tiled-compressed format) because they better preserve 
                    both point sources and extended structures.
                    """)
        
        elif option == "Compression Comparison":
            st.markdown("""
            ## Compression Method Comparison for Astronomy
            
            Different compression methods have varying impacts on astronomical data:
            
            | Method | Lossless | Pros | Cons | Best For |
            |--------|----------|------|------|----------|
            | RLE | Yes | Simple, fast | Only efficient for uniform areas | Binary masks, sparse fields |
            | Huffman | Yes | Good for skewed distributions | Limited compression ratio | Preserving all data for analysis |
            | JPEG (DCT) | No | High compression ratios | Can lose faint objects | Public outreach images |
            | Wavelets | Can be both | Multi-resolution, preserves structures | More complex | Scientific archives |
            
            Let's compare them all on your astronomical image.
            """)
            
            uploaded_image = st.file_uploader("Upload an astronomical image for comparison", type=["jpg", "png", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                image_array = np.array(image)
                
                # Display original image
                st.image(image, caption="Original Image", use_column_width=True)
                
                if len(image_array.shape) > 2:
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image_array
                
                # Apply all compression methods
                # RLE
                rle_result, encoded_size, original_size = run_length_encoding(gray)
                rle_ratio = original_size / encoded_size
                
                # Huffman
                huffman_codes, unique_values, avg_bits, original_bits, compressed_bits = apply_huffman_coding(gray)
                huffman_ratio = original_bits / compressed_bits if compressed_bits else 0
                
                # JPEG with different quality factors
                quality_factors = [50, 75, 95]
                jpeg_images = []
                jpeg_sizes = []
                
                for quality in quality_factors:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                    result, encimg = cv2.imencode('.jpg', gray, encode_param)
                    decimg = cv2.imdecode(encimg, 0)
                    
                    jpeg_images.append(decimg)
                    jpeg_sizes.append(len(encimg))
                
                # Create columns for JPEG comparison
                st.subheader("JPEG Compression at Different Quality Levels")
                cols = st.columns(len(quality_factors))
                
                for i, (quality, img, size) in enumerate(zip(quality_factors, jpeg_images, jpeg_sizes)):
                    with cols[i]:
                        st.image(img, caption=f"Quality: {quality}", use_column_width=True)
                        mb_size = size / (1024 * 1024)  # Convert bytes to MB
                        st.markdown(f"Size: {size} bytes ({mb_size:.2f} MB)\nRatio: {original_size / size:.2f}:1")                
                # Create a comparison table
                st.subheader("Compression Method Comparison")
                
                # Calculate average JPEG results
                avg_jpeg_ratio = sum([original_size / size for size in jpeg_sizes]) / len(jpeg_sizes)
                
                comparison_data = {
                    "Method": ["Run Length Encoding", "Huffman Coding", "JPEG (Average)", "JPEG (Quality 95)"],
                    "Lossless": ["Yes", "Yes", "No", "No"],
                    "Compression Ratio": [f"{rle_ratio:.2f}:1", f"{huffman_ratio:.2f}:1", 
                                         f"{avg_jpeg_ratio:.2f}:1", f"{original_size / jpeg_sizes[2]:.2f}:1"],
                    "Recommended For": ["Binary star masks", "Scientific preservation", 
                                       "General purpose", "High-quality publication"]
                }
                
                df = pd.DataFrame(comparison_data)
                st.table(df)
                
                # Show preservation of faint objects
                st.subheader("Preservation of Faint Objects")
                
                # Create side-by-side comparison of high-quality vs low-quality
                col1, col2 = st.columns(2)
                with col1:
                    # Enhance contrast to show faint objects
                    enhanced_original = exposure.equalize_hist(gray)
                    st.image(enhanced_original, caption="Original (Enhanced to show faint objects)", use_column_width=True)
                
                with col2:
                    # Enhance low quality to show what's lost
                    enhanced_low = exposure.equalize_hist(jpeg_images[0])
                    st.image(enhanced_low, caption="Low Quality JPEG (Enhanced)", use_column_width=True)
                
                # Show difference image
                diff = cv2.absdiff(gray, jpeg_images[0])
                diff_enhanced = exposure.equalize_hist(diff)
                st.image(diff_enhanced, caption="Enhanced Difference Image - Bright areas show lost information", use_column_width=True)
                
                st.markdown("""
                ### Recommendations for Astronomical Images
                
                **For Scientific Data:**
                - Use lossless methods (RLE, Huffman, FITS compression) for raw scientific data
                - Preserve full bit depth (often 16-bit or 32-bit float)
                - Consider wavelet-based compression for large survey images
                
                **For Publication/Sharing:**
                - High-quality JPEG (95+) for most public images
                - Consider PNG for images with text overlays or sharp boundaries
                - JPEG-2000 or HEIC for advanced applications
                
                **For Large Surveys:**
                - Tiled compression formats for access to specific regions
                - Progressive encoding for quick preview of large areas
                - Custom formats with region-of-interest capabilities
                """)
    
    elif app_mode == "Astronomical Image Processing":
        st.title("Specialized Astronomical Image Processing")
        
        option = st.selectbox("Choose a processing technique", [
            "Star Detection & Counting",
            "Enhancing Faint Objects",
            "Noise Reduction",
            "Combined Analysis"
        ])
        
        if option == "Star Detection & Counting":
            st.markdown("""
            ## Star Detection & Counting
            
            Detecting stars in astronomical images is a fundamental task that:
            - Helps create star catalogs
            - Provides input for astrometry (positioning)
            - Aids in identifying non-stellar objects
            
            This process works by:
            1. Applying thresholding to identify bright points
            2. Finding contours or using blob detection
            3. Filtering by size/intensity to distinguish stars from noise
            """)
            
            uploaded_image = st.file_uploader("Upload a star field image", type=["jpg", "png", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                image_array = np.array(image)
                
                # Display original image
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Allow threshold adjustment
                threshold = st.slider("Detection Threshold", 0, 255, 127)
                
                # Process for star detection
                if len(image_array.shape) > 2:
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image_array
                
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Apply threshold to identify bright spots
                _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
                
                # Find contours of bright spots
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw circles around detected stars
                result = image_array.copy() if len(image_array.shape) > 2 else cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                for contour in contours:
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # Draw circle
                        cv2.circle(result, (cX, cY), 10, (0, 255, 0), 2)
                
                # Display processed image
                st.image(result, caption=f"Detected Stars: {len(contours)}", use_column_width=True)
                
                # Display binary threshold
                st.image(thresh, caption="Binary Threshold Image", use_column_width=True)
                
                st.markdown(f"""
                ### Star Detection Results
                
                - Total stars detected: {len(contours)}
                - Detection threshold: {threshold}
                
                For astronomical purposes, this initial detection would typically be followed by:
                
                1. **Photometry**: Measuring the brightness of each star
                2. **Astrometry**: Precise position determination
                3. **Classification**: Separating stars from galaxies and artifacts
                
                This basic detection provides a foundation for more advanced analysis.
                """)
        
        elif option == "Enhancing Faint Objects":
            st.markdown("""
            ## Enhancing Faint Objects
            
            Astronomical images often contain faint objects that are difficult to see:
            - Distant galaxies
            - Nebulous regions
            - Faint stars
            
            Several techniques can enhance these features:
            
            1. **Histogram Equalization**: Improves overall contrast
            2. **Adaptive Histogram Equalization**: Enhances local contrast
            3. **Log Transformation**: Compresses dynamic range
            4. **Gamma Correction**: Adjusts mid-tone contrast
            """)
            
            uploaded_image = st.file_uploader("Upload an astronomical image with faint features", type=["jpg", "png", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                image_array = np.array(image)
                
                # Display original image
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Create enhancement options
                enhancement_method = st.radio("Enhancement Method", [
                    "Histogram Equalization", 
                    "Adaptive Histogram Equalization (CLAHE)",
                    "Log Transform",
                    "Gamma Correction"
                ])
                
                if enhancement_method == "Histogram Equalization":
                    if len(image_array.shape) > 2:
                        # For color images, enhance each channel
                        enhanced = np.zeros_like(image_array)
                        for i in range(3):
                            enhanced[:,:,i] = cv2.equalizeHist(image_array[:,:,i])
                    else:
                        enhanced = cv2.equalizeHist(image_array)
                
                elif enhancement_method == "Adaptive Histogram Equalization (CLAHE)":
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    
                    if len(image_array.shape) > 2:
                        # For color images, convert to LAB color space
                        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
                        lab_planes = cv2.split(lab)
                        
                        # Apply CLAHE to L-channel
                        lab_planes[0] = clahe.apply(lab_planes[0])
                        
                        # Merge back
                        lab = cv2.merge(lab_planes)
                        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    else:
                        enhanced = clahe.apply(image_array)
                
                elif enhancement_method == "Log Transform":
                    # Apply log transform c*log(1+r)
                    c = 255 / np.log(1 + np.max(image_array))
                    
                    if len(image_array.shape) > 2:
                        enhanced = np.zeros_like(image_array)
                        for i in range(3):
                            enhanced[:,:,i] = c * np.log1p(image_array[:,:,i])
                    else:
                        enhanced = c * np.log1p(image_array)
                    
                    enhanced = np.uint8(enhanced)
                
                elif enhancement_method == "Gamma Correction":
                    gamma = st.slider("Gamma Value", 0.1, 5.0, 0.5, 0.1)
                    
                    # Apply gamma correction
                    if len(image_array.shape) > 2:
                        # Normalize to 0-1 range
                        normalized = image_array / 255.0
                        # Apply gamma
                        corrected = np.power(normalized, gamma)
                        # Scale back to 0-255
                        enhanced = (corrected * 255).astype(np.uint8)
                    else:
                        normalized = image_array / 255.0
                        corrected = np.power(normalized, gamma)
                        enhanced = (corrected * 255).astype(np.uint8)
                
                # Display enhanced image
                st.image(enhanced, caption=f"Enhanced Image ({enhancement_method})", use_column_width=True)
                
                # Side-by-side comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_array, caption="Original Detail", use_column_width=True)
                with col2:
                    st.image(enhanced, caption="Enhanced Detail", use_column_width=True)
                
                st.markdown(f"""
                ### Enhancement Analysis
                
                The {enhancement_method} technique has revealed:
                
                - **Improved visibility** of faint features
                - **Enhanced contrast** between foreground and background
                - **Better detail** in previously dark regions
                
                For scientific analysis, it's important to note that these enhancements can:
                - Alter the relative brightness of objects
                - Potentially introduce artifacts
                - Change the statistical properties of the data
                
                These enhanced images are best used for:
                - Visual inspection and discovery
                - Initial feature identification
                - Public outreach and education
                
                For quantitative measurements, the original calibrated data should be used.
                """)
        
        elif option == "Noise Reduction":
            st.markdown("""
            ## Noise Reduction for Astronomical Images
            
            Astronomical imaging is often plagued by various noise sources:
            
            - **Shot noise**: From photon counting statistics
            - **Read noise**: From camera electronics
            - **Hot pixels**: Sensor defects appearing as bright spots
            - **Cosmic rays**: High energy particles creating streaks
            
            Effective noise reduction must preserve scientific data while removing unwanted artifacts.
            """)
            
            uploaded_image = st.file_uploader("Upload an astronomical image with noise", type=["jpg", "png", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                image_array = np.array(image)
                
                # Display original image
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Create noise reduction options
                noise_method = st.radio("Noise Reduction Method", [
                    "Gaussian Blur", 
                    "Median Filter",
                    "Non-Local Means Denoising",
                    "Wavelet-based Denoising"
                ])
                
                if noise_method == "Gaussian Blur":
                    kernel_size = st.slider("Kernel Size", 1, 31, 5, 2)
                    sigma = st.slider("Sigma", 0.1, 10.0, 1.5, 0.1)
                    
                    denoised = cv2.GaussianBlur(image_array, (kernel_size, kernel_size), sigma)
                
                elif noise_method == "Median Filter":
                    kernel_size = st.slider("Kernel Size", 1, 31, 5, 2)
                    
                    if kernel_size % 2 == 0:  # Ensure kernel size is odd
                        kernel_size += 1
                    
                    denoised = cv2.medianBlur(image_array, kernel_size)
                
                elif noise_method == "Non-Local Means Denoising":
                    h = st.slider("Filter Strength", 1, 30, 10)
                    
                    if len(image_array.shape) > 2:
                        denoised = cv2.fastNlMeansDenoisingColored(image_array, None, h, h, 7, 21)
                    else:
                        denoised = cv2.fastNlMeansDenoising(image_array, None, h, 7, 21)
                
                elif noise_method == "Wavelet-based Denoising":
                    if len(image_array.shape) > 2:
                        # Convert to grayscale for wavelet demonstration
                        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = image_array
                    
                    # Apply wavelet transform
                    coeffs = pywt.wavedec2(gray, 'haar', level=2)
                    
                    # Threshold percent
                    threshold_percent = st.slider("Threshold %", 10, 90, 30)
                    
                    # Threshold coefficients (keep approximation, threshold details)
                    threshold = np.percentile(np.abs(coeffs[1][0]), threshold_percent)
                    
                    new_coeffs = [coeffs[0]]  # Keep approximation
                    for detail_level in range(1, len(coeffs)):
                        level_coeffs = []
                        for detail_coeff in coeffs[detail_level]:
                            # Apply soft thresholding
                            thresholded = pywt.threshold(detail_coeff, threshold, 'soft')
                            level_coeffs.append(thresholded)
                        new_coeffs.append(tuple(level_coeffs))
                    
                    # Reconstruct image
                    denoised_gray = pywt.waverec2(new_coeffs, 'haar')
                    
                    # Resize to original dimensions
                    denoised_gray = denoised_gray[:gray.shape[0], :gray.shape[1]]
                    
                    if len(image_array.shape) > 2:
                        # For color images, create a colored version by applying the same operation
                        # to each channel for demonstration
                        denoised = np.zeros_like(image_array)
                        for i in range(3):
                            channel_coeffs = pywt.wavedec2(image_array[:,:,i], 'haar', level=2)
                            
                            channel_new_coeffs = [channel_coeffs[0]]
                            for detail_level in range(1, len(channel_coeffs)):
                                level_coeffs = []
                                for detail_coeff in channel_coeffs[detail_level]:
                                    thresholded = pywt.threshold(detail_coeff, threshold, 'soft')
                                    level_coeffs.append(thresholded)
                                channel_new_coeffs.append(tuple(level_coeffs))
                            
                            channel_denoised = pywt.waverec2(channel_new_coeffs, 'haar')
                            denoised[:,:,i] = channel_denoised[:image_array.shape[0], :image_array.shape[1]]
                        
                        denoised = np.clip(denoised, 0, 255).astype(np.uint8)
                    else:
                        denoised = np.clip(denoised_gray, 0, 255).astype(np.uint8)
                
                # Display denoised image
                st.image(denoised, caption=f"Denoised Image ({noise_method})", use_column_width=True)
                
                # Show difference
                if len(image_array.shape) > 2:
                    difference = cv2.absdiff(image_array, denoised)
                    difference_gray = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)
                else:
                    difference_gray = cv2.absdiff(image_array, denoised)
                
                # Enhance difference to make it visible
                difference_enhanced = cv2.equalizeHist(difference_gray)
                
                st.image(difference_enhanced, caption="Enhanced Difference (Removed Noise)", use_column_width=True)
                
                st.markdown(f"""
                ### Noise Reduction Analysis
                
                The {noise_method} technique has:
                
                - Reduced random noise patterns
                - {"Preserved star shapes well" if noise_method in ["Non-Local Means Denoising", "Wavelet-based Denoising"] else "Smoothed out some fine details"}
                - {"Maintained edges of larger structures" if noise_method in ["Non-Local Means Denoising", "Wavelet-based Denoising", "Median Filter"] else "Softened edges somewhat"}
                
                **Best practices for astronomical noise reduction:**
                
                1. **Median filters** are excellent for removing hot pixels and cosmic rays
                2. **Wavelet denoising** preserves structures at different scales
                3. **Non-local means** works well for preserving faint extended features
                4. **Stack multiple exposures** when possible for optimal results
                
                For scientific measurements, the uncertainty introduced by denoising must be accounted for.
                """)
        
        elif option == "Combined Analysis":
            st.markdown("""
            ## Combined Astronomical Image Analysis
            
            Many astronomical research tasks require combining multiple image processing techniques:
            
            1. **Initial preprocessing**: Calibration, noise reduction
            2. **Feature enhancement**: Bringing out faint structures
            3. **Object detection**: Finding stars, galaxies, and other objects
            4. **Measurement**: Photometry, astrometry, morphology
            5. **Storage optimization**: Compression for archiving
            
            Let's apply a complete workflow to your astronomical image.
            """)
            
            uploaded_image = st.file_uploader("Upload an astronomical image for complete analysis", type=["jpg", "png", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                image_array = np.array(image)
                
                # Display original image
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Apply noise reduction
                with st.expander("Step 1: Noise Reduction"):
                    noise_method = st.selectbox("Noise Reduction Method", ["Non-Local Means", "Median Filter"])
                    
                    if noise_method == "Non-Local Means":
                        if len(image_array.shape) > 2:
                            denoised = cv2.fastNlMeansDenoisingColored(image_array, None, 10, 10, 7, 21)
                        else:
                            denoised = cv2.fastNlMeansDenoising(image_array, None, 10, 7, 21)
                    else:
                        kernel_size = 5
                        denoised = cv2.medianBlur(image_array, kernel_size)
                    
                    st.image(denoised, caption="Denoised Image", use_column_width=True)
                
                # Enhance faint features
                with st.expander("Step 2: Enhance Faint Features"):
                    enhancement_method = st.selectbox("Enhancement Method", ["CLAHE", "Histogram Equalization"])
                    
                    if enhancement_method == "CLAHE":
                        enhanced = adaptive_histogram_equalization(denoised)
                    else:
                        if len(denoised.shape) > 2:
                            enhanced = np.zeros_like(denoised)
                            for i in range(3):
                                enhanced[:,:,i] = cv2.equalizeHist(denoised[:,:,i])
                        else:
                            enhanced = cv2.equalizeHist(denoised)
                    
                    st.image(enhanced, caption="Enhanced Image", use_column_width=True)
                
                # Detect stars/objects
                with st.expander("Step 3: Object Detection"):
                    detection_result, object_count = star_detection(enhanced)
                    st.image(detection_result, caption=f"Detected Objects: {object_count}", use_column_width=True)
                
                # Apply compression
                with st.expander("Step 4: Optimal Compression"):
                    compression_method = st.selectbox("Compression Method", ["High-Quality JPEG", "Lossless (PNG)"])
                    
                    if compression_method == "High-Quality JPEG":
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                        result, encimg = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR) if len(image_array.shape) > 2 else image_array, encode_param)
                        
                        original_size = image_array.size * image_array.itemsize
                        compressed_size = len(encimg)
                        
                        st.markdown(f"""
                        **JPEG Compression Results:**
                        - Original size: {original_size} bytes
                        - Compressed size: {compressed_size} bytes
                        - Compression ratio: {original_size / compressed_size:.2f}:1
                        """)
                    else:
                        result, encimg = cv2.imencode('.png', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR) if len(image_array.shape) > 2 else image_array)
                        
                        original_size = image_array.size * image_array.itemsize
                        compressed_size = len(encimg)
                        
                        st.markdown(f"""
                        **Lossless PNG Compression Results:**
                        - Original size: {original_size} bytes
                        - Compressed size: {compressed_size} bytes
                        - Compression ratio: {original_size / compressed_size:.2f}:1
                        """)
                
                # Final report
                st.subheader("Complete Analysis Report")
                
                st.markdown(f"""
                ### Image Analysis Summary
                
                **Original Image:**
                - Dimensions: {image_array.shape[0]} x {image_array.shape[1]} pixels
                - {'Color (RGB)' if len(image_array.shape) > 2 else 'Grayscale'} image
                
                **Processing Applied:**
                1. **Noise Reduction**: {noise_method}
                2. **Enhancement**: {enhancement_method}
                3. **Object Detection**: {object_count} objects found
                4. **Compression**: {compression_method} ({original_size / compressed_size:.2f}:1 ratio)
                
                **Scientific Information Preserved:**
                - {'High preservation of point sources (stars)' if noise_method == 'Non-Local Means' else 'Good preservation of point sources'}
                - {'Excellent enhancement of faint structures' if enhancement_method == 'CLAHE' else 'Good overall contrast enhancement'}
                - {'Lossless storage - all scientific data preserved' if compression_method == 'Lossless (PNG)' else 'High-quality lossy compression - minimal scientific data loss'}
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_column_width=True)
                with col2:
                    st.image(detection_result, caption="Final Processed Result", use_column_width=True)
    
    elif app_mode == "Video Frame Processing":
        st.title("Video Frame Processing")
        
        # Upload video
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
        
        if uploaded_video:
            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.getbuffer())
                    video_path = tmp_file.name
                
                # Display the video
                st.video(video_path)
                
                # Processing options
                col1, col2 = st.columns(2)
                with col1:
                    processing_technique = st.selectbox(
                        "Select Processing Technique",
                        [
                            "Original",
                            "Thresholding",
                            "Gray Level Slicing",
                            "Bit Plane Slicing",
                            "Low Pass Filter",
                            "High Pass Filter",
                            "High Boost Filter",
                            "Sobel Edge Detection",
                            "Prewitt Edge Detection",
                            "Robert Edge Detection",
                            "Laplacian",
                            "LoG (Laplacian of Gaussian)",
                            "Gaussian Blur"
                        ]
                    )
                
                with col2:
                    if processing_technique == "Thresholding":
                        threshold = st.slider("Threshold Value", 0, 255, 127)
                    elif processing_technique == "Gaussian Blur":
                        kernel_size = st.slider("Kernel Size", 1, 31, 5, 2)
                        sigma = st.slider("Sigma", 0.1, 10.0, 1.5, 0.1)
                
                # Frame extraction options
                st.markdown("### Frame Extraction Options")
                col1, col2 = st.columns(2)
                with col1:
                    extraction_mode = st.radio(
                        "Extraction Mode",
                        ["Uniform", "Key Frames"]
                    )
                with col2:
                    num_frames = st.slider("Number of Frames", 1, 20, 5)
                
                if st.button("Process Video Frames"):
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("Error: Could not open video file")
                    else:
                        # Get video properties
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Extract frames based on mode
                        frames = []
                        if extraction_mode == "Uniform":
                            positions = [int(i * frame_count / (num_frames + 1)) for i in range(1, num_frames + 1)]
                        else:  # Key Frames
                            prev_frame = None
                            positions = []
                            for i in range(frame_count):
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                if prev_frame is not None:
                                    diff = np.mean(np.abs(frame - prev_frame))
                                    if diff > 30:  # Threshold for key frame detection
                                        positions.append(i)
                                        if len(positions) >= num_frames:
                                            break
                                prev_frame = frame.copy()
                        
                        # Process the selected frames
                        processed_frames = []
                        for pos in positions:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                            ret, frame = cap.read()
                            if ret:
                                # Convert BGR to RGB for processing
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                # Apply selected processing technique
                                if processing_technique == "Original":
                                    processed = frame_rgb.copy()
                                elif processing_technique == "Thresholding":
                                    processed = process_thresholding(frame_rgb, threshold)
                                elif processing_technique == "Gray Level Slicing":
                                    processed = process_gray_level_slicing(frame_rgb)
                                elif processing_technique == "Bit Plane Slicing":
                                    processed = process_bit_plane_slicing(frame_rgb)
                                elif processing_technique == "Low Pass Filter":
                                    processed = process_low_pass(frame_rgb)
                                elif processing_technique == "High Pass Filter":
                                    processed = process_high_pass(frame_rgb)
                                elif processing_technique == "High Boost Filter":
                                    processed = process_high_boost(frame_rgb)
                                elif processing_technique == "Sobel Edge Detection":
                                    processed = process_sobel(frame_rgb)
                                elif processing_technique == "Prewitt Edge Detection":
                                    processed = process_prewitt(frame_rgb)
                                elif processing_technique == "Robert Edge Detection":
                                    processed = process_robert(frame_rgb)
                                elif processing_technique == "Laplacian":
                                    processed = process_laplacian(frame_rgb)
                                elif processing_technique == "LoG (Laplacian of Gaussian)":
                                    processed = process_log(frame_rgb)
                                elif processing_technique == "Gaussian Blur":
                                    processed = cv2.GaussianBlur(frame_rgb, (kernel_size, kernel_size), sigma)
                                
                                processed_frames.append(processed)
                        
                        cap.release()
                        
                        # Display processed frames
                        st.markdown("### Processed Frames")
                        # Calculate number of rows needed
                        num_cols = 3
                        num_rows = (len(processed_frames) + num_cols - 1) // num_cols
                        
                        # Display frames in a grid
                        for row in range(num_rows):
                            cols = st.columns(num_cols)
                            for col_idx in range(num_cols):
                                frame_idx = row * num_cols + col_idx
                                if frame_idx < len(processed_frames):
                                    with cols[col_idx]:
                                        st.image(processed_frames[frame_idx], 
                                                caption=f"Frame {positions[frame_idx]} (Time: {positions[frame_idx]/fps:.2f}s)")
                        
                        # Clean up temporary file
                        os.unlink(video_path)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if 'video_path' in locals():
                    try:
                        os.unlink(video_path)
                    except:
                        pass

if __name__ == "__main__":
    main()