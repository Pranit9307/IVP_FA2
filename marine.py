import streamlit as st
import cv2
import numpy as np
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
from skimage import exposure, color, restoration

def main():
    st.set_page_config(page_title="Underwater Image Compression for Marine Biology", layout="wide")
    
    st.title("Underwater Image Compression for Marine Biology")
    st.write("""
    This application helps marine biologists compress underwater imagery while preserving features crucial 
    for species identification. The app provides various compression techniques and image enhancement methods 
    specifically optimized for underwater conditions.
    """)
    
    # Sidebar for uploading and configuration
    with st.sidebar:
        st.header("Image Upload")
        uploaded_file = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])
        
        st.header("Compression Settings")
        compression_method = st.selectbox(
            "Compression Method",
            ["JPEG (Quality Based)", "PNG", "WebP", "JPEG 2000"]
        )
        
        if compression_method == "JPEG (Quality Based)":
            quality = st.slider("JPEG Quality", 1, 100, 75)
        elif compression_method == "WebP":
            quality = st.slider("WebP Quality", 1, 100, 80)
        elif compression_method == "JPEG 2000":
            compression_ratio = st.slider("Compression Ratio", 10, 100, 40)
        
        st.header("Image Enhancement")
        enhancement_options = st.multiselect(
            "Select Enhancement Methods",
            [
                "Contrast Limited Adaptive Histogram Equalization (CLAHE)",
                "White Balance Correction",
                "Blue Color Channel Emphasis",
                "Denoising",
                "Underwater Color Correction"
            ],
            default=["White Balance Correction"]
        )
        
        apply_enhancements_before = st.checkbox("Apply enhancements before compression", value=True)
        
        st.header("Analysis")
        show_histograms = st.checkbox("Show RGB Histograms", value=True)
        show_feature_analysis = st.checkbox("Show Feature Preservation Analysis", value=True)
    
    # Main content area
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Create columns for image display
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Original Image")
            st.image(image, use_column_width=True)
            
            original_size = len(uploaded_file.getvalue()) / 1024  # KB
            st.write(f"Original Size: {original_size:.2f} KB")
            
            if show_histograms:
                st.subheader("Original RGB Histogram")
                fig, ax = plt.subplots(figsize=(8, 4))
                display_rgb_histogram(img_array, ax)
                st.pyplot(fig)
        
        # Process image
        processed_img = img_array.copy()
        
        # Apply enhancements if requested before compression
        if apply_enhancements_before:
            processed_img = apply_enhancements(processed_img, enhancement_options)
        
        # Apply compression
        compressed_img, compressed_size = compress_image(
            processed_img, 
            method=compression_method, 
            quality=quality if compression_method in ["JPEG (Quality Based)", "WebP"] else None,
            compression_ratio=compression_ratio if compression_method == "JPEG 2000" else None
        )
        
        # Apply enhancements after compression if requested
        if not apply_enhancements_before:
            compressed_img = apply_enhancements(compressed_img, enhancement_options)
        
        with col2:
            st.header("Processed Image")
            st.image(compressed_img, use_column_width=True)
            
            compression_ratio_val = original_size / compressed_size
            st.write(f"Compressed Size: {compressed_size:.2f} KB")
            st.write(f"Compression Ratio: {compression_ratio_val:.2f}x")
            
            if show_histograms:
                st.subheader("Processed RGB Histogram")
                fig, ax = plt.subplots(figsize=(8, 4))
                display_rgb_histogram(compressed_img, ax)
                st.pyplot(fig)
        
        # Feature Preservation Analysis
        if show_feature_analysis:
            st.header("Feature Preservation Analysis")
            
            # Calculate metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # PSNR (Peak Signal-to-Noise Ratio)
                psnr_value = calculate_psnr(img_array, compressed_img)
                st.metric("PSNR (Higher is better)", f"{psnr_value:.2f} dB")
                
                # SSIM (Structural Similarity Index)
                ssim_value = calculate_ssim(img_array, compressed_img)
                st.metric("SSIM (Higher is better)", f"{ssim_value:.4f}")
            
            with col2:
                # Edge Preservation
                edge_score = edge_preservation_metric(img_array, compressed_img)
                st.metric("Edge Preservation (Higher is better)", f"{edge_score:.4f}")
                
                # Color Fidelity
                color_fidelity = color_fidelity_metric(img_array, compressed_img)
                st.metric("Color Fidelity (Lower is better)", f"{color_fidelity:.4f}")
            
            # Display edge detection comparison
            st.subheader("Edge Detection Comparison")
            col1, col2 = st.columns(2)
            
            original_edges = detect_edges(img_array)
            compressed_edges = detect_edges(compressed_img)
            
            with col1:
                st.write("Original Image Edges")
                st.image(original_edges, use_column_width=True)
            
            with col2:
                st.write("Compressed Image Edges")
                st.image(compressed_edges, use_column_width=True)

def apply_enhancements(image, enhancement_options):
    """Apply selected image enhancement techniques"""
    result = image.copy()
    
    # Convert to float for processing if needed
    if result.dtype != np.float32:
        result = result.astype(np.float32) / 255.0
    
    if "Contrast Limited Adaptive Histogram Equalization (CLAHE)" in enhancement_options:
        # Convert to LAB color space
        if len(result.shape) == 3 and result.shape[2] == 3:
            if result.max() <= 1.0:
                lab = color.rgb2lab(result)
            else:
                lab = color.rgb2lab(result / 255.0)
            
            # Apply CLAHE to L channel
            l_channel = lab[:,:,0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(np.uint8(l_channel * 255 / 100)) / 255.0 * 100
            lab[:,:,0] = cl
            
            # Convert back to RGB
            result = color.lab2rgb(lab)
            if image.dtype == np.uint8:
                result = (result * 255).astype(np.uint8)
    
    if "White Balance Correction" in enhancement_options:
        if len(result.shape) == 3:
            if result.max() > 1.0:
                result = result / 255.0
                
            # Simple grey world assumption
            avg_r = np.mean(result[:,:,0])
            avg_g = np.mean(result[:,:,1])
            avg_b = np.mean(result[:,:,2])
            avg_gray = (avg_r + avg_g + avg_b) / 3
            
            result[:,:,0] = np.clip(result[:,:,0] * (avg_gray / avg_r), 0, 1)
            result[:,:,1] = np.clip(result[:,:,1] * (avg_gray / avg_g), 0, 1)
            result[:,:,2] = np.clip(result[:,:,2] * (avg_gray / avg_b), 0, 1)
            
            if image.dtype == np.uint8:
                result = (result * 255).astype(np.uint8)
    
    if "Blue Color Channel Emphasis" in enhancement_options:
        if len(result.shape) == 3:
            if result.max() <= 1.0 and image.dtype == np.uint8:
                result = (result * 255).astype(np.uint8)
            
            # Increase the weight of the blue channel to compensate for underwater attenuation
            # Normalize other channels to maintain overall brightness
            b_gain = 1.2  # Blue gain
            result = result.astype(np.float32)
            result[:,:,2] = np.clip(result[:,:,2] * b_gain, 0, 255 if result.max() > 1.0 else 1.0)
            
            if image.dtype == np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)
    
    if "Denoising" in enhancement_options:
        if result.max() <= 1.0 and image.dtype == np.uint8:
            result = (result * 255).astype(np.uint8)
            
        # Non-local means denoising
        if len(result.shape) == 3:
            result = cv2.fastNlMeansDenoisingColored(
                np.uint8(result) if result.dtype != np.uint8 else result, 
                None, 10, 10, 7, 21
            )
        else:
            result = cv2.fastNlMeansDenoising(
                np.uint8(result) if result.dtype != np.uint8 else result
            )
            
        if image.dtype != np.uint8:
            result = result.astype(np.float32) / 255.0
    
    if "Underwater Color Correction" in enhancement_options:
        if len(result.shape) == 3:
            if result.max() > 1.0:
                result = result / 255.0
                
            # Underwater image color correction - Red channel compensation
            # Underwater images typically lose red wavelengths first
            min_r = np.min(result[:,:,0])
            min_g = np.min(result[:,:,1])
            min_b = np.min(result[:,:,2])
            
            result[:,:,0] = np.clip((result[:,:,0] - min_r) * 1.3, 0, 1)
            result[:,:,1] = np.clip(result[:,:,1] - min_g, 0, 1)
            result[:,:,2] = np.clip(result[:,:,2] - min_b, 0, 1)
            
            # Adjust overall saturation
            hsv = color.rgb2hsv(result)
            hsv[:,:,1] = hsv[:,:,1] * 1.2  # Increase saturation
            result = color.hsv2rgb(hsv)
            
            if image.dtype == np.uint8:
                result = (result * 255).astype(np.uint8)
    
    # Final conversion to original dtype
    if image.dtype == np.uint8 and result.max() <= 1.0:
        result = (result * 255).astype(np.uint8)
    
    return result

def compress_image(image, method, quality=None, compression_ratio=None):
    """Compress the image using the specified method"""
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create a bytes buffer for the compressed image
    buffer = io.BytesIO()
    
    if method == "JPEG (Quality Based)":
        # Convert to RGB if the image has an alpha channel
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        pil_img = Image.fromarray(image)
        pil_img.save(buffer, format="JPEG", quality=quality)
    
    elif method == "PNG":
        pil_img = Image.fromarray(image)
        pil_img.save(buffer, format="PNG", optimize=True)
    
    elif method == "WebP":
        pil_img = Image.fromarray(image)
        pil_img.save(buffer, format="WebP", quality=quality)
    
    elif method == "JPEG 2000":
        # Implement JPEG 2000 compression (using OpenCV)
        # Compute compression parameter based on compression ratio
        compression_param = 100 - compression_ratio
        # Scale to OpenCV's expected range (0-9 where 0 is lossless)
        cv_param = int(compression_param / 10)
        
        # Convert to RGB if the image has an alpha channel
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # OpenCV requires specific format for file name in JPEG 2000
        temp_filename = "temp.jp2"
        cv2.imwrite(temp_filename, image, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, cv_param * 1000])
        
        # Read back the file and get its size
        with open(temp_filename, "rb") as f:
            buffer.write(f.read())
        
        # Clean up the temporary file
        import os
        os.remove(temp_filename)
    
    # Read the compressed image back
    buffer.seek(0)
    compressed_size = len(buffer.getvalue()) / 1024  # KB
    
    # Convert buffer back to image
    compressed_img = Image.open(buffer)
    compressed_img = np.array(compressed_img)
    
    return compressed_img, compressed_size

def display_rgb_histogram(image, ax):
    """Display RGB histogram of an image"""
    if len(image.shape) == 3:  # Color image
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
        ax.set_xlim([0, 256])
        ax.set_title('RGB Histogram')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
    else:  # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='gray')
        ax.set_xlim([0, 256])
        ax.set_title('Grayscale Histogram')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')

def calculate_psnr(original, compressed):
    """Calculate Peak Signal-to-Noise Ratio"""
    if original.dtype != np.float32:
        original = original.astype(np.float32)
    if compressed.dtype != np.float32:
        compressed = compressed.astype(np.float32)
    
    # Ensure images have the same shape and type
    if original.shape != compressed.shape:
        # Resize compressed to match original
        compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
    
    # Calculate PSNR
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100  # Perfect match
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, compressed):
    """Calculate Structural Similarity Index"""
    # Convert to grayscale if color
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        
    if len(compressed.shape) == 3:
        compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)
    else:
        compressed_gray = compressed
    
    # Ensure images have the same shape
    if original_gray.shape != compressed_gray.shape:
        compressed_gray = cv2.resize(compressed_gray, (original_gray.shape[1], original_gray.shape[0]))
    
    # Calculate SSIM
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    
    original_gray = original_gray.astype(np.float32)
    compressed_gray = compressed_gray.astype(np.float32)
    
    mu1 = cv2.GaussianBlur(original_gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(compressed_gray, (11, 11), 1.5)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(original_gray * original_gray, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(compressed_gray * compressed_gray, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original_gray * compressed_gray, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return np.mean(ssim_map)

def detect_edges(image):
    """Detect edges in an image using Canny edge detector"""
    if len(image.shape) == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def edge_preservation_metric(original, compressed):
    """Calculate edge preservation metric"""
    original_edges = detect_edges(original)
    compressed_edges = detect_edges(compressed)
    
    # Ensure same size
    if original_edges.shape != compressed_edges.shape:
        compressed_edges = cv2.resize(compressed_edges, (original_edges.shape[1], original_edges.shape[0]))
    
    # Calculate edge preservation ratio (intersection over union)
    intersection = np.logical_and(original_edges, compressed_edges)
    union = np.logical_or(original_edges, compressed_edges)
    
    if np.sum(union) == 0:
        return 1.0  # Perfect match if no edges detected in either image
    
    iou = np.sum(intersection) / np.sum(union)
    return iou

def color_fidelity_metric(original, compressed):
    """Calculate color fidelity metric (mean color difference)"""
    if len(original.shape) != 3 or len(compressed.shape) != 3:
        return 0.0  # Not applicable to grayscale images
    
    # Ensure same size
    if original.shape != compressed.shape:
        compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
    
    # Convert to float for processing
    original = original.astype(np.float32)
    compressed = compressed.astype(np.float32)
    
    # Calculate Euclidean distance in RGB space
    diff = np.sqrt(np.sum((original - compressed) ** 2, axis=2))
    mean_diff = np.mean(diff) / np.sqrt(3 * 255**2)  # Normalize to [0,1]
    
    return mean_diff

if __name__ == "__main__":
    main()