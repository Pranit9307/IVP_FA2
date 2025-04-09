import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import base64
from io import BytesIO

# Set page config with proper theme
st.set_page_config(
    page_title="Weather Satellite Analysis",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better contrast and layout
st.markdown("""
<style>
    /* Main content area */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* Text */
    p, div, span {
        color: #E0E0E0 !important;
    }
    
    /* Images container */
    .image-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    
    /* Individual image styling */
    .image-wrapper {
        width: 48%;
        text-align: center;
    }
    
    /* Sidebar */
    .css-1aumxhk {
        background-color: #1A1D23 !important;
    }
    
    /* File uploader */
    .stFileUploader label {
        color: #FFFFFF !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    
    /* Sliders */
    .stSlider label {
        color: #FFFFFF !important;
    }
    
    /* Select boxes */
    .stSelectbox label {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

def extract_frames(video_path, num_frames=5):
    """Extract frames from video"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            st.error("Video contains no frames")
            return []
            
        interval = max(1, total_frames // num_frames)
        frames = []

        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return []

def apply_processing(image_array, technique):
    """Apply image processing techniques"""
    if technique == "None":
        return image_array

    try:
        if technique == "Contrast Enhancement":
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        elif technique == "Edge Detection (Sobel)":
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.magnitude(sobelx, sobely)
            return cv2.cvtColor(np.uint8(np.clip(edges, 0, 255)), cv2.COLOR_GRAY2RGB)

        elif technique == "Histogram Equalization":
            ycrcb = cv2.cvtColor(image_array, cv2.COLOR_RGB2YCrCb)
            channels = cv2.split(ycrcb)
            cv2.equalizeHist(channels[0], channels[0])
            cv2.merge(channels, ycrcb)
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        elif technique == "Cloud Segmentation (Thresholding)":
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        elif technique == "False Color Mapping":
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    except Exception as e:
        st.error(f"Error in image processing: {str(e)}")
        return image_array

    return image_array

def display_images_side_by_side(original, processed, caption1="Original", caption2="Processed"):
    """Display two images side by side with proper layout"""
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption=caption1, use_column_width=True)
    with col2:
        st.image(processed, caption=caption2, use_column_width=True)

def calculate_metrics(original, processed):
    """Calculate image quality metrics"""
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Convert to grayscale for metrics calculation
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    
    # Calculate MSE and PSNR
    mse = np.mean((original_gray - processed_gray) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    return {
        "MSE": mse,
        "PSNR": psnr
    }

def main():
    st.title("🌤️ Weather Satellite Image Compression & Analysis")
    st.markdown("""
    In meteorological forecasting, rapid transmission of satellite imagery is crucial while preserving critical features like cloud formations.
    """)

    # Sidebar navigation
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/GOES-16_first_light_anim.gif/320px-GOES-16_first_light_anim.gif", 
                use_column_width=True)
        option = st.selectbox("Select Operation", [
            "Image Processing Techniques",
            "Video Processing Techniques",
            "Lossless Compression",
            "Lossy Compression",
            "Transform Compression"
        ])
        st.markdown("---")
        st.markdown("**About**")
        st.info("Tool for analyzing weather satellite imagery")

    if option == "Image Processing Techniques":
        st.header("Image Processing Techniques")
        uploaded_image = st.file_uploader("Upload weather satellite image", 
                                        type=["jpg", "jpeg", "png", "tif"], 
                                        key="processing")
        
        if uploaded_image:
            original_image = Image.open(uploaded_image).convert("RGB")
            image_array = np.array(original_image)
            
            technique = st.selectbox("Select Processing Technique", [
                "None",
                "Contrast Enhancement",
                "Edge Detection (Sobel)",
                "Histogram Equalization",
                "Cloud Segmentation (Thresholding)",
                "False Color Mapping"
            ])
            
            if st.button("Process Image"):
                processed = apply_processing(image_array, technique)
                display_images_side_by_side(original_image, processed)
                
                # Calculate and display metrics
                metrics = calculate_metrics(image_array, processed)
                st.markdown("### Quality Metrics")
                st.write(f"**MSE (Lower is better):** {metrics['MSE']:.2f}")
                st.write(f"**PSNR (Higher is better):** {metrics['PSNR']:.2f} dB")
                
                # Download button
                processed_pil = Image.fromarray(processed)
                buffered = BytesIO()
                processed_pil.save(buffered, format="PNG")
                st.download_button(
                    label="Download Processed Image",
                    data=buffered.getvalue(),
                    file_name="processed.png",
                    mime="image/png"
                )

    elif option == "Video Processing Techniques":
        st.header("Video Processing")
        uploaded_video = st.file_uploader("Upload satellite video", 
                                        type=["mp4", "avi", "mov"], 
                                        key="video")
        
        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                tmpfile.write(uploaded_video.read())
                video_path = tmpfile.name

            st.video(uploaded_video)
            
            num_frames = st.slider("Number of Frames to Extract", 1, 10, 3)
            technique = st.selectbox("Processing Technique", [
                "None",
                "Contrast Enhancement",
                "Edge Detection (Sobel)",
                "Histogram Equalization"
            ])

            if st.button("Process Video Frames"):
                frames = extract_frames(video_path, num_frames)
                if frames:
                    st.markdown(f"### Processed Frames ({technique})")
                    for i, frame in enumerate(frames):
                        original = Image.fromarray(frame)
                        processed = apply_processing(frame, technique)
                        display_images_side_by_side(
                            original, 
                            processed,
                            f"Frame {i+1} - Original",
                            f"Frame {i+1} - Processed"
                        )

    elif option == "Lossless Compression":
        st.header("Lossless Compression")
        uploaded_image = st.file_uploader("Upload weather satellite image", 
                                        type=["jpg", "png", "jpeg"], 
                                        key="lossless")
        if uploaded_image:
            original_image = Image.open(uploaded_image)
            image_array = np.array(original_image)
            
            display_images_side_by_side(
                original_image,
                original_image,
                "Original Image",
                "Lossless Compression Preview (No visible change)"
            )

            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            result, encimg = cv2.imencode('.png', image_bgr, encode_param)
            
            original_size = len(uploaded_image.getvalue())
            compressed_size = len(encimg)
            
            st.markdown("### Compression Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Size", f"{original_size/1024:.2f} KB")
            col2.metric("Compressed Size", f"{compressed_size/1024:.2f} KB")
            col3.metric("Compression Ratio", f"{original_size/compressed_size:.2f}:1")
            
            st.download_button(
                label="Download Compressed Image",
                data=encimg.tobytes(),
                file_name="compressed.png",
                mime="image/png"
            )

    elif option == "Lossy Compression":
        st.header("Lossy Compression")
        uploaded_image = st.file_uploader("Upload weather satellite image", 
                                        type=["jpg", "png", "jpeg"], 
                                        key="lossy")
        if uploaded_image:
            original_image = Image.open(uploaded_image)
            image_array = np.array(original_image)
            
            quality_factor = st.slider("JPEG Quality Factor", 1, 100, 85)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
            result, encimg = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR), encode_param)
            decimg = cv2.imdecode(encimg, 1)
            processed_image = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
            
            display_images_side_by_side(
                original_image,
                processed_image,
                "Original Image",
                f"Compressed (JPEG Quality: {quality_factor})"
            )

            original_size = len(uploaded_image.getvalue())
            compressed_size = len(encimg)
            metrics = calculate_metrics(image_array, processed_image)
            
            st.markdown("### Compression Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Size", f"{original_size/1024:.2f} KB")
            col2.metric("Compressed Size", f"{compressed_size/1024:.2f} KB")
            col3.metric("Compression Ratio", f"{original_size/compressed_size:.2f}:1")
            
            st.markdown("### Quality Metrics")
            st.write(f"**PSNR:** {metrics['PSNR']:.2f} dB (Higher is better)")
            
            st.download_button(
                label="Download Compressed Image",
                data=encimg.tobytes(),
                file_name="compressed.jpg",
                mime="image/jpeg"
            )

    elif option == "Transform Compression":
        st.header("Transform Domain Compression")
        uploaded_image = st.file_uploader("Upload weather satellite image", 
                                        type=["jpg", "png", "jpeg"], 
                                        key="transform")
        if uploaded_image:
            original_image = Image.open(uploaded_image).convert("L")
            image_array = np.array(original_image)
            
            st.image(original_image, caption="Original Grayscale Image", use_column_width=True)
            
            st.markdown("### DCT-Based Compression")
            image_float = np.float32(image_array) / 255.0
            dct = cv2.dct(image_float)

            quant_factor = st.slider("DCT Quantization Factor", 1, 50, 10)
            quantized = np.round(dct / quant_factor) * quant_factor
            idct = cv2.idct(quantized)
            idct_image = np.uint8(np.clip(idct * 255, 0, 255))
            
            display_images_side_by_side(
                original_image,
                idct_image,
                "Original Image",
                "Reconstructed Image after DCT Compression"
            )
            
            metrics = calculate_metrics(
                np.stack([image_array]*3, axis=-1),
                np.stack([idct_image]*3, axis=-1)
            )
            
            st.markdown("### Reconstruction Quality")
            st.write(f"**PSNR:** {metrics['PSNR']:.2f} dB")

if __name__ == "__main__":
    main()