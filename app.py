import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import threading

from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Vision Studio",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎥 AI Vision Studio</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-Time AI Background Segmentation & Replacement'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Control Panel")

st.sidebar.markdown("---")

st.sidebar.subheader("🎨 Background")

background_mode = st.sidebar.selectbox(
    "Background Mode",
    [
        "Green",
        "Blue",
        "Red",
        "Black",
        "White",
        "Blur",
        "Custom Image"
    ]
)


# ============================================================
# COLORS
# ============================================================

color_map = {
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Red": (0, 0, 255),
    "Black": (0, 0, 0),
    "White": (255, 255, 255)
}


# ============================================================
# BLUR CONTROL
# ============================================================

blur_strength = 31

if background_mode == "Blur":

    st.sidebar.markdown("### 🌫️ Blur Settings")

    blur_strength = st.sidebar.slider(
        "Blur Strength",
        min_value=5,
        max_value=61,
        value=31,
        step=2
    )


# ============================================================
# CUSTOM IMAGE
# ============================================================

uploaded_file = None

if background_mode == "Custom Image":

    st.sidebar.markdown("### 🖼️ Custom Background")

    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )


# ============================================================
# LOAD CUSTOM IMAGE
# ============================================================

custom_background = None

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    custom_background = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if custom_background is not None:

        st.sidebar.image(
            cv2.cvtColor(
                custom_background,
                cv2.COLOR_BGR2RGB
            ),
            caption="Background Preview",
            use_container_width=True
        )


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class BackgroundRemover(VideoProcessorBase):

    def __init__(self):

        # ----------------------------------------------------
        # Thread lock
        # ----------------------------------------------------

        self.lock = threading.Lock()

        # ----------------------------------------------------
        # Initial settings
        # ----------------------------------------------------

        self.background_mode = "Green"

        self.selected_color = color_map["Green"]

        self.custom_background = None

        self.blur_strength = 31

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        options = mp.tasks.vision.ImageSegmenterOptions(

            base_options=mp.tasks.BaseOptions(
                model_asset_path="selfie_segmenter.tflite"
            ),

            output_category_mask=True,

            output_confidence_masks=False
        )

        self.segmenter = (
            mp.tasks.vision.ImageSegmenter.create_from_options(
                options
            )
        )


    # ========================================================
    # UPDATE SETTINGS
    # ========================================================

    def update_settings(
        self,
        background_mode,
        selected_color,
        blur_strength,
        custom_background
    ):

        with self.lock:

            self.background_mode = background_mode

            self.selected_color = selected_color

            self.blur_strength = blur_strength

            if custom_background is not None:

                self.custom_background = (
                    custom_background.copy()
                )

            else:

                self.custom_background = None


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def recv(self, frame: av.VideoFrame):

        # ----------------------------------------------------
        # Receive frame
        # ----------------------------------------------------

        img = frame.to_ndarray(
            format="bgr24"
        )

        img = cv2.flip(
            img,
            1
        )


        # ----------------------------------------------------
        # Safely read settings
        # ----------------------------------------------------

        with self.lock:

            mode = self.background_mode

            color = self.selected_color

            blur = self.blur_strength

            custom = self.custom_background


        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ----------------------------------------------------
        # AI segmentation
        # ----------------------------------------------------

        result = self.segmenter.segment(
            mp_image
        )


        # ----------------------------------------------------
        # Get mask
        # ----------------------------------------------------

        mask = result.category_mask.numpy_view()

        mask = cv2.resize(
            mask,
            (
                img.shape[1],
                img.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )


        # ----------------------------------------------------
        # Person mask
        # ----------------------------------------------------

        person_mask = np.where(
            mask == 0,
            255,
            0
        ).astype(np.uint8)


        # ----------------------------------------------------
        # Smooth mask
        # ----------------------------------------------------

        person_mask = cv2.GaussianBlur(
            person_mask,
            (7, 7),
            0
        )


        # ====================================================
        # CREATE BACKGROUND
        # ====================================================

        if mode == "Blur":

            # Make sure kernel is odd
            kernel = blur

            if kernel % 2 == 0:
                kernel += 1

            background = cv2.GaussianBlur(
                img,
                (kernel, kernel),
                0
            )


        elif mode == "Custom Image":

            if custom is not None:

                background = cv2.resize(
                    custom,
                    (
                        img.shape[1],
                        img.shape[0]
                    )
                )

            else:

                background = np.zeros_like(
                    img
                )

                background[:] = (
                    50,
                    50,
                    50
                )


        else:

            background = np.zeros_like(
                img
            )

            background[:] = color


        # ====================================================
        # ALPHA BLENDING
        # ====================================================

        alpha = (
            person_mask.astype(
                np.float32
            ) / 255.0
        )

        alpha = alpha[:, :, np.newaxis]


        output = (
            img.astype(np.float32) * alpha
            +
            background.astype(np.float32)
            * (1 - alpha)
        ).astype(np.uint8)


        # ====================================================
        # RETURN FRAME
        # ====================================================

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )


# ============================================================
# WEBRTC
# ============================================================

webrtc_ctx = webrtc_streamer(

    key="ai-vision-studio",

    video_processor_factory=BackgroundRemover,

    media_stream_constraints={
        "video": True,
        "audio": False
    }
)


# ============================================================
# UPDATE RUNNING PROCESSOR
# ============================================================

if webrtc_ctx.video_processor is not None:

    if background_mode in color_map:

        selected_color = color_map[
            background_mode
        ]

    else:

        selected_color = (
            0,
            0,
            0
        )


    webrtc_ctx.video_processor.update_settings(

        background_mode=background_mode,

        selected_color=selected_color,

        blur_strength=blur_strength,

        custom_background=custom_background
    )


# ============================================================
# AI SYSTEM INFORMATION
# ============================================================

st.markdown("---")

st.subheader("🤖 AI System Information")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        label="AI MODEL",
        value="🟢 Selfie Segmenter"
    )


with col2:

    st.metric(
        label="PROCESSING",
        value="🟢 Real-Time"
    )


with col3:

    st.metric(
        label="BACKGROUND",
        value=f"🎨 {background_mode}"
    )


with col4:

    st.metric(
        label="ENGINE",
        value="⚡ MediaPipe"
    )


# ============================================================
# ABOUT
# ============================================================

with st.expander("ℹ️ About this project"):

    st.markdown("""
### AI Vision Studio

A real-time computer vision application that uses **MediaPipe
Image Segmentation** to separate a person from their background.

### Features

- 📷 Real-time webcam processing
- 🤖 AI human segmentation
- 🎨 Solid-color backgrounds
- 🌫️ Adjustable background blur
- 🖼️ Custom background images
- ⚡ Real-time video processing

### Technology Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Streamlit
- Streamlit-WebRTC

### Processing Pipeline

**Webcam → WebRTC → OpenCV → MediaPipe → Segmentation Mask →
Background Processing → Final Video**
""")