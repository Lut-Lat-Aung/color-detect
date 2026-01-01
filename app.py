import cv2
import numpy as np
import streamlit as st
from PIL import Image

from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Color Detector", layout="wide")

# ---------- Utilities ----------
def bgr_to_hex(bgr: np.ndarray) -> str:
    b, g, r = [int(x) for x in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"

def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    bgr_1px = np.uint8([[bgr]])
    hsv = cv2.cvtColor(bgr_1px, cv2.COLOR_BGR2HSV)[0][0]
    return hsv  # H: 0-179, S: 0-255, V: 0-255

def basic_color_name_from_hsv(hsv: np.ndarray) -> str:
    h, s, v = [int(x) for x in hsv]

    # low saturation => gray/white/black
    if s < 35:
        if v < 50:
            return "Black"
        elif v > 200:
            return "White"
        else:
            return "Gray"

    # low value => dark
    if v < 50:
        return "Black"

    # Hue buckets (OpenCV hue 0-179)
    # red wraps around
    if h < 10 or h >= 170:
        return "Red"
    if 10 <= h < 25:
        return "Orange"
    if 25 <= h < 35:
        return "Yellow"
    if 35 <= h < 85:
        return "Green"
    if 85 <= h < 100:
        return "Cyan"
    if 100 <= h < 130:
        return "Blue"
    if 130 <= h < 160:
        return "Purple"
    if 160 <= h < 170:
        return "Pink"
    return "Unknown"

def dominant_color_bgr(image_bgr: np.ndarray, k: int = 3) -> np.ndarray:
   
    img = image_bgr.copy()
    # downscale for speed
    h, w = img.shape[:2]
    scale = 300 / max(h, w) if max(h, w) > 300 else 1.0
    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    pixels = img.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()

    # find most frequent cluster
    counts = np.bincount(labels)
    dom = centers[np.argmax(counts)]
    return dom.astype(np.uint8)

def show_color_info(title: str, bgr: np.ndarray):
    hsv = bgr_to_hsv(bgr)
    hex_color = bgr_to_hex(bgr)
    name = basic_color_name_from_hsv(hsv)

    b, g, r = [int(x) for x in bgr]
    h, s, v = [int(x) for x in hsv]

    st.markdown(f"### {title}")
    st.markdown(
        f"""
        <div style="display:flex; gap:14px; align-items:center;">
          <div style="width:60px;height:60px;border-radius:12px;border:1px solid #ddd;background:{hex_color};"></div>
          <div>
            <div><b>Name:</b> {name}</div>
            <div><b>HEX:</b> {hex_color}</div>
            <div><b>RGB:</b> ({r}, {g}, {b})</div>
            <div><b>HSV:</b> (H={h}, S={s}, V={v})</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Webcam processor ----------
class ColorPickerProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_bgr = None
        self.pick_x = None
        self.pick_y = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        if self.pick_x is not None and self.pick_y is not None:
            x, y = self.pick_x, self.pick_y
            h, w = img_bgr.shape[:2]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))

            r = 3
            x1, x2 = max(0, x - r), min(w, x + r + 1)
            y1, y2 = max(0, y - r), min(h, y + r + 1)
            patch = img_bgr[y1:y2, x1:x2]
            avg = patch.reshape(-1, 3).mean(axis=0).astype(np.uint8)
            self.last_bgr = avg

            # draw crosshair
            cv2.drawMarker(img_bgr, (x, y), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


st.title("🎨 Color Detector (Python + OpenCV + Streamlit)")

tab1, tab2 = st.tabs(["📷 Image Upload", "🎥 Webcam (Live)"])

with tab1:
    st.subheader("1) Upload an image and click on it to detect the color")

    uploaded = st.file_uploader("Upload image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
    colA, colB = st.columns([1.2, 1])

    if uploaded is None:
        st.info("Upload an image to begin.")
    else:
        pil_img = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        with colA:
            st.caption("Click on the image to pick a color.")
            click = streamlit_image_coordinates(pil_img, key="img_coords", use_column_width=True)
            
        with colB:
            st.caption("Detection results")
            if click is not None and "x" in click and "y" in click:
                # original image size
                orig_w, orig_h = pil_img.size

                # streamlit-image-coordinates often returns displayed image width/height too (depends on version)
                disp_w = click.get("width", orig_w)
                disp_h = click.get("height", orig_h)

                # map displayed coords -> original coords
                x = int(click["x"] * (orig_w / disp_w))
                y = int(click["y"] * (orig_h / disp_h))

                # clamp to bounds
                x = max(0, min(orig_w - 1, x))
                y = max(0, min(orig_h - 1, y))

                # sample average color in a small patch for stability
                r = st.slider("Sampling radius (pixels)", 1, 15, 5)
                h, w = img_bgr.shape[:2]
                x1, x2 = max(0, x - r), min(w, x + r + 1)
                y1, y2 = max(0, y - r), min(h, y + r + 1)
                patch = img_bgr[y1:y2, x1:x2]
                avg_bgr = patch.reshape(-1, 3).mean(axis=0).astype(np.uint8)

                show_color_info("Picked Color", avg_bgr)
            else:
                st.write("Click the image to detect a color.")

            st.divider()
            st.subheader("2) Dominant color (K-means)")
            k = st.slider("Number of clusters (k)", 2, 8, 3)
            dom = dominant_color_bgr(img_bgr, k=k)
            show_color_info("Dominant Color", dom)

with tab2:
    st.subheader("Webcam live color detection (pick a point)")
    st.write("Start the webcam. Then click a point inside the video to sample the color.")


    ctx = webrtc_streamer(
        key="color-webcam",
        video_processor_factory=ColorPickerProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        st.markdown("### Pick point (X, Y) on the current frame")
        st.caption("Tip: Start with center. Adjust until it points at the object color you want.")
        x = st.number_input("X", min_value=0, max_value=2000, value=320, step=1)
        y = st.number_input("Y", min_value=0, max_value=2000, value=240, step=1)

        ctx.video_processor.pick_x = int(x)
        ctx.video_processor.pick_y = int(y)

        if ctx.video_processor.last_bgr is not None:
            show_color_info("Webcam Picked Color", ctx.video_processor.last_bgr)
        else:
            st.info("Waiting for video frame…")

st.caption("Made with Streamlit + OpenCV.")
