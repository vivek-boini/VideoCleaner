import os
import sys
import streamlit as st
from ultralytics import YOLO

sys.path.append("src")

from processor import process_video


INPUT_DIR = "temp"
OUTPUT_DIR = "output"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


st.set_page_config(
    page_title="Video Cleaner",
    page_icon="🎬",
    layout="centered"
)


st.title("🎬 Video Cleaner")
st.caption(
    "Remove audio and automatically remove sections containing "
    "people and animals."
)

st.divider()


uploaded_files = st.file_uploader(
    "📤 Upload your videos",
    type=["mp4", "mov", "avi", "mkv"],
    accept_multiple_files=True,
    help="You can select multiple videos for batch processing."
)


if uploaded_files:

    st.subheader("📋 Selected Videos")

    for file in uploaded_files:
        file_size = file.size / (1024 * 1024)

        st.write(
            f"🎥 **{file.name}** — {file_size:.1f} MB"
        )

    st.divider()

    if st.button(
        "🧹 Clean Videos",
        type="primary",
        use_container_width=True
    ):

        model = load_model()

        st.subheader("⚙️ Processing")

        for index, file in enumerate(uploaded_files, 1):

            st.write(
                f"### {index}. {file.name}"
            )

            input_path = os.path.join(
                INPUT_DIR,
                file.name
            )

            base_name = os.path.splitext(
                file.name
            )[0]

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{base_name}_cleaned.mp4"
            )

            with open(input_path, "wb") as f:
                f.write(file.getbuffer())

            progress = st.progress(0)

            status = st.empty()

            status.info(
                "🔍 Detecting people and animals..."
            )

            progress.progress(30)

            try:

                success, duration = process_video(
                    input_path,
                    output_path,
                    model
                )

                progress.progress(90)

                if success and os.path.exists(output_path):

                    progress.progress(100)

                    status.success(
                        f"✅ Completed — original duration "
                        f"{duration:.1f}s"
                    )

                    st.video(output_path)

                    with open(output_path, "rb") as f:
                        video_data = f.read()

                    st.download_button(
                        "⬇️ Download Cleaned Video",
                        data=video_data,
                        file_name=os.path.basename(
                            output_path
                        ),
                        mime="video/mp4",
                        use_container_width=True,
                        key=f"download_{index}"
                    )

                else:

                    progress.empty()

                    status.error(
                        "❌ Failed to create cleaned video."
                    )

            except Exception as e:

                progress.empty()

                status.error(
                    f"❌ Error: {str(e)}"
                )

            st.divider()


else:

    st.info(
        "👆 Upload one or more videos to get started."
    )