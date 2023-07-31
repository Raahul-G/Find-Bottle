import streamlit as st
import torch
from detect import run
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time
import cv2


def imageInput(src):
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            imgpath = os.path.join('data/uploads', image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())
            # call Model prediction--
            model = torch.hub.load('.', 'custom', 'best.pt', source='local')
            model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            # --Display predicton

            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')

    elif src == 'From test set.':
        # Image selector slider
        imgpath = glob.glob('data/images/*')
        imgsel = st.slider('Select random images from test set.', min_value=1, max_value=len(imgpath), step=1)
        image_file = imgpath[imgsel - 1]
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:
            if image_file is not None and submit:
                # call Model prediction--
                model = torch.hub.load('.', 'custom', 'best.pt', source='local')
                pred = model(image_file)
                pred.render()  # render bbox in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                    # --Display predicton
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='Model Prediction(s)')


# Function to read and video
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, 60.0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


def videoInput(src):
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video != None:
        imgpath = os.path.join('data/uploads', uploaded_video.name)
        outputpath = os.path.join('runs/detect/exp', os.path.basename(imgpath))
        with open(imgpath, mode='wb') as f:
            f.write(uploaded_video.read())  # save video to disk

        # Display video using OpenCV
        read_video(imgpath)
        run(weights="best.pt", source=imgpath, device='cpu')

        st_video = open(outputpath, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Model Prediction")
        opencv_output(outputpath)


def opencv_output(outputpath):

    cap = cv2.VideoCapture(outputpath)

    while True:
        while cap.isOpened():
            ret, frame = cap.read()
            print(frame, ret)
            if ret:
                cv2.imshow("frame", frame)
            else:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'])

    option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
    # if torch.cuda.is_available():
    #     deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = False, index=1)
    # else:
    #     deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = True, index=0)
    # # -- End of Sidebar

    st.header('Find Bottle♻️')
    st.subheader('👈🏽 Select the options')
    st.sidebar.markdown("https://github.com/Raahul-G?tab=repositories")

    if option == "Image":
        imageInput(datasrc)
    elif option == "Video":
        videoInput(datasrc)


if __name__ == '__main__':
    main()


@st.cache_data
def loadModel():
    start_dl = time.time()
    model_file = wget.download('https://archive.org/download/findbottle_weights/best.pt', out="models/")
    finished_dl = time.time()
    print(f"Model Downloaded, ETA:{finished_dl - start_dl}")


loadModel()
