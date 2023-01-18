import threading
import av
import cv2
import os
import streamlit as st
from streamlit.components.v1 import html
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from yolo.yolo_predict import YoloPredict
import numpy as np
import queue
import pandas as pd


def navigation():
    try:
        path = st.experimental_get_query_params()['p'][0]
    except Exception as e:
        st.error('Please use the main app.')
        return None
    return path


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


@st.cache(allow_output_mutation=True)
def load_network():
    yp = YoloPredict()
    return yp


yp = load_network()
result_queue: queue.Queue = (
    queue.Queue()
)
if navigation() == "home":
    st.warning('啟動相機速度會較久一點,可以移至vpn展示頁面', icon="⚠️")


    genre = st.radio(
        "選擇模式 👇",
        ["常見物件偵測", "車牌辨識"],
        key="visibility",
        horizontal=True,
    )

    lock = threading.Lock()
    img_container = {"img": None}
    a = 1

    def video_frame_callback(frame):
        global a
        img = frame.to_ndarray(format="bgr24")
        # print(type(frame), img.shape)

        if genre == "常見物件偵測":
            img = yp.predict_default(img)
        else:
            # img = yp.predict(img)
            img = yp.predict_trt(img)
        with lock:
            img_container["img"] = img
            fruit_dict = {
                "width": img.shape[0],
                "height": img.shape[1]
            }
            df = pd.DataFrame([fruit_dict], index=["pixel"])
            result_queue.put(df) 
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {
                                    "width": {"min": 640, "ideal": 4096 / 2},
                                    "height": {"min": 480, "ideal": 2304 / 2},
                                    "aspectRatio": 1.777777778,
                                    "frameRate": {"min": 20},
                                    }, 
                                  "audio": False},
        async_processing=False,
        video_frame_callback=video_frame_callback,
        sendback_audio=False
    )

    labels_placeholder = st.empty()
    while True:
        try:
            result = result_queue.get(timeout=1.0)
        except queue.Empty:
            result = None
        
        labels_placeholder.table(result)

    js_code = '''
        $(document).ready(function(){
            $("button[kind=header]", window.parent.document).remove()
        });
    '''
    html(f'''

        <script src="https://cdn.bootcdn.net/ajax/libs/jquery/2.2.4/jquery.min.js"></script>
        <script>{js_code}</script>

        ''', width=0, height=0)

elif navigation() == "vpn":
    st.warning('需要使用專用的vpn才能啟動', icon="⚠️")

    genre = st.radio(
        "選擇模式 👇",
        ["常見物件偵測", "車牌辨識"],
        key="visibility",
        horizontal=True,
    )

    lock = threading.Lock()
    img_container = {"img": None}

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # print(type(frame), img.shape)

        if genre == "常見物件偵測":
            img = yp.predict_default(img)
        else:
            # img = yp.predict(img)
            img = yp.predict_trt(img)
        with lock:
            img_container["img"] = img
            fruit_dict = {
                "width": img.shape[0],
                "height": img.shape[1]
            }
            df = pd.DataFrame([fruit_dict], index=["pixel"])
            result_queue.put(df) 

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="example",
        # mode=WebRtcMode.SENDRECV,
        # rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {
                                    "width": {"min": 640, "ideal": 4096 / 2},
                                    "height": {"min": 480, "ideal": 2304 / 2},
                                    "aspectRatio": 1.777777778,
                                    "frameRate": {"min": 20},
                                    }, 
                                  "audio": False},
        async_processing=False,
        video_frame_callback=video_frame_callback,
        sendback_audio=False
    )

    labels_placeholder = st.empty()
    while True:
        try:
            result = result_queue.get(timeout=1.0)
        except queue.Empty:
            result = None
        labels_placeholder.table(result)

    js_code = '''
        $(document).ready(function(){
            $("button[kind=header]", window.parent.document).remove()
        });
    '''
    html(f'''

        <script src="https://cdn.bootcdn.net/ajax/libs/jquery/2.2.4/jquery.min.js"></script>
        <script>{js_code}</script>

        ''', width=0, height=0)

elif navigation() == "vedio":
    print(3)
    st.title('上傳影片 尚未完成')
    js_code = '''
        $(document).ready(function(){
            $("button[kind=header]", window.parent.document).remove()
        });
    '''
    html(f'''

        <script src="https://cdn.bootcdn.net/ajax/libs/jquery/2.2.4/jquery.min.js"></script>
        <script>{js_code}</script>

        ''', width=0, height=0)





