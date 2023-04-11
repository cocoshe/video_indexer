import streamlit as st
from utils import *

st.title('Video Search Engine')
st.write('This is a video search engine based on Jina. You can search for videos by entering a description of the video. The search results will be displayed below.')

VIDEO_FILE_PATH = f'tmp{os.sep}videos'
video_uri = None
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # video_file = open(uploaded_file, 'rb')
    video_bytes = uploaded_file.read()
    # video_ = copy.deepcopy(video_bytes)
    st.video(video_bytes)
    with open(f'{VIDEO_FILE_PATH}/{uploaded_file.name}', 'wb') as f:
        f.write(video_bytes)

    video_uri = f'{VIDEO_FILE_PATH}/{uploaded_file.name}'

disc = st.text_input('Input description here!', placeholder='a man is playing a guitar')
number = st.number_input('find top K moments', min_value=1, max_value=100,
                         value=5, step=1,)
btn = st.button('search')
if btn:
    if uploaded_file is not None:
        if disc == "":
            st.warning('Please input description!')
        else:
            with st.spinner('Searching...'):
                # st.success('Done!')
                videos_output = hf_search(video_uri, disc, int(number))
                for v in videos_output:
                    st.video(v)
