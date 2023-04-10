import streamlit as st
from jina import Document, DocumentArray
import ffmpeg
import numpy as np
import torch
from PIL import Image

def _convert_video_to_tensor(video):
    print('video', video)
    prob = ffmpeg.probe(video)
    stream = prob['streams'][0]
    w, h = int(stream['width']), int(stream['height'])

    out, _ = (
        ffmpeg
        .input(video)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )
    video = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, h, w, 3])
    )  # N,H,W,C

    return video


def search(video_uri, disc, number, c):
    # video = Document(uri=video_uri).load_uri_to_video_tensor(only_keyframes=True)

    disc = Document(text=disc)

    # print(video.tensor.shape)  # N, H, W, C
    video = _convert_video_to_tensor(video_uri)
    video = Document(tensor=video)

    doc = Document()
    for i, frame in enumerate(video.tensor):
        doc.chunks.append(Document(tensor=frame))

    video_emb = c.post('/video_encode', doc)
    video_emb = video_emb[0].embedding
    # print(ret)
    # print(ret[0].embedding)
    print('video_emb', torch.tensor(video_emb).shape)

    text_emb = c.post('/text_encode', disc)
    text_emb = text_emb[0].embedding
    print('text_emb', torch.tensor(text_emb).shape)

    # d = Document(tags={'video_emb': video_emb, 'text_emb': text_emb})
    # video_emb = torch.Tensor(video_emb).squeeze(1).tolist()
    # text_emb = torch.Tensor(text_emb).tolist()
    video_emb = torch.Tensor(video_emb).unsqueeze(1).tolist()
    text_emb = torch.Tensor(text_emb).tolist()
    results = c.post('/index', [video_emb, text_emb],
                        parameters={'top_k': int(number),
                                    'threshold': 0.1},
                        return_results=True)  # DocumentArray
    print('results', results)
    videos_output = clip_video(video_uri, results)  # save
    return videos_output


def clip_video(video_uri, results):
    video_stream = ffmpeg.input(video_uri)
    videos_output = []
    for i, result in enumerate(results):
        start = result.tags['start']
        end = result.tags['end']
        video_path = f'{video_uri.split(".")[0]}_top_{i}.mp4'
        video = (
            video_stream
            .trim(start_frame=start, end_frame=end)
            .setpts('PTS-STARTPTS')
            .output(video_path)
            .overwrite_output()
            .run_async(pipe_stdout=True)
        )
        videos_output.append(video_path)

    return videos_output