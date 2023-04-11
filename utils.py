import numpy as np
import torch
import ffmpeg
from PIL import Image
import os
from tqdm import tqdm
from hf_model import *


def _convert_video_to_tensor(video):
    print('video', video)
    prob = ffmpeg.probe(video)
    stream = prob['streams'][0]
    w, h = int(stream['width']), int(stream['height'])
    # get stream time
    stream_time = stream['duration']
    print('stream_time', stream_time)

    out, _ = (
        ffmpeg
        .input(video)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=1)
        .run(capture_stdout=True)
    )
    video = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, h, w, 3])
    )  # N,H,W,C
    print('-' * 100)
    print('video to tensor done')
    print('-' * 100)
    return video, stream_time


def hf_search(video_uri, disc, topk):
    video, total_time = _convert_video_to_tensor(video_uri)
    images = []
    for frame in tqdm(video, desc='loading images'):
        img = Image.fromarray(frame)
        images.append(img)

    batch_size = 8
    batch_num = len(images) // batch_size
    logits_list = []
    for i in range(batch_num):
        batch = images[i * batch_size: (i + 1) * batch_size]
        batch = processor(text=[disc], images=batch, return_tensors="pt", padding=True)
        outputs = model(**batch)
        logits_per_image = outputs.logits_per_image
        logits_per_image = logits_per_image.reshape(-1).tolist()
        logits_list += logits_per_image
    probs = torch.softmax(torch.tensor(logits_list), dim=0)

    values, indices = [], []
    for i in range(topk):
        value, index = torch.max(probs, dim=0)
        values.append(value.item())
        indices.append(index.item())
        probs[index - 2: index + 2] = 0

    # values, indices = probs.topk(topk)

    print('values', values)
    print('indices', indices)
    for i in indices:
        images[i].save(f'./images/{i}.jpg')
    ratios = [indices[i] / len(images) for i in range(len(indices))]
    print('ratios', ratios)
    times = [float(total_time) * ratios[i] for i in range(len(ratios))]
    print('times', times)

    ret_videos = clip_video_from_times(video_uri, times)
    return ret_videos


def clip_video_from_times(video_uri, times):
    ret_videos = []
    if not os.path.exists('clips'):
        os.mkdir('clips')
    for i, time in tqdm(enumerate(times), desc='clipping videos'):
        out, _ = (
            ffmpeg
            .input(video_uri)
            .output(f'./clips/{i}.mp4', ss=time - 2, t=4)
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True, )# quiet=True)
        )
        ret_videos.append(f'./clips/{i}.mp4')
    return ret_videos



# video_uri = 'asd.mp4'
# hf_search(video_uri, 'a white dog', 5)
# --------------------------------------------------------------------------------------------------------
def test_clip_video():
    indices = [9, 10, 12, 13, 15, 16, 17, 18, 19, 23]
    images = [0 for i in range(168)]
    total_time = 166.0
    video_uri = 'asd.mp4'

    ratios = [indices[i] / len(images) for i in range(len(indices))]
    print('ratios', ratios)
    times = [float(total_time) * ratios[i] for i in range(len(ratios))]
    print('times', times)

    clip_video_from_times(video_uri, times)
