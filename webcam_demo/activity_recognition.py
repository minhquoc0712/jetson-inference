# Copyright 2020-2022 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import threading
import time
from typing import Dict
import numpy as np
from opendr.engine.target import Category
import torch
import torchvision
import cv2
from imutils import resize
from flask import Flask, Response, render_template
from imutils.video import VideoStream
from pathlib import Path

# OpenDR imports
from opendr.perception.activity_recognition import X3DLearner
from opendr.perception.activity_recognition import CoX3DLearner
from opendr.perception.activity_recognition import CLASSES as KINETICS400_CLASSES
from opendr.engine.data import Video, Image

TEXT_COLOR = (0, 0, 255)  # B G R


# Initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
output_frame = None
lock = threading.Lock()


# initialize a flask object
app = Flask(__name__)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def runnig_fps(alpha=0.1):
    # t0 = time.time_ns()
    t0 = time.time() * 1e9

    fps_avg = 10

    def wrapped():
        nonlocal t0, alpha, fps_avg
        # t1 = time.time_ns()
        t1 = time.time() * 1e9
        delta = (t1 - t0) * 1e-9
        t0 = t1
        fps_avg = alpha * (1 / delta) + (1 - alpha) * fps_avg
        return fps_avg

    return wrapped


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"{fps:.1f} FPS",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        TEXT_COLOR,
        2,
    )


def draw_preds(frame, preds: Dict, threshold=0.0):
    if preds[next(iter(preds))] < threshold:
        return

    base_skip = 40
    delta_skip = 30
    for i, (cls, prob) in enumerate(preds.items()):
        cv2.putText(
            frame,
            f"{prob:04.3f} {cls}",
            (10, base_skip + i * delta_skip),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            TEXT_COLOR,
            2,
        )


def draw_centered_box(frame, border):
    border = 10
    minX = (frame.shape[1] - frame.shape[0]) // 2 + border
    minY = border
    maxX = (frame.shape[1] + frame.shape[0]) // 2 - border
    maxY = frame.shape[0] - border
    cv2.rectangle(frame, (minX, minY), (maxX, maxY), color=TEXT_COLOR, thickness=1)


def center_crop(frame):
    height, width = frame.shape[0], frame.shape[1]
    e = min(height, width)
    x0 = (width - e) // 2
    y0 = (height - e) // 2
    cropped_frame = frame[y0: y0 + e, x0: x0 + e]
    return cropped_frame


def image_har_preprocessing(image_size: int):
    standardize = torchvision.transforms.Normalize(
        mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)
    )

    def wrapped(frame):
        nonlocal standardize
        frame = resize(frame, height=image_size, width=image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame).permute((2, 0, 1))  # H, W, C -> C, H, W
        frame = frame / 255.0  # [0, 255] -> [0.0, 1.0]
        frame = standardize(frame)
        return Image(frame, dtype=np.float)

    return wrapped


def video_har_preprocessing(image_size: int, window_size: int):
    frames = []

    standardize = torchvision.transforms.Normalize(
        mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)
    )

    def wrapped(frame):
        nonlocal frames, standardize
        frame = resize(frame, height=image_size, width=image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame).permute((2, 0, 1))  # H, W, C -> C, H, W
        frame = frame / 255.0  # [0, 255] -> [0.0, 1.0]
        frame = standardize(frame)
        if not frames:
            frames = [frame for _ in range(window_size)]
        else:
            frames.pop(0)
            frames.append(frame)
        vid = Video(torch.stack(frames, dim=1))
        return vid

    return wrapped


def clean_kinetics_preds(preds):
    k = 3
    class_scores, class_inds = torch.topk(preds[0].confidence, k=k)
    preds = {
        KINETICS400_CLASSES[int(class_inds[i])]: float(class_scores[i].item())
        for i in range(k)
    }
    return preds


def get_model(algorithms, model_name, device, execution_provider, use_onnx=False):
    if algorithms == 'x3d':
        # Init model
        learner = X3DLearner(device=device, backbone=model_name, num_workers=0)
        X3DLearner.download(path="model_weights", model_names={model_name})
        learner.load(Path("model_weights") / f"x3d_{model_name}.pyth")

        preprocess = video_har_preprocessing(
            image_size=learner.model_hparams["image_size"],
            window_size=learner.model_hparams["frames_per_clip"],
        )

        
    if algorithms == 'cox3d':
        # Init model
        learner = CoX3DLearner(device=device, backbone=model_name, num_workers=0)
        CoX3DLearner.download(path="model_weights", model_names={model_name})
        learner.load(Path("model_weights") / f"x3d_{model_name}.pyth")
        preprocess = image_har_preprocessing(image_size=learner.model_hparams["image_size"])

    if use_onnx:
        learner.optimize()
        if execution_provider == "cpu":
            ep = 'CPUExecutionProvider'
        elif execution_provider == "cuda":
            ep = 'CUDAExecutionProvider'
        elif execution_provider == "tensorrt":
            ep = 'TensorrtExecutionProvider'
        else:
            raise RuntimeError("Unknown model name for execution provider")

        learner.ort_session.set_providers([ep])

    return learner, preprocess

def ort_infer(input, learner):
    if learner.ort_session == None:
        return None

    input = input.numpy()
    t_start = time.time()
    outputs = learner.ort_session.run(None, {'video': input[np.newaxis, ...]})
    t_end = time.time()
    print(f"Infernece time: {t_end - t_start}")
    outputs = torch.tensor(outputs[0])
    results = [Category(prediction=int(r.argmax(dim=0)), confidence=r) for r in outputs]
    return results
    

def read_cam(cap, algorithm, model_name, device, execution_provider):
    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    font = cv2.FONT_HERSHEY_PLAIN

    # Prep stats
    fps = runnig_fps()

    learner, preprocess = get_model(algorithm, model_name, device, execution_provider, use_onnx=True)
    print(f'onnx runtime providers: {learner.ort_session.get_provider_options()}')

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break
        _, img = cap.read() # grab the next image frame from camera
        if show_help:
            cv2.putText(img, help_text, (11, 20), font,
                        1.0, (32, 32, 32), 4, cv2.LINE_AA)
            cv2.putText(img, help_text, (10, 20), font,
                        1.0, (240, 240, 240), 1, cv2.LINE_AA)

        img = center_crop(img)
        vid = preprocess(img)

        # Gererate preds
        # preds = learner.infer(vid)
        preds = ort_infer(vid, learner)
        preds = clean_kinetics_preds(preds)

        draw_preds(img, preds)
        draw_fps(img, fps())

        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(10)
        if key == 27: # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'): # toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'): # toggle fullscreen
            full_scrn = not full_scrn
            if full_scrn:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)



# check to see if this is the main thread of execution
if __name__ == "__main__":
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--ip", type=str, required=True, help="IP address of the device"
    )
    ap.add_argument(
        "-o",
        "--port",
        type=int,
        required=True,
        help="Ephemeral port number of the server (1024 to 65535)",
    )
    ap.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="xs",
        help="Model identifier",
    )
    ap.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device",
    )
    ap.add_argument(
        "-v",
        "--video_source",
        type=int,
        default=0,
        help="ID of the video source to use",
    )
    ap.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="x3d",
        help="Which algortihm to run",
        choices=["cox3d", "x3d"],
    )
    ap.add_argument(
        "-ep",
        "--execution_provider",
        type=str,
        default="cpu",
        help="Execution providers for onnxruntime inference session",
        choices=["cpu", "cuda", "tensorrt"]
    )
    args = vars(ap.parse_args())

    width = 1920
    height = 1080
    gst_str = ('nvarguscamerasrc ! '
                'video/x-raw(memory:NVMM), '
                'width=(int)1920, height=(int)1080, '
                'format=(string)NV12, framerate=(fraction)30/1 ! '
                'nvvidconv flip-method=0 ! '
                'video/x-raw, width=(int){}, height=(int){}, '
                'format=(string)BGRx ! '
                'videoconvert ! appsink').format(width, height)

    # gst_str = ('nvarguscamerasrc exposuretimerange="1 1" gainrange="1 1" ! video/x-raw(memory:NVMM), format=NV12,width=1280, height=720, framerate=120/1 ! nvoverlaysink')

    vs = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    WINDOW_NAME = 'CameraDemo'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson TX2/TX1')

    algorithm = args["algorithm"]
    model_name = args["model_name"]
    device = args["device"]
    ep = args["execution_provider"]

    read_cam(vs, algorithm, model_name, device, ep)


    # release the video stream pointer
    vs.release()
    cv2.destroyAllWindows()