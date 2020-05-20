#!/usr/bin/env python
# coding: utf-8

# Load necessary modules
import sys
sys.path.insert(0, '../')

import cv2
import os
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu


class FPS(object):
    """ Calculate FPS.
        example) fps = FPS()
                 while(cap.isOpended()):
                     # Your processing
                     fps.calculate(draw)
                     cv2.imshow('test', draw)
    """

    def __init__(self):
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        self.prev_time = timer()

    def calculate(self, draw, show=True):
        curr_time = timer()
        exec_time = curr_time - self.prev_time
        self.prev_time = curr_time
        self.accum_time += exec_time
        self.curr_fps += 1
        if self.accum_time > 1:
            self.accum_time -= 1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0
        if show:
            cv2.rectangle(draw, (0, 0), (60, 20), (255, 255, 255), -1)
            cv2.putText(draw, self.fps, (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        else:
            print(self.fps)


def render(draw, results, scale, labels_to_names, score_threshold=0.5):
    """draw box and class name on box.
    args: draw(BGR), results(shape=(3, )), 
          labels_to_names: {class_id: classname},
    return: drawed image
    """
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    # get bbox, confidential score, and label id
    boxes, scores, labels = results
    boxes /= scale
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < score_threshold:
            break
        # draw bbox
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        # put caption of box
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
    return draw


def make_predict_movie(movie_path, save_path,
                       model, labels_to_names,
                       start_frame=0, end_frame=1000000):
    # for video read(input)
    cap = cv2.VideoCapture(movie_path)
    raw_size = (int(cap.get(3)), int(cap.get(4)))
    cap.set(1, start_frame)
    # for video write(output)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, raw_size)
    # for calculate frame per second
    fps = FPS()
    for _ in tqdm(range(end_frame - start_frame)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is False:
            break
        # Preprocessing
        input_image = preprocess_image(frame.copy())
        input_image, scale = resize_image(input_image)
        # Prediction on model
        results = model.predict_on_batch(np.expand_dims(input_image, axis=0))[:3]
        # Display the predicted frame
        output = render(frame, results, scale, labels_to_names)
        fps.calculate(output)
        out.write(output)

    cap.release()
    out.release()


if __name__ == '__main__':
    # use this to change which GPU to use
    gpu = 0
    setup_gpu(gpu)  # set the modified tf session as backend in keras
    # classname mapping
    labels_to_names = {0: 'root'}
    # input data path
    movie_path = '/data/input/IMA_root/test/case2_25_18-26_13.mp4'
    start_frame = 0
    end_frame = 30 * 60
    # save predicted movie
    save_path = 'case2_IMArootDet.avi'
    # trained model path
    model_path = '/data/result/IMA_root/season5/snapshot_trial3/model/resnet50_csv_50.h5'

    ### main ###
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    model = models.convert_model(model)  # necessary

    make_predict_movie(
        movie_path, save_path,
        model, labels_to_names,
        start_frame, end_frame,
    )
