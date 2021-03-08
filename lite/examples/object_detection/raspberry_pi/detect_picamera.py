#!/usr/bin/env python3
# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Copyright 2021 Brian Katona. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from https://github.com/tensorflow/examples/lite/examples/object_detection/raspberry_pi
"""Example using TF Lite to detect objects from images in a directory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import re
import time
import os


import numpy as np

from PIL import Image, ImageDraw
from tflite_runtime.interpreter import Interpreter

RESULT_PATH = "positive"


def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results


def main():
  #Create folder for results
  if not os.path.isdir(os.getcwd() + "/" + RESULT_PATH): 
    os.mkdir(os.getcwd() + "/" + RESULT_PATH)

  #Read Arguments
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  parser.add_argument(
      '--threshold',
      help='Score threshold for detected objects.',
      required=False,
      type=float,
      default=0.4)
  args = parser.parse_args()
  
  #Set up labels and interpreter
  labels = load_labels(args.labels)
  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
  
  #Iterate through files in working directory, process any JPEG or PNG images
  for file in os.scandir(os.getcwd()):
    if (file.path.endswith(".jpg") or file.path.endswith(".png")):
      print(file.name)
      image = Image.open(file.name).convert('RGB').resize((input_width, input_height), Image.ANTIALIAS)
      start_time = time.monotonic()
      results = detect_objects(interpreter, image, args.threshold)
      elapsed_ms = (time.monotonic() - start_time) * 1000
  #Iterate through results. If a person is found, draw a bounding box around the person and save image
  # in results directory
      for obj in results:
        print(obj)
        if (obj['class_id'] == 0):
          draw = ImageDraw.Draw(image)
          ymin, xmin, ymax, xmax = obj['bounding_box']
          im_width, im_height = image.size
          abs_coordinates = [xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height]
          draw.rectangle(abs_coordinates, fill=None, outline='yellow', width=2)
          image.save(os.getcwd() + "/" +  RESULT_PATH + "/" +  file.name)
      image.close() 

if __name__ == '__main__':
  main()
