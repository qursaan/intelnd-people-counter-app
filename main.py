"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_masks(result, width, height):
    '''
    Draw semantic mask classes onto the frame.
    '''
    # Create a mask with color by class
    classes = cv2.resize(result[0].transpose((1,2,0)), (width,height),
        interpolation=cv2.INTER_NEAREST)
    unique_classes = np.unique(classes)
    out_mask = classes * (255/20)

    # Stack the mask so FFmpeg understands it
    out_mask = np.dstack((out_mask, out_mask, out_mask))
    out_mask = np.uint8(out_mask)

    return out_mask, unique_classes

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model=args.model,
                             device=args.device,
                             cpu_extension=args.cpu_extension)

    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###

    ### Handle image, video or webcam
    image_flag = False
    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
    # Checks if the input is an image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True
    # else assume input is vedio file

    # Get and open video capture
    cap = cv2.VideoCapture(args.input)
    if args.input:
        cap.open(args.input)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    # iniatilize variables
    count_total   = 0
    count_prev    = 0
    count_curr    = 0
    duration_curr = 0
    duration_prev = 0
    duration_total= 0
    frame_time    = 0
    frame_count   = 0
    timer_curr_start = 0
    request_id    = 0

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)


        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        # Update layout
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        timer_infer_start = time.time()
        infer_network.exec_net(p_frame, request_id)

        ### TODO: Wait for the result ###
        if infer_network.wait(request_id) == 0:

            ### TODO: Get the results of the inference request ###
            timer_infer_delay = time.time() - timer_infer_start
            result = infer_network.get_output(request_id)


            ### TODO: Extract any desired stats from the results ###

            #  Draw bounding box
            conf = result[0, 0, :, 2]
            count_curr = 0
            for i, c in enumerate(conf):
                if c > prob_threshold:
                    rect_box = result[0, 0, i, 3:]
                    min_x = int(rect_box[0] * width)
                    min_y = int(rect_box[1] * height)
                    max_x = int(rect_box[2] * width)
                    max_y = int(rect_box[3] * height)
                    frame = cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255,0, 0), 1)
                    count_curr = count_curr + 1

            ### TODO: Calculate and send relevant information on ###

            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

            # IF new person comes inside imapge
            if count_curr > count_prev:
                timer_curr_start = time.time()
                count_total = count_total + count_curr - count_prev
                client.publish('person', payload=json.dumps({'total': count_total}))

            # Calc Person Duration
            if count_curr < count_prev:
                timer_curr_delay = time.time() - timer_curr_start
                client.publish('person/duration', payload=json.dumps({'duration': timer_curr_delay}))

            # Write out information
            text_infer = "Inference Delay: {:.3f}ms".format(timer_infer_delay * 1000)
            text_counter = "Current Counter: {}".format(count_curr)
            cv2.putText(frame, text_infer, (10, 15),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5,  (255, 0, 0), 1)
            cv2.putText(frame, text_counter, (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5,  (0, 0, 255), 1)

            if count_curr > 0:
                text_duration = "Current Duration: {:.1f}s".format(time.time() - timer_curr_start)
                cv2.putText(frame, text_duration, (10, 45),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5,  (0, 255, 0), 1)

            count_prev = count_curr
            client.publish("person", json.dumps({"count": count_curr}))


        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        # Break if escape key pressed
        if key_pressed == 27:
            break

        ### TODO: Write an output image if `single_image_mode` ###

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
