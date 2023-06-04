# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Network evaluation."""
import argparse
import os
import time

import cv2
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import float32 as dtype
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from models import Generator, UnetGenerator1, UnetGenerator3
from tqdm import tqdm
from utils.animegan_utils import denormalize_input, normalize_input
from utils.img_tools import *

def parse_args():
    """Argument parsing."""
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--device_target', default='GPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', default=0, type=int)
    # parser.add_argument('--test_dir', default='/home/ne438/ztf/myanimegan_git/dataset/HR1', type=str) #FSJ230331
    parser.add_argument('--test_dir', default='dataset/test_face1', type=str)
    parser.add_argument('--test_output', default='./dataset/output1', type=str)
    # parser.add_argument('--ckpt_file_name', default="/home/ne438/ztf/myanimegan_git/_training_dir/2022-08-27 10:57_Hayao@24/checkpoints/netG_24.ckpt") #FSJ230331
    parser.add_argument('--ckpt_file_name',default="/home/ne438/ztf/landscapegan2/landscapegan/_training_dir/2023-05-23 15:14_3d/checkpoints/netG_29.ckpt")
    return parser.parse_args()

def load_data(path):
    """
    Load data.
    """
    return os.listdir(path)


def transform(fpath):
    """
    Image normalization.
    """
    image = cv2.imread(fpath)[:, :, ::-1]
    image = normalize_input(image)
    image = np.expand_dims(image.transpose(2, 0, 1), axis=0)

    return image


def inverse_transform(image):
    """
    Image denormalization.
    """
    image = denormalize_input(image).asnumpy()
    image = cv2.cvtColor(image[0, :, :, :].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    return image

def count_params(net):
    """Count number of parameters in the network
    Args:
        net (mindspore.nn.Cell): Mindspore network instance
    Returns:
        total_params (int): Total number of trainable params
    """
    total_params = 0
    for param in net.trainable_params():
        total_params += np.prod(param.shape)
    return total_params


def main():
    """
    Convert real image to anime image.
    """
    net = UnetGenerator3()
    print(count_params(net))
    param_dict = load_checkpoint(args.ckpt_file_name)
    load_param_into_net(net, param_dict)
    data = load_data(args.test_dir)
    bar = tqdm(data)
    model = Model(net)

    if not os.path.exists(args.test_output):
        os.mkdir(args.test_output)
    total_time = 0
    list = os.listdir(args.test_dir)
    for img_path in bar:
        img = transform(os.path.join(args.test_dir, img_path))
        img = Tensor(img, dtype=dtype)
        time1 = time.time()
        output = model.predict(img)
        time2 = time.time()
        total_time += (time2 - time1)
        img = inverse_transform(img)
        output = inverse_transform(output)
        output = cv2.resize(output, (img.shape[1], img.shape[0]))
        # result = stylized[0, :, :, :].transpose(1, 2, 0)
        # adjust_config = [0.5, 50, 10]

        # result = adjust_saturation(result, adjust_config[0])
        # result = adjust_contrast(result, adjust_config[1])
        # result = adjust_luminance(result, adjust_config[2])
        # a = os.path.join(args.test_output, img_path)
        # cv2.imwrite(os.path.join(args.test_output, img_path), np.concatenate((img, output), axis=0))
        cv2.imwrite(os.path.join(args.test_output, img_path), output)
    print(total_time / len(list))
    print('Successfully output images in ' + args.test_output)


if __name__ == '__main__':
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    main()
