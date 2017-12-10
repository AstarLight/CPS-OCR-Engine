#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import pickle
import argparse
from argparse import RawTextHelpFormatter
import fnmatch
import os
import cv2
import json
import random
import numpy as np
import shutil
from deep_ocr.lang_aux import LangCharsGenerate
from deep_ocr.lang_aux import FontCheck
from deep_ocr.lang_aux import Font2Image

def get_label_dict():
    f=open('./chinese_labels','r')
    label_dict = pickle.load(f)
    f.close()
    return label_dict

if __name__ == "__main__":

    description = '''
        deep_ocr_make_caffe_dataset --out_caffe_dir /root/data/caffe_dataset \
            --font_dir /root/workspace/deep_ocr_fonts/chinese_fonts \
            --width 30 --height 30 --margin 4 --langs lower_eng
    '''

    parser = argparse.ArgumentParser(
        description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--out_caffe_dir', dest='out_caffe_dir',
                        default=None, required=True,
                        help='write a caffe dir')
    parser.add_argument('--font_dir', dest='font_dir',
                        default=None, required=True,
                        help='font dir to to produce images')
    parser.add_argument('--test_ratio', dest='test_ratio',
                        default=0.3, required=False,
                        help='test dataset size')
    parser.add_argument('--width', dest='width',
                        default=None, required=True,
                        help='width')
    parser.add_argument('--height', dest='height',
                        default=None, required=True,
                        help='height')
    parser.add_argument('--no_crop', dest='no_crop',
                        default=True, required=False,
                        help='', action='store_true')
    parser.add_argument('--margin', dest='margin',
                        default=0, required=False,
                        help='', )
    parser.add_argument('--langs', dest='langs',
                        default="chi_sim", required=False,
                        help='deep_ocr.langs.*, e.g. chi_sim, chi_tra, digits...')
    parser.add_argument('--rotate', dest='rotate',
                        default=0, required=False,
                        help='max rotate degree 0-45')
    parser.add_argument('--rotate_step', dest='rotate_step',
                        default=0, required=False,
                        help='rotate step for the rotate angle')
    options = parser.parse_args()

    out_caffe_dir = os.path.expanduser(options.out_caffe_dir)
    font_dir = os.path.expanduser(options.font_dir)
    test_ratio = float(options.test_ratio)
    width = int(options.width)
    height = int(options.height)
    need_crop = not options.no_crop
    margin = int(options.margin)
    langs = options.langs
    rotate = int(options.rotate)
    rotate_step = int(options.rotate_step)

    train_image_dir_name = "train"
    test_image_dir_name = "test"


    train_images_dir = os.path.join(out_caffe_dir, train_image_dir_name)
    test_images_dir = os.path.join(out_caffe_dir, test_image_dir_name)
    if os.path.isdir(train_images_dir):
        shutil.rmtree(train_images_dir)
    os.makedirs(train_images_dir)

    if os.path.isdir(test_images_dir):
        shutil.rmtree(test_images_dir)
    os.makedirs(test_images_dir)
    
    label_dict = get_label_dict()
    
    #lang_chars_gen = LangCharsGenerate(langs)
    #lang_chars = lang_chars_gen.do()
    
    char_list=[]
    value_list=[]
    for (value,chars) in label_dict.items():
        print (value,chars)
        char_list.append(chars)
        value_list.append(value)

    lang_chars = dict(zip(char_list,value_list))
    font_check = FontCheck(lang_chars)
    
    
    if rotate < 0:
        roate = - rotate
    

    if rotate > 0 and rotate <= 45:
        all_rotate_angles = []
        for i in range(0, rotate+1, rotate_step):
            all_rotate_angles.append(i)
        for i in range(-rotate, 0, rotate_step):
            all_rotate_angles.append(i)
        #print(all_rotate_angles)
    
    verified_font_paths = []
    ## search for file fonts
    for font_name in os.listdir(font_dir):
        path_font_file = os.path.join(font_dir, font_name)
        if font_check.do(path_font_file):
            verified_font_paths.append(path_font_file)

    train_list = []
    test_list = []
    max_train_i = int(len(verified_font_paths) * 5) # we select 5 pic every char for test
    print ('max train: ', max_train_i)
    font2image = Font2Image(width, height, need_crop, margin)

    
    for (char, value) in lang_chars.items():
        count = 1000
        print (char,value)
        #char_dir = os.path.join(images_dir, "%0.5d" % value)
        for j, verified_font_path in enumerate(verified_font_paths):           
            if rotate == 0:
                if count < (1000+max_train_i):
                    char_dir = os.path.join(test_images_dir, "%0.5d" % value)
                else:
                    char_dir = os.path.join(train_images_dir, "%0.5d" % value)

                if not os.path.isdir(char_dir):
                    os.makedirs(char_dir)
                path_image = os.path.join(char_dir,"%d.png" % count)
                font2image.do(verified_font_path, char, path_image)
                count += 1
            else:
                for k in all_rotate_angles:
                    if count < (1000+max_train_i):
                        char_dir = os.path.join(test_images_dir, "%0.5d" % value)
                    else:
                        char_dir = os.path.join(train_images_dir, "%0.5d" % value)

                    if not os.path.isdir(char_dir):
                        os.makedirs(char_dir)
                    path_image = os.path.join(char_dir,"%d.png" % count)
                    font2image.do(verified_font_path, char, path_image, rotate=k)
                    count += 1
                
    
