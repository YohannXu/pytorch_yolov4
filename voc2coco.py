# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-07-09 15:50:27
# Description: voc_2_coco.py
import json
from glob import glob
from xml.etree import ElementTree as ET

from tqdm import tqdm

classes = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}


def main(image_dirs):
    f = {}

    f['categories'] = [
        {'id': 1, 'name': 'aeroplane'},
        {'id': 2, 'name': 'bicycle'},
        {'id': 3, 'name': 'bird'},
        {'id': 4, 'name': 'boat'},
        {'id': 5, 'name': 'bottle'},
        {'id': 6, 'name': 'bus'},
        {'id': 7, 'name': 'car'},
        {'id': 8, 'name': 'cat'},
        {'id': 9, 'name': 'chair'},
        {'id': 10, 'name': 'cow'},
        {'id': 11, 'name': 'diningtable'},
        {'id': 12, 'name': 'dog'},
        {'id': 13, 'name': 'horse'},
        {'id': 14, 'name': 'motorbike'},
        {'id': 15, 'name': 'person'},
        {'id': 16, 'name': 'pottedplant'},
        {'id': 17, 'name': 'sheep'},
        {'id': 18, 'name': 'sofa'},
        {'id': 19, 'name': 'train'},
        {'id': 20, 'name': 'tvmonitor'}
    ]

    f['images'] = []
    f['annotations'] = []
    image_id = 1
    anno_id = 1

    for image_dir in image_dirs:
        xml_names = glob(image_dir + '/Annotations/*')
        for xml_name in tqdm(xml_names):
            tree = ET.parse(xml_name)
            root = tree.getroot()

            image_name = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            image = {
                'file_name': image_name,
                'height': height,
                'width': width,
                'id': image_id
            }
            f['images'].append(image)

            objs = root.findall('object')
            for obj in objs:
                category_name = obj.find('name').text
                category_id = classes[category_name]
                box = obj.find('bndbox')
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
                w = xmax - xmin
                h = ymax - ymin

                anno = {
                    'area': w * h,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                    'category_id': category_id,
                    'id': anno_id
                }
                f['annotations'].append(anno)
                anno_id += 1

            image_id += 1

    json.dump(f, open('annotations.json', 'w'))


if __name__ == '__main__':
    main(['VOCdevkit2/VOC2007', 'VOCdevkit3/VOC2012'])
    # main(['VOCdevkit/VOC2007'])
