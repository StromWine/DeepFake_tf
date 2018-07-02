import cv2
import numpy as np
import argparse
from tqdm import tqdm
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def test_net(images, args):
    sess = tf.Session()

    files = os.listdir(args.meta_path)
    meta_list = []
    for file in files:
        if file.endswith('.meta'):
            meta_list.append(os.path.join(args.meta_path, file))
    if len(meta_list) != 0:
        saver = tf.train.import_meta_graph(meta_list[len(meta_list)-1])
        saver.restore(sess, tf.train.latest_checkpoint(args.meta_path))
        print("******resotre over!!!!******")
    else:
        print("******cannot find the meta files!!!!******")

    graph = tf.get_default_graph()
    input_wrap = graph.get_tensor_by_name("input/input_wrap:0")
    if args.trans == 'AtoB':
        pre = graph.get_tensor_by_name("decoder_B/decoder_B_pre:0")
        pre_mask = graph.get_tensor_by_name("decoder_B/decoder_B_mask:0")
    elif args.trans == 'BtoA':
        pre = graph.get_tensor_by_name("decoder_A/decoder_A_pre:0")
        pre_mask = graph.get_tensor_by_name("decoder_A/decoder_A_mask:0")
    print("******get ready to pre!!!!******")

    pbar = tqdm(range(len(images)))
    for num in pbar:
        image = images[num]
        pre_dict = {input_wrap: image}
        pre_image, _ = sess.run([pre, pre_mask], feed_dict=pre_dict)
        cv2.imshow('pre', pre_image[0])
        cv2.waitKey(2)

def get_image(args):
    files = os.listdir(args.pre_image)
    image_list = []
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            image_list.append(os.path.join(args.pre_image, file))

    images =[]
    pbar = tqdm(range(len(image_list)))
    for num in pbar:
        image = cv2.imread(image_list[num])
        image = cv2.resize(image, (64, 64), cv2.INTER_AREA)
        image = np.expand_dims(image, 0)
        images.append(image/255.0)
    return images

def test(args):
    images = get_image(args)
    test_net(images, args)

if __name__ == '__main__':
    print('testing!!!!')
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, default='./models')
    parser.add_argument("--trans", type=str, default='AtoB')
    parser.add_argument("--pre_image", type=str, default='./image_3/aligned')
    args = parser.parse_args()
    test(args)
