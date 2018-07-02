import argparse
import os
import cv2
import json
import numpy
from tqdm import tqdm
import tensorflow as tf
import tensorflow.contrib.slim as slim
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
n = 0

imageSize = 256
croppedSize = 240
zmask = numpy.zeros((1, 128, 128, 1), float)


def image_stats(image):
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    return (lMean, lStd, aMean, aStd, bMean, bStd)


def adjust_avg_color(img_old, img_new):
    w, h, c = img_new.shape
    for i in range(img_new.shape[-1]):
        old_avg = img_old[:, :, i].mean()
        new_avg = img_new[:, :, i].mean()
        diff_int = (int)(old_avg - new_avg)
        for m in range(img_new.shape[0]):
            for n in range(img_new.shape[1]):
                temp = (img_new[m, n, i] + diff_int)
                if temp < 0:
                    img_new[m, n, i] = 0
                elif temp > 255:
                    img_new[m, n, i] = 255
                else:
                    img_new[m, n, i] = temp


def transfer_avg_color(img_old, img_new):
    assert (img_old.shape == img_new.shape)
    source = cv2.cvtColor(img_old, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(img_new, cv2.COLOR_BGR2LAB).astype("float32")

    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    (l, a, b) = cv2.split(target)

    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b

    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    l = numpy.clip(l, 0, 255)
    a = numpy.clip(a, 0, 255)
    b = numpy.clip(b, 0, 255)

    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    return transfer


def convert_one_image(args, new_face_rgb_, new_face_m_, sourceFace, image, mat):
    image_size = image.shape[1], image.shape[0]

    # _,other_face_m = otherautoencoder.predict( [face / 255.0,zmaskcv2.imshow('new_face_rgb', new_face_rgb)
    #     cv2.waitKey(2)
    #     cv2.destroyAllWindows()] )

    # new_face_m = numpy.maximum(new_face_m, other_face_m )

    new_face_rgb = numpy.clip(new_face_rgb_[0] * 255, 0, 255).astype(image.dtype)
    new_face_m = numpy.clip(new_face_m_[0], 0, 1).astype(float) * numpy.ones(
        (new_face_m_.shape[0], new_face_m_.shape[1], 3))

    # just for test
    cv2.imshow('new_face_rgb', new_face_rgb)
    cv2.waitKey(2)
    cv2.imshow('new_face_m', new_face_m)
    cv2.waitKey(2)
    cv2.destroyAllWindows()

    base_image = numpy.copy(image)
    new_image = numpy.copy(image)

    transmat = mat * (64 - 16) * 2
    transmat[::, 2] += 8 * 2

    adjust_avg_color(sourceFace, new_face_rgb)
    cv2.imshow('sourceface', sourceFace)
    cv2.waitKey(2)
    cv2.imshow('new_face_rgb', new_face_rgb)
    cv2.waitKey(2)
    cv2.destroyAllWindows()

    cv2.warpAffine(new_face_rgb, transmat, image_size, new_image, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC,
                   cv2.BORDER_TRANSPARENT)
    cv2.imshow('new_face_rgb', new_face_rgb)
    cv2.waitKey(2)
    cv2.destroyAllWindows()

    image_mask = numpy.zeros_like(new_image, dtype=float)

    cv2.warpAffine(new_face_m, transmat, image_size, image_mask, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC,
                   cv2.BORDER_TRANSPARENT)
    cv2.imshow('new_face_rgb', new_face_rgb)
    cv2.waitKey(2)
    cv2.destroyAllWindows()
    # slightly enlarge the mask area
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    image_mask = cv2.dilate(image_mask, kernel, iterations=1)

    if args.seamlessClone:

        unitMask = numpy.clip(image_mask * 365, 0, 255).astype(numpy.uint8)

        maxregion = numpy.argwhere(unitMask == 255)

        if maxregion.size > 0:
            miny, minx = maxregion.min(axis=0)[:2]
            maxy, maxx = maxregion.max(axis=0)[:2]
            lenx = maxx - minx;
            leny = maxy - miny;
            masky = int(minx + (lenx // 2))
            maskx = int(miny + (leny // 2))

            new_image = cv2.seamlessClone(new_image.astype(numpy.uint8), base_image.astype(numpy.uint8), unitMask,
                                          (masky, maskx), cv2.NORMAL_CLONE)
            cv2.imshow('new_image', new_image)
            cv2.waitKey(2)
            cv2.destroyAllWindows()

    image_mask = cv2.GaussianBlur(image_mask, (11, 11), 0)

    foreground = cv2.multiply(image_mask, new_image.astype(float))
    background = cv2.multiply(1.0 - image_mask, base_image.astype(float))

    output = numpy.add(background, foreground)
    if args.vision:
        cv2.imshow("output", output.astype(numpy.uint8))
        if cv2.waitKey(1) == ord('q'):
            exit()
    return output

def get_faceFun(image, mat):
    global n
    n += 1
    image_size = image.shape[1], image.shape[0]

    sourceMat = mat.copy()

    sourceMat = sourceMat * (240 + (16 * 2))
    sourceMat[:, 2] += 48

    face = cv2.warpAffine(image, sourceMat, (240 + (48 + 16) * 2, 240 + (48 + 16) * 2))
    cv2.imshow('image', image)
    cv2.waitKey(1)
    cv2.imshow('face', face)
    cv2.destroyAllWindows()
    sourceFace = face.copy()
    sourceFace = cv2.resize(sourceFace, (128, 128), cv2.INTER_CUBIC)

    face = cv2.resize(face, (64, 64), cv2.INTER_AREA)
    face = numpy.expand_dims(face, 0)
    face = face / 255.0

    return face, sourceFace


def main(args):
    input_dir = args.input_dir
    assert os.path.exists(input_dir)

    alignments = os.path.join(input_dir, 'alignments.json')
    print(alignments)
    alignments = json.loads(open(alignments).read())

    output_dir = os.path.join(input_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.blurSize % 2 == 0:
        args.blurSize += 1

    if args.erosionKernelSize > 0:
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.erosionKernelSize, args.erosionKernelSize))
    else:
        erosion_kernel = None

    for e in alignments:
        if len(e) < 4:
            raise LookupError('This script expects new format json files with face points included.')

    sess = tf.Session()

    files = os.listdir(args.meta_path)
    meta_list = []
    for file in files:
        if file.endswith('.meta'):
            meta_list.append(os.path.join(args.meta_path, file))
    if len(meta_list) != 0:
        saver = tf.train.import_meta_graph(meta_list[len(meta_list) - 1])
        saver.restore(sess, tf.train.latest_checkpoint(args.meta_path))
        print("******resotre over!!!!******")
    else:
        print("******cannot find the meta files!!!!******")

    graph = tf.get_default_graph()
    input_wrap = graph.get_tensor_by_name("input/input_wrap:0")
    if args.direction == 'AtoB':
        pre = graph.get_tensor_by_name("decoder_B/decoder_B_pre:0")
        pre_mask = graph.get_tensor_by_name("decoder_B/decoder_B_mask:0")
    elif args.direction == 'BtoA':
        pre = graph.get_tensor_by_name("decoder_A/decoder_A_pre:0")
        pre_mask = graph.get_tensor_by_name("decoder_A/decoder_A_mask:0")
    print("******get ready to pre!!!!******")

    for image_file, face_file, mat, facepoints in tqdm(alignments[args.startframe::args.frameSkip]):
        image = cv2.imread(os.path.join(input_dir, image_file))
        face_dir = os.path.join(input_dir + '/aligned', image_file)
        face = cv2.imread(face_dir)

        mat = numpy.array(mat).reshape(2, 3)

        if image is None: continue
        if face is None: continue

        get_face, get_sourceface = get_faceFun(image, mat)
        feed_dict = {input_wrap:get_face}

        new_face, new_face_m = sess.run([pre, pre_mask], feed_dict=feed_dict)
        if args.doublePass:
            # feed the original prediction back into the network for a second round.
            new_face_rgb = new_face.reshape((128, 128, 3))
            new_face_rgb = cv2.resize(new_face_rgb, (64, 64))
            new_face_rgb = numpy.expand_dims(new_face_rgb, 0)
            feed_dict = {input_wrap:new_face_rgb}
            new_face_rgb, _ = sess.run([pre, pre_mask], feed_dict=feed_dict)

        new_image = convert_one_image(args, new_face, new_face_m, get_sourceface, image, mat)

        output_file = os.path.join(output_dir, image_file)
        cv2.imwrite(str(output_file), new_image)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, nargs='?', default='./image_3')
    parser.add_argument("--alignments", type=str, nargs='?', default='alignments.json')
    parser.add_argument("--output_dir", type=str, nargs='?', default='merged')
    parser.add_argument("--meta_path", type=str, nargs='?', default='./models')
    parser.add_argument("--aligned_path", type=str, nargs='?', default='./image_3/aligned')
    parser.add_argument("--seamlessClone", type=str2bool, nargs='?', const=False, default='False',
                        help="Attempt to use opencv seamlessClone.")

    parser.add_argument("--doublePass", type=str2bool, nargs='?', const=False, default='False',
                        help="Pass the original prediction output back through for a second pass.")

    parser.add_argument('--maskType', type=str, default='FaceHullAndRect',
                        choices=['FaceHullAndRect', 'FaceHull', 'Rect'],
                        help="The type of masking to use around the face.")

    parser.add_argument("--startframe", type=int, default='0')
    parser.add_argument("--frameSkip", type=int, default='1')
    parser.add_argument("--blurSize", type=int, default='4')
    parser.add_argument("--erosionKernelSize", type=int, default='2')
    parser.add_argument("--vision", type=str2bool, nargs='?', const=False, default='False')
    parser.add_argument("--direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
    main(parser.parse_args())

