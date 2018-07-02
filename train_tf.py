import cv2
import numpy
import argparse
import random
from model import model_tf
from tqdm import tqdm
import tensorflow as tf
from model.training_data import get_training_data
from model.utils import get_image_paths, load_images, stack_images
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train_net(images_A, landmark_A, images_B, landmark_B, args):
    print("**********************starting training************************")
    batch_size = args.batch_size
    log_dir = args.log_dir
    with tf.Graph().as_default():
        sess = tf.Session()
        with tf.name_scope('input'):
            input_wrap = tf.placeholder(tf.float32, (None, 64, 64, 3), name='input_wrap')
            mask_A_tf = tf.placeholder(tf.float32, (None, 128, 128, 1), name='mask_A_tf')
            mask_B_tf = tf.placeholder(tf.float32, (None, 128, 128, 1), name='mask_B_tf')
            targ_A_tf = tf.placeholder(tf.float32, (None, 128, 128, 3), name='targ_A_tf')
            targ_B_tf = tf.placeholder(tf.float32, (None, 128, 128, 3), name='targ_B_tf')

        model = model_tf.model_tf(lossFun=args.lossFun, weight_decy=args.weight_decy)

        with tf.name_scope('encoder'):
            encoder = model.encoder(input_wrap)
        with tf.name_scope('decoder_A'):
            A_pre, A_mask, feature_map_A = model.decoder(encoder, 'decoder_A')
        with tf.name_scope('decoder_B'):
            B_pre, B_mask, feature_map_B = model.decoder(encoder, 'decoder_B')

        encoder_weight_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'encoder'),
                                           name='encoder_wight')
        decoder_weight_A = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'decoder_A'),
                                         name='decoder_A_weight')
        decoder_weight_B = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'decoder_B'),
                                         name='decoder_B_weight')

        with tf.name_scope('loss'):
            loss_A = model.loss(mask_A_tf, targ_A_tf, A_pre)
            loss_B = model.loss(mask_B_tf, targ_B_tf, B_pre)
            loss_A_mask = tf.losses.mean_squared_error(mask_A_tf, A_mask)
            loss_B_mask = tf.losses.mean_squared_error(mask_B_tf, B_mask)
            if len(loss_A) == 2:
                loss_A_total = (args.loss_weight_dss * loss_A[0] + loss_A[1] + args.loss_weight_en * encoder_weight_reg
                                + loss_A_mask)
                loss_B_total = (args.loss_weight_dss * loss_B[0] + loss_B[1] + args.loss_weight_en * encoder_weight_reg
                                + loss_B_mask)
                tf.summary.scalar('loss_B_DSSIM', loss_B[0])
                tf.summary.scalar('loss_B_MSE', loss_B[1])
            else:
                loss_A_total = loss_A[0] + 0.1 * loss_A_mask + args.loss_weight_en * encoder_weight_reg
                loss_B_total = loss_B[0] + 0.1 * loss_B_mask + args.loss_weight_en * encoder_weight_reg
                tf.summary.scalar('loss_B', loss_B[0])
                tf.summary.scalar('lossB_mask', loss_B_mask)

        global_step = tf.Variable(0, trainable=False)
        with tf.name_scope('optimizer'):
            lr = tf.train.exponential_decay(learning_rate=args.lr_init, global_step=global_step,
                                            decay_steps=1000, decay_rate=0.90, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999)
            train_op_A = optimizer.minimize(loss_A_total, global_step=global_step)
            train_op_B = optimizer.minimize(loss_B_total)

        split = tf.split(feature_map_B, num_or_size_splits=64, axis=3)
        tf.summary.image('feature_map_B_1', split[4], 4)
        tf.summary.image('feature_map_B_2', split[6], 4)
        tf.summary.scalar('global_step', global_step)
        tf.summary.scalar('encoder_wight', encoder_weight_reg)
        tf.summary.scalar('decoder_B_weight', decoder_weight_B)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        Saver = tf.train.Saver(max_to_keep=args.max_to_keep)
        try:
            Saver.restore(sess, tf.train.latest_checkpoint(args.restore_path))
            print("\n**********************restore over!!**********************\n")
        except:
            init = tf.global_variables_initializer()
            sess.run(init)
        print("************start training now!!!!**************")
        while 1:
            pbar = tqdm(range(1000000))
            for epoch in pbar:
                wraped_A, target_A, mask_A = get_training_data(images_A, landmark_A, landmark_B, batch_size)
                wraped_B, target_B, mask_B = get_training_data(images_B, landmark_B, landmark_A, batch_size)

                train_A_dict = {input_wrap: wraped_A, mask_A_tf: mask_A, targ_A_tf: target_A}
                train_B_dict = {input_wrap: wraped_B, mask_B_tf: mask_B, targ_B_tf: target_B}

                sess.run(train_op_A, feed_dict=train_A_dict)
                loss_A_, loss_A_mask_, total_A, global_A = sess.run([loss_A, loss_A_mask, loss_A_total, global_step],
                                                                    feed_dict=train_A_dict)
                sess.run(train_op_B, feed_dict=train_B_dict)
                loss_B_, loss_B_mask_, total_B, summary = sess.run([loss_B, loss_B_mask, loss_B_total, merged],
                                                                   feed_dict=train_B_dict)
                train_writer.add_summary(summary, global_step=global_A)

                loss_A_.append([loss_A_mask_, total_A])
                loss_B_.append([loss_B_mask_, total_B])
                pbar.set_description("Step:[{}] Loss_A:[{}] Loss_B:[{}]".format(global_A, loss_A_, loss_B_))


                epoch_step = args.save_step
                if epoch % epoch_step == 0 and epoch != 0:
                    Saver.save(sess, global_step=global_A, write_meta_graph=True, save_path=args.save_path)
                    print("Save model done!!!!")

                if args.vision == True:
                    if epoch % 100 == 0:
                        test_A = target_A[0:batch_size, :, :, :3]
                        test_B = target_B[0:batch_size, :, :, :3]
                        test_A_i = []
                        test_B_i = []
                        for i in test_A:
                            test_A_i.append(cv2.resize(i, (64, 64), cv2.INTER_AREA))
                        test_A_i = numpy.array(test_A_i).reshape((-1, 64, 64, 3))
                        for i in test_B:
                            test_B_i.append(cv2.resize(i, (64, 64), cv2.INTER_AREA))
                        test_B_i = numpy.array(test_B_i).reshape((-1, 64, 64, 3))

                    A_pre_A = sess.run(A_pre, feed_dict={input_wrap: test_A_i})
                    B_pre_A = sess.run(A_pre, feed_dict={input_wrap: test_B_i})
                    A_pre_B = sess.run(B_pre, feed_dict={input_wrap: test_A_i})
                    B_pre_B = sess.run(B_pre, feed_dict={input_wrap: test_B_i})

                    A_pre_A = A_pre_A[0:8, :, :, :3]
                    A_pre_B = A_pre_B[0:8, :, :, :3]
                    B_pre_A = B_pre_A[0:8, :, :, :3]
                    B_pre_B = B_pre_B[0:8, :, :, :3]

                    figure_A = numpy.stack([test_A[0:8], A_pre_A, A_pre_B], axis=1)
                    figure_B = numpy.stack([test_B[0:8], B_pre_B, B_pre_A], axis=1)

                    figure = numpy.concatenate([figure_A, figure_B], axis=0)
                    figure = figure.reshape((4, 4) + figure.shape[1:])
                    figure = stack_images(figure)
                    figure = numpy.clip(figure * 255, 0, 255).astype('uint8')
                    cv2.imshow("p", figure)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        exit()
            train_writer.close()

def get_image_batch(args):
    images_A = get_image_paths(args.data_A)
    images_B = get_image_paths(args.data_B)

    minImages = args.minImage
    random.shuffle(images_A)
    random.shuffle(images_B)

    images_A, landmark_A = load_images(images_A[:minImages])
    images_B, landmark_B = load_images(images_B[:minImages])

    print("Images A", images_A.shape)
    print("Images B", images_B.shape)

    images_A = images_A / 255.0
    images_B = images_B / 255.0

    images_A[:, :, :3] += images_B[:, :, :3].mean(axis=(0, 1, 2)) - images_A[:, :, :3].mean(axis=(0, 1, 2))
    return images_A, landmark_A, images_B, landmark_B

def train(args):
    images_A, lanmark_A, images_B, lanmark_B = get_image_batch(args)
    train_net(images_A, lanmark_A, images_B, lanmark_B, args)

if __name__ == '__main__':
    print('running!!!!')
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_A", type=str, default='./data/A')
    parser.add_argument("--data_B", type=str, default='./data/B')
    parser.add_argument("--minImage", type=int, default='1600')
    parser.add_argument("--batch_size", type=int, default='16')
    parser.add_argument("--lossFun", type=str, default='Dssim')
    parser.add_argument("--weight_decy", type=float, default='0.0001')
    parser.add_argument("--loss_weight_en", type=float, default='0.001')
    parser.add_argument("--loss_weight_dss", type=float, default='0.2')
    parser.add_argument("--lr_init", type=float, default='0.0001')
    parser.add_argument("--max_to_keep", type=int, default='2')
    parser.add_argument("--restore_path", type=str, default='./models/')
    parser.add_argument("--save_path", type=str, default='./models/')
    parser.add_argument("--log_dir", type=str, default='./Tensorboard')
    parser.add_argument("--save_step", type=int, default='500')
    parser.add_argument("--vision", type=bool, default=True)
    args = parser.parse_args()
    train(args)




