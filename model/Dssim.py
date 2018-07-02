import tensorflow as tf
import numpy as np

class DSSIMObjective():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2

    def __int_shape(selfself, x):
        return x.shape.as_list()

    def extract_image_patches(self, x, ksizes, ssizes, padding='VALID', data_format='channels_last'):
        kernel = [1, ksizes[0], ksizes[1], 1]
        strides = [1, ssizes[0], ssizes[1], 1]
        if data_format == 'channels_first':
            x = tf.transpose(x, (0, 2, 3, 1))
        bs_i, w_i, h_i, ch_i = self.__int_shape(x)
        patches = tf.extract_image_patches(x, kernel, strides, [1, 1, 1, 1], padding=padding)
        bs, w, h, ch = self.__int_shape(patches)
        patches = tf.reshape(tf.transpose(tf.reshape(patches, [-1, w, h, tf.floordiv(ch, ch_i), ch_i]), [0, 1, 2, 4, 3]),
                             [-1, w, h, ch_i, ksizes[0], ksizes[1]])
        if data_format == 'channels_last':
            patches = tf.transpose(patches, [0, 1, 2, 4, 5, 3])
        return patches

    def var(self, x, axis=None, keepdims=False):
        if x.dtype.base_dtype == tf.bool:
            x = tf.cast(x, tf.float32)
        m = tf.reduce_mean(x, axis, True)
        devs_squared = tf.square(x - m)
        return tf.reduce_mean(devs_squared, axis, keepdims)

    def compute_loss(self, y_true, y_pred):
        kernel = [self.kernel_size, self.kernel_size]
        y_true = tf.reshape(y_true, [-1] + self.__int_shape(y_pred)[1:])
        y_pred = tf.reshape(y_pred, [-1] + self.__int_shape(y_pred)[1:])

        patches_pred = self.extract_image_patches(y_pred, kernel, kernel, padding='VALID', data_format='channels_last')
        patches_true = self.extract_image_patches(y_true, kernel, kernel, padding='VALID', data_format='channels_last')

        bs, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        patches_pred = tf.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_true = tf.reshape(patches_true, [-1, w, h, c1 * c2 * c3])

        u_true = tf.reduce_mean(patches_true, axis=-1)
        u_pred = tf.reduce_mean(patches_pred, axis=-1)
        var_true = self.var(patches_true, axis=-1)
        var_pred = self.var(patches_pred, axis=-1)

        covar_true_pred = tf.reduce_mean(patches_pred * patches_true, axis=-1) - u_pred * u_true
        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = (tf.square(u_true) + tf.square(u_pred) + self.c1) * (var_pred + var_true + self.c2)
        ssim /= denom
        return tf.reduce_mean((1.0 - ssim) / 2.0)




