import tensorflow as tf
import tensorflow.contrib.slim as slim
import Dssim


class model_tf(object):
    def __init__(self, lossFun='MAE', weight_decy=0.01):
        self.name = 'unnamed_decoder'
        self.lossFun = lossFun
        self.weight_decy = weight_decy

    def pixelshuffel(self, inputs):
        input_shape = inputs.shape.as_list()
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        batch_size, h, w, c = input_shape
        if batch_size is None:
            batch_size = -1
        rh = 2
        rw = 2
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)
        out = tf.reshape(inputs, (batch_size, h, w, rh, rw, oc))
        out = tf.transpose(out, (0, 1, 3, 2, 4, 5))
        out = tf.reshape(out, (batch_size, oh, ow, oc))
        return out

    def res_block(self, inputs_tensor, f):
        with slim.arg_scope([slim.conv2d], activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(self.weight_decy),
                            biases_initializer=False, ):
            net = inputs_tensor
            # net = slim.repeat(inputs=net, repetitions=2, layer=conv2d, f, [3, 3], padding ="SAME")
            net = slim.conv2d(inputs=net, num_outputs=f, kernel_size=3, padding='SAME')
            net = tf.nn.leaky_relu(net, alpha=0.2)
            net = slim.conv2d(inputs=net, num_outputs=f, kernel_size=3, padding='SAME')
            net = tf.add(net, inputs_tensor)
            net = tf.nn.leaky_relu(net, alpha=0.2)
            return net

    def encoder(self, inputs):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(self.weight_decy)):
            net = slim.conv2d(inputs, num_outputs=128, kernel_size=5, stride=2, padding='SAME', scope='encode_cov1')
            net = tf.nn.leaky_relu(net, alpha=0.1)
            net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=5, stride=2, padding='SAME', scope='encode_cov2')
            net = tf.nn.leaky_relu(net, alpha=0.1)
            net = slim.conv2d(inputs=net, num_outputs=512, kernel_size=5, stride=2, padding='SAME', scope='encode_cov3')
            net = tf.nn.leaky_relu(net, alpha=0.1)
            net = slim.conv2d(inputs=net, num_outputs=1024, kernel_size=5, stride=2, padding='SAME', scope='encode_cov4')
            net = tf.nn.leaky_relu(net, alpha=0.1)
            net = slim.flatten(net)
            net = slim.fully_connected(net, num_outputs=1024, activation_fn=None, scope='encode_fc1')
            net = slim.fully_connected(net, num_outputs=4*4*1024, activation_fn=None, scope='encode_fc2')
            net = tf.reshape(net, (-1, 4, 4, 1024))
            net = slim.conv2d(net, num_outputs=512*4, kernel_size=3, padding='SAME', scope='encode_cov5')
            net = tf.nn.leaky_relu(net, alpha=0.1)
            net = self.pixelshuffel(net)
            return net

    def decoder(self, inputs, name):
        self.name = name
        net_x = inputs
        with slim.arg_scope([slim.conv2d], activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(self.weight_decy)):
            net_x = slim.conv2d(net_x, num_outputs=512 * 4, kernel_size=3,  padding='SAME')
            net_x = tf.nn.leaky_relu(net_x, alpha=0.1)
            net_x = self.pixelshuffel(net_x)
            net_x = self.res_block(net_x, 512)
            net_x = slim.conv2d(net_x, num_outputs=256 * 4, kernel_size=3,  padding='SAME')
            net_x = tf.nn.leaky_relu(net_x, alpha=0.1)
            net_x = self.pixelshuffel(net_x)
            net_x = self.res_block(net_x, 256)
            net_x = slim.conv2d(net_x, num_outputs=128 * 4, kernel_size=3, padding='SAME')
            net_x = tf.nn.leaky_relu(net_x, alpha=0.1)
            net_x = self.pixelshuffel(net_x)
            net_x = self.res_block(net_x, 128)
            net_x = slim.conv2d(net_x, num_outputs=64 * 4, kernel_size=3, padding='SAME')
            net_x = tf.nn.leaky_relu(net_x, alpha=0.1)
            net_x = self.pixelshuffel(net_x)
            feature_out = net_x
            net_x = slim.conv2d(net_x, num_outputs=3, kernel_size=5, padding='SAME')
            net_x = tf.nn.sigmoid(net_x, name=self.name + '_pre')

        net_y = inputs
        with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(self.weight_decy)):
            net_y = slim.conv2d(net_y, num_outputs=512 * 4, kernel_size=3, padding='SAME')
            net_y = tf.nn.leaky_relu(net_y, alpha=0.1)
            net_y = self.pixelshuffel(net_y)
            net_y = slim.conv2d(net_y, num_outputs=256 * 4, kernel_size=3, padding='SAME')
            net_y = tf.nn.leaky_relu(net_y, alpha=0.1)
            net_y = self.pixelshuffel(net_y)
            net_y = slim.conv2d(net_y, num_outputs=128 * 4, kernel_size=3, padding='SAME')
            net_y = tf.nn.leaky_relu(net_y, alpha=0.1)
            net_y = self.pixelshuffel(net_y)
            net_y = slim.conv2d(net_y, num_outputs=64 * 4, kernel_size=3, padding='SAME')
            net_y = tf.nn.leaky_relu(net_y, alpha=0.1)
            net_y = self.pixelshuffel(net_y)
            net_y = slim.conv2d(net_y, num_outputs=1, kernel_size=5, padding='SAME')
            net_y = tf.nn.sigmoid(net_y, name=self.name + '_mask')
        return net_x, net_y, feature_out

    def loss(self, mask, y_true, y_pred, maskProp = 1.0):
        tro, tgo, tbo = tf.split(y_true, 3, 3)
        pro, pgo, pbo = tf.split(y_pred, 3, 3)

        tr = tro
        tg = tgo
        tb = tbo

        pr = pro
        pg = pgo
        pb = pbo
        m  = mask

        m   = m* maskProp
        m  += (1 - maskProp)
        tr *= m
        tg *= m
        tb *= m

        pr *= m
        pg *= m
        pb *= m

        y = tf.concat([tr, tg, tb],3)
        p = tf.concat([pr, pg, pb],3)

        if self.lossFun == 'Dssim':
            dssim = Dssim.DSSIMObjective()
            loss = [dssim.compute_loss(y, p)]
        elif self.lossFun == 'Dssim_mse':
            dssim = Dssim.DSSIMObjective()
            dss_loss = dssim.compute_loss(y, p)
            mse_loss = tf.losses.mean_squared_error(y, p)
            loss = [dss_loss, mse_loss]
        elif self.lossFun == 'mse':
            loss = [tf.losses.mean_squared_error(y, p)]
        else:
            loss = [tf.losses.absolute_difference(y, p)]
        return loss

























