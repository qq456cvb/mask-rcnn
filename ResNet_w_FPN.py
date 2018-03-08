import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.keras as keras


def identity_block(input, channel_in, channel_out, kernel_size=3, stride=1):
    a = slim.conv2d(inputs=input, num_outputs=channel_in, kernel_size=1, stride=stride, activation_fn=None)
    a = slim.batch_norm(inputs=a, is_training=False, scale=True, epsilon=0)
    a = tf.nn.relu(a)

    b = slim.conv2d(inputs=a, num_outputs=channel_in, kernel_size=kernel_size, stride=1, activation_fn=None)
    b = slim.batch_norm(inputs=b, is_training=False, scale=True, epsilon=0)
    b = tf.nn.relu(b)

    c = slim.conv2d(inputs=b, num_outputs=channel_out, kernel_size=1, stride=1, activation_fn=None)
    c = slim.batch_norm(inputs=c, is_training=False, scale=True, epsilon=0)
    return tf.nn.relu(c + input)


def branch_block(input, channel_in, channel_out, kernel_size=3, stride=2):
    a1 = slim.conv2d(inputs=input, num_outputs=channel_in, kernel_size=1, stride=stride, activation_fn=None)
    a1 = slim.batch_norm(inputs=a1, is_training=False, scale=True, epsilon=0)
    a1 = tf.nn.relu(a1)

    b1 = slim.conv2d(inputs=a1, num_outputs=channel_in, kernel_size=kernel_size, stride=1, activation_fn=None)
    b1 = slim.batch_norm(inputs=b1, is_training=False, scale=True, epsilon=0)
    b1 = tf.nn.relu(b1)

    c1 = slim.conv2d(inputs=b1, num_outputs=channel_out, kernel_size=1, stride=1, activation_fn=None)
    c1 = slim.batch_norm(inputs=c1, is_training=False, scale=True, epsilon=0)

    a2 = slim.conv2d(inputs=input, num_outputs=channel_out, kernel_size=1, stride=stride, activation_fn=None)
    a2 = slim.batch_norm(inputs=a2, is_training=False, scale=True, epsilon=0)

    return tf.nn.relu(c1 + a2)


def forward(input):
    with tf.variable_scope('resnet_backbone'):
        C1 = slim.conv2d(inputs=input, num_outputs=64, kernel_size=7, padding='SAME', stride=2, activation_fn=None)
        C1 = slim.batch_norm(C1, is_training=False, scale=True, epsilon=0)
        C1 = tf.nn.relu(C1)
        C1 = slim.max_pool2d(C1, 3, padding='SAME')

        C2 = branch_block(C1, 64, 256, stride=1)
        C2 = identity_block(C2, 64, 256)
        C2 = identity_block(C2, 64, 256)

        C3 = branch_block(C2, 128, 512)
        C3 = identity_block(C3, 128, 512)
        C3 = identity_block(C3, 128, 512)
        C3 = identity_block(C3, 128, 512)

        C4 = branch_block(C3, 256, 1024)
        C4 = identity_block(C4, 256, 1024)
        C4 = identity_block(C4, 256, 1024)
        C4 = identity_block(C4, 256, 1024)
        C4 = identity_block(C4, 256, 1024)
        C4 = identity_block(C4, 256, 1024)

        C5 = branch_block(C4, 512, 2048)
        C5 = identity_block(C5, 512, 2048)
        C5 = identity_block(C5, 512, 2048)

        # FPN
        P5 = slim.conv2d(inputs=C5, num_outputs=256, kernel_size=1, activation_fn=None)
        P4 = keras.layers.UpSampling2D()(P5) + slim.conv2d(inputs=C4, num_outputs=256, kernel_size=1, activation_fn=None)
        P3 = keras.layers.UpSampling2D()(P4) + slim.conv2d(inputs=C3, num_outputs=256, kernel_size=1, activation_fn=None)
        P2 = keras.layers.UpSampling2D()(P3) + slim.conv2d(inputs=C2, num_outputs=256, kernel_size=1, activation_fn=None)

        # reduce upsampling effects
        P2 = slim.conv2d(inputs=P2, num_outputs=256, kernel_size=3, activation_fn=None)
        P3 = slim.conv2d(inputs=P3, num_outputs=256, kernel_size=3, activation_fn=None)
        P4 = slim.conv2d(inputs=P4, num_outputs=256, kernel_size=3, activation_fn=None)
        # unnecessary for P5, but for consistency
        P5 = slim.conv2d(inputs=P5, num_outputs=256, kernel_size=3, activation_fn=None)

        # one more layer for the coarsest scale of anchor
        P6 = slim.avg_pool2d(P5, 3, padding='SAME')
        return P2, P3, P4, P5, P6