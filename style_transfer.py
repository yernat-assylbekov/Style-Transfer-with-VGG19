"""
author: Yernat M. Assylbekov
email: yernat.assylbekov@gmail.com
date: 03/10/2020
"""

import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt

from IPython import display
from time import time
import sys

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


def features_extractor(style_layers, content_layer):
    """
    creates pretrained VGG19 model and extracts the outputs of its layers corresponding to style and content.
    ------------------------------------------------
    Input:
        style_layers -- list of strings containing names of style layers of VGG19;
        content_layer -- string, name of content layer of VGG19.

    Output:
        model -- tf.keras.Model that extracts outputs of style and content layers of input image.
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [[vgg.get_layer(layer_name).output for layer_name in style_layers], vgg.get_layer(content_layer).output]

    model = Model(inputs=vgg.input, outputs=outputs)
    return model

def gram_matrix(input_tensor):
    """
    for each element in a batch computes gram matrix among different channels
    (middle two indices are considered as coordinates while last index corresponds to different vectors).
    ------------------------------------------------
    Input:
        input_tensor -- numpy array of dimension 4.

    Output:
        numpy array of dimension 3.
    """
    return tf.linalg.einsum('mijc, mijd -> mcd', input_tensor, input_tensor)

def content_cost(content_C, content_G):
    """
    computes content cost (extracted from content layer of VGG19) between content and generated images.
    ------------------------------------------------
    Input:
        content_C, content_G -- numpy arrays extracted from content layer of VGG19, have same shapes.

    Output:
        J_content -- content cost.
    """
    m, n_H, n_W, n_C = content_C.shape

    J_content = tf.math.reduce_sum((content_C - content_G) ** 2) / (4 * n_H * n_W * n_C)

    return J_content

def layer_style_cost(layer_style_C, layer_style_S):
    """
    computes style cost for current style layer (extracted from VGG19) between content and style images.
    ------------------------------------------------
    Input:
        layer_style_C, layer_style_S -- numpy arrays extracted from a style layer of VGG19, have same shapes.

    Output:
        J_style_layer -- style cost for current layer.
    """
    m, n_H, n_W, n_C = layer_style_C.shape

    G_C = gram_matrix(layer_style_C)
    G_S = gram_matrix(layer_style_S)

    J_style_layer = tf.math.reduce_sum((G_C - G_S) ** 2) / (4 * (n_H * n_W * n_C) ** 2)

    return J_style_layer

def style_cost(styles_S, styles_G):
    """
    computes total style cost for all style layers (extracted from VGG19) between content and style images.
    ------------------------------------------------
    Input:
        styles_S, styles_G -- lists of numpy arrays representing style layers extracted from VGG19, have same shapes.

    Output:
        J_style -- total style cost.
    """
    J_style = 0

    for layer in range(len(style_layers)):
        J_style += layer_style_cost(styles_S[layer], styles_G[layer])

    return J_style

def style_transfer(C, S, weight_C, weight_S, weight_TV, steps):
    """
    performs transfer of content image to style of style image.
    ------------------------------------------------
    Input:
        C, S -- numpy arrays representing content and style images both of shape (512, 512, 3);
        weight_C, weight_S -- weights for content and style, respectively;
        weight_TV -- weight for TV regularization of generated image;
        steps -- int, number of iteration steps.

    Output:
        saves C, S and G (generated image) in a single jpg file named 'result.jpg'.
    """
    # feature extractor is given globally
    global extractor

    # extract style and content features from style and content images (C and S)
    styles_C, content_C = extractor(C)
    styles_S, content_S = extractor(S)

    # initiate generated image as the content image C
    G = tf.Variable(C)

    # print the original content image C
    print('Original image')
    fig = plt.figure(figsize=(7, 7))
    plt.imshow(G[0])
    plt.axis('off')
    plt.show()

    # create an instance of Adam optimizer
    optimizer = Adam(learning_rate=0.025, beta_1=0.99, epsilon=1e-1)

    for step in range(1, steps + 1):

        # watch trainable variables (pixels of G) for the total loss
        with tf.GradientTape() as t:
            # extract style and content features from the generated image G
            styles_G, content_G = extractor(G)

            # compute total loss function
            loss = weight_C * content_cost(content_C, content_G) + weight_S * style_cost(styles_S, styles_G) + weight_TV * tf.image.total_variation(G)

        # compute gradients and perform one step gradient descend
        Grads = t.gradient(loss, G)
        optimizer.apply_gradients([(Grads, G)])
        G.assign(tf.clip_by_value(G, clip_value_min=0.0, clip_value_max=1.0))

        # print the generated image at every 20 epochs
        if step % 20 == 0:
            display.clear_output(wait=True)
            print('Step: {}'.format(step))
            fig = plt.figure(figsize=(7, 7))
            plt.imshow(G[0])
            plt.axis('off')
            plt.show()

    # print and save the final result: G, S, and C
    display.clear_output(wait=True)
    print('Final result comparison:')
    fig = plt.figure(figsize=(21, 7))
    plt.subplot(1, 3, 1)
    plt.imshow(G[0])
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(S[0])
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(C[0])
    plt.axis('off')
    plt.savefig('/images/result.jpg')
    plt.show()

# fix content and style layers
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layer = 'block5_conv2'

# create an instance of feature extractor
extractor = features_extractor(style_layers, content_layer)

# read paths to content and style images
_, path_to_content, path_to_style = sys.argv

# read content and style images
size = (512, 512)
C = Image.open(path_to_content).resize(size)
C = np.asarray(C, dtype=np.float32) / 255.

S = Image.open(path_to_style).resize(size)
S = np.asarray(S, dtype=np.float32) / 255.

# print content and style images
fig = plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(C)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(S)
plt.axis('off')
plt.show()

# add one more dimension from left (just for technical reasons)
C = np.expand_dims(C, axis=0)
S = np.expand_dims(S, axis=0)

# initiate weights and number of iteration steps
weight_C, weight_S, weight_TV = 1e4, 5e1, 0.0025
steps = 1000

# perform style transfer
tic = time()
style_transfer(C, S, weight_C, weight_S, weight_TV, steps)
print('Total time: {} minutes'.format((time() - tic) / 60.))
