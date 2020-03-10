# Style Transfer with VGG19

In this work, I perform style transfer with pre-trained VGG19. I use Python 3.6.7, TensorFlow 2.0, Numpy and Matplotlib.

## Details

I use the following layers of VGG19 to retrieve style features: `block1_conv1`, `block2_conv1`, `block3_conv1`, `block4_conv1`, `block5_conv1`.

To extract content features we use the layer `block5_conv2`.

## Results

Below, given the results after 1000 iterations (result, style, content):

![alt text](https://github.com/yernat-assylbekov/Style-Transfer-with-VGG19/blob/master/images/result1.jpg?raw=true)

![alt text](https://github.com/yernat-assylbekov/Style-Transfer-with-VGG19/blob/master/images/result2.jpg?raw=true)

![alt text](https://github.com/yernat-assylbekov/Style-Transfer-with-VGG19/blob/master/images/result3.jpg?raw=true)

![alt text](https://github.com/yernat-assylbekov/Style-Transfer-with-VGG19/blob/master/images/result4.jpg?raw=true)

![alt text](https://github.com/yernat-assylbekov/Style-Transfer-with-VGG19/blob/master/images/result5.jpg?raw=true)

## Usage

Just run `style_transfer.py path_to_content path_to_style` where `path_to_content` and `path_to_style` are paths to content and style images.

## References

<a href="https://arxiv.org/pdf/1409.1556.pdf">[1]</a> K. Simonyan and A. Zisserman, <i>Very Deep Convolutional Networks for Large-Scale Image Recognition Karen</i>, ICLR (2015).
