# corolization

AI agent to colorize b&w images

## Motivations

Back when we were in high school, there were a lot of shops where
you can hire someone to colour old photos of your childhood where cameras
had not been as good. And do you know that there is a [subreddit](https://www.reddit.com/r/Colorization/) dedicated to colorizing black and white photos. It would take an artist quite an amount of time ranging from a few hours to a few days to colorize a b&w photo.

Imagine if we have the power to instantaneously convert b&w into vibrant, colorful photos. We would be able to breathe life into hundreds of thousands of photos, offering the viewers an more immersive experience looking at those photos from a different era. An automated colorizer would also be very useful for other purposes like enhancing/color-correcting photos.

We aim to produce an automated colorizer using deep learning computer vision techniques to produce colorful images from b&w photos, that look reasonably good with the actual outside world as a benchmark.

## The saga of Corolization

This is the story about how our colorizer has evolved over time:

### The rudimentary generator

Our first attempt was to use a generator on the b&w image input and generate every pixel of the output image as RGB values. We tested the idea out using a simple neural network with multiple fully-connected layers. After about 900 epochs of training, the net seemed to have converged and produced this

![rudimentary_generator](pictures/rudimentary_gen.png)
<br/>_Test case from CIFAR-10 dataset_

From the example above, we can see that the neural net has learnt a bit to immitate the general shape of the object in the b&w image, however, the result generated is far from what is considered "comprehensible" by human vision.

We realised that it is quite tedious to train a "from scratch" generator with the b&w input to perform decently. It would be more reasonable to leverage on what we already have as input (b&w image) and generate the color layers. This would require using a colour space that has "lightness" as one channel, such as HSL, YUV or Lab, so that the neural net can be trained to output the 2 colour layers, and we just concatenate the lightness layer to get the final image. We decided to try this approach out. 

### The "too-generic" colorizer

We were looking for an inspiration when we stumbled on a [blog post](http://tinyclouds.org/colorize/) by [Ryan Dahl](https://github.com/ry). We tried implementing his proposed approach of a residual encoder backed with VGG-16.

![residual_encoder](pictures/residual_encoder.png)

As the name suggests, this architecture convolves the input through multiple layers (that are drawn from VGG-16), and at the same time upscales the output to add back to the previous output to be use for further upscaling. The final output (a pair of UV layers) will then be added back to Y-layer input. There is a huge drawback in terms of resource for this model. It would take a few hours to cycle through one epoch of [SUN dataset](https://groups.csail.mit.edu/vision/SUN/). Furthermore, the improvement is rather slow and incremental, after a few epochs of the dataset, we only achieved uniformly sepia-toned outputs. Even after more training epochs, we only achieved rather desaturated results, which is not close to what could be a good guess of colors and would be too "ugly" for human cognition. 

We decided that this approach would be not robust enough and too slow for us to keep re-iterating over the course of our limited timeframe for this project, so we moved on to a new approach.

### The colorful colorizer

After looking out for a few more inspiration, we settled on a [research idea](http://richzhang.github.io/colorization/) from UC Berkerley [Rich Zhang](http://richzhang.github.io/). The neural net architecture that was explored is rather simple and can be summed up with this diagram below

![colorful colorizer](pictures/colorful_colorizer.jpg)

This approach is rather simple: Taking in the L-layer (in a LAB color scheme) of an image as input, convolute it through 8 convolutional blocks, then generate a probability distribution of colors for each pixels (blue block) and finally output the actual pair of AB-layers and upscale to full-size color layers.

This research approach is actually trying to tackle the problem of de-saturated coloring guess from precedent CNN attempts to colorize photos. Most of magic lies inside the way the loss is formulated. In previous attempts, the loss function is based on Euclidean distance between the predicted colour and ground truth colour of each pixel. However, for objects that can be one of many different colours (eg. a ball can be any colour), the neural network will tend to predict something close to the average of the different coloured objects it was trained on. This will result in a more neutral prediction output.

Hence, in this paper, the problem is treated as a multimodal classification problem. Rather than outputting color values directly like other approaches, all possible AB pairs of values are put into 313 different "bins" (or can be preceived as categories) and converted back to actual color values during prediction. Training labels are also soft-encoded using nearest neighbors based on this color table belows.

![colors](pictures/colors.png)

We went ahead and implemented this neural net and achieved this decent result after 47 epochs.

![wo rebal 47](<pictures/10_apr-(3.04)-47epoch_vgood.png>)
<br/>_From left to right (input, prediction, actual image)_

The result was not much better than the previously mentioned residual encoder, however, it took a lot less time to train. Nevertheless, one problem with this is that it always predicts the general colors, which works well for things like trees, sky that tend to always be around the same colour. However, it doesn't work as well for non-generic things (which could be of any colors) as they tend to be of "rare" colours, which the neural net is unable to accurately predict. This results in very neutral-coloured output images especially for indoor scenes.

![wo rebal 47](<pictures/10_apr-(3.04)-47epoch_indoor.png>)
<br/>_From left to right (input, prediction, actual image)_

As a result, we draw another technique from the paper called "class rebalancing". For this, we calculate weights for each color bin using the prior distribution of colors of the whole dataset, using this formula:

![formula](pictures/formula_rebal.png)
<br/>_q is the color bin, Z is the one-hot label at corresponding pixel_

The basic idea behind this formula is to rebalance the weightage of each color during the training process by giving those that appear more often less weightage and vice versa. We implemented this and trained the network through a few epochs.

![w rebal](pictures/10_apr-rebal3.png)
<br/>_From left to right (input, prediction, actual image)_

Ignoring the fact that we only trained it for a few epochs (which explains why the predictions are not very good), we can observe that the neural net now outputs more "daring" guesses, making the predictions look more colorful and life-like.

At this moment, we also realised another problem with the current approach is that the prediction often overflows from one object to another. After a long process of investigation by overfitting the neural net into one image and observing that the predictions still overflow, we found out that the reason is because all parameters were initialized from a uniform distribution (same starting point!) as the default from PyTorch. We changed the initilization method into Xavier intilization, which produced a very good distition between objects when overfitted.

![overfitted](pictures/overfit_2.png)
<br/>_From left to right (input, prediction, actual image)_

## Tuning the "hyper-paradio" (hyper-parameters)

We ultilized a specific technique of plotting traning loss for every batch with different learning rates ([reference](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)). We plotted multiple ranges of learning rates but from this best plot of learning rates ranging from 0.1 to 10, we managed to find the optimal learning rate for our approach to be about 0.5 and went ahead with it for training

![lr plot](pictures/lr_finder_10percent.png)

## Further improvement

We have achieved decent predictions with our current approach but we are aware that there is still space for us to improve our model. One way we thought of is to use Generative Adversarial Networks (GAN). The "game" can be formulated such that our current neural net will be the "forger", and there will be an additional network acting as the "detector", trying to distinguish between output from the "forger" and the actual image. However, due to the time constraints of this project, we did not have the chance to put this into practice.

## Technologies

* [PyTorch](http://pytorch.org/) (It's an awesome, python-first deep learning framework)
* [sk-learn](http://scikit-learn.org/) (Our casual friend)
* [sk-image](http://scikit-image.org/) (Python image magician)

## Team Members

* Vivek Kalyan
* Teo Si-Yan
* Stanley Nguyen
* Jo Hsi Keong
* Tan Li Xuan

## Acknowledgement

_SUTD 50.035 Computer Vision Instructors and TAs:_

* Prof. Ngai-Man (Man) Cheung
* Prof. Gemma Roig
* Tuan Nguyen Anh Hoang
