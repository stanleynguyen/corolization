# corolization

AI agent to colorize b&w images

## Motivation

Back in high school, there were plenty of shops which let you hire artists to colour old childhood photographs from an age where camera technology had been much more primitive. Today, the enduring appeal of colour has sparked a [subreddit](https://www.reddit.com/r/Colorization/) dedicated to colorizing black-and-white photos.

Yet it takes an artist a considerable amount of time to colour a black-and-white photograph. Imagine if we had the power to instantaneously transform black-and-white photographs into vibrant, colourful ones - to breathe life into relics, to journey through time! An automated colorizer would also come in handy for enhancing and correcting photos.

Our hope is to create an automated colourizer using deep learning techniques to generate coloured images from black-and-white photos that are reasonably convincing, with the real world as a benchmark.

## The saga of Corolization

This is the story about how our colorizer has evolved over time:

### The rudimentary generator

Our first attempt was to use a generator on the black-and-white image input and generate every pixel of the output image as RGB values. We tested the idea on a simple neural network with multiple fully-connected layers. After about 900 epochs of training, the net seemed to converged and produced the following results

![rudimentary_generator](pictures/rudimentary_gen.png)
<br/>_Test case from CIFAR-10 dataset_

From the example above, we can see that the neural net has learnt a bit to imitate the general shape of the object in the black-and-white image, although the result is far from "comprehensible" by standards of human vision.

We realised that it is quite tedious to train a "from scratch" generator with the black-and-white input to perform decently. It would be more reasonable to leverage on what we already have as input (black-and-white image) and generate the colour layers. This would require using a colour space that has "lightness" as one of its channels, such as HSL, LUV or Lab, so that the neural net can be trained to output the 2 colour layers, and we would just have to concatenate the lightness layer to get the final image. We decided to try out this approach.

### The "over-generic" colorizer

We were looking for an inspiration when we stumbled on a [blog post](http://tinyclouds.org/colorize/) by [Ryan Dahl](https://github.com/ry). We tried implementing his proposed approach of a residual encoder backed with VGG-16.

![residual_encoder](pictures/residual_encoder.png)

As its name suggests, the architecture convolves the input through multiple layers drawn from VGG-16. At the same time, it upscales the output and adds to the previous output for further upscaling. The final output (a pair of UV layers) is then added back to the Y-layer input. There is a huge drawback in terms of resource for this model. It would take several hours to cycle through one epoch of [SUN dataset](https://groups.csail.mit.edu/vision/SUN/). Furthermore, improvement is slow and incremental - after a few epochs of the dataset, we achieved only uniformly sepia-toned outputs. Increasing the number of training epochs produced no further improvement, with desaturated colours hardly convincing to human cognition.

We decided that this approach would be not robust enough and too slow for us to keep re-iterating over the course of the limited timeframe for this project, so we moved on to a new approach.

### The colorful colorizer

After looking around for more inspiration, we settled on a [research idea](http://richzhang.github.io/colorization/) from UC Berkerley's [Rich Zhang](http://richzhang.github.io/). The neural net architecture proposed can be summed up in the diagram below:

![colorful colorizer](pictures/colorful_colorizer.jpg)

The architecture is straightforward - the net takes in the L-layer (in a Lab colour scheme) of an image as input, convolves it through 8 convolutional blocks, generates a probability distribution of colors for each pixel (blue block) and finally outputs the actual pair of ab-layers and upscales them to full-sized color layers.

This approach tackles the problem of desaturated guesses from precedent CNN attempts to colourize photos. Much of the magic lies in the way the loss is formulated. In precedent attempts, the loss functions were calculated as the Euclidean distance between the predicted colour and ground truth colour of each pixel. This means that for any object, the neural network will tend to predict something close to the average range of all the different colour and saturation values for the particular object it has been trained on. For objects that present a wide variety of colours and saturations, this averaging effect tends to favour grayish and desaturated predictions.

In this paper, colorization is treated as a multimodal classification problem. Rather than directly outputting colour values like in other approaches, all possible ab pairs of values are put into 313 different "bins" or categories and converted back to actual colour values during prediction. Training labels are also soft-encoded using the nearest neighbors algorithm based on the colour table shown below.

![colors](pictures/colors.png)

We implemented this neural net and achieved a decent result after 47 epochs.

![wo rebal 47](<pictures/10_apr-(3.04)-47epoch_vgood.png>)
<br/>_From left to right (input, prediction, actual image)_

Although the result was not much better than that of the residual encoder, training time was reduced significantly. The problem with this approach is that it always predicts general colors. While this works well for objects that tend always to be the same colour such as trees and skies it is unable to accurately predict the colours of non-generic entities, which might present in rare colours. This resulted in neutral-coloured outputs, especially for indoor scenes.

![wo rebal 47](<pictures/10_apr-(3.04)-47epoch_indoor.png>)
<br/>_From left to right (input, prediction, actual image)_

As a result, we drew another technique from the paper called "class rebalancing". For this, we calculate weights for each color bin using the prior distribution of colours from the whole dataset using this formula:

![formula](pictures/formula_rebal.png)
<br/>_q is the color bin, Z is the one-hot label at corresponding pixel_

The idea behind this formula is to reweight the loss of each pixel during the training process by reducing the weightage on colours that appear more frequently and increasing the weightage on rare colours. Training the network through several epochs produced the following results:

![w rebal](pictures/10_apr-rebal3.png)
<br/>_From left to right (input, prediction, actual image)_

Notwithstanding the fact that it was only trained for a few epochs (which explains why the predictions are not very good), the neural net made more "daring" guesses, resulting in predictions that were more colourful and life-like.

At this point, we discovered yet another problem - the prediction frequently overflows from one object to another. After a long process of investigation which involved overfitting the neural net into one image and observing that the predictions still overflow, we traced the reason to the fact that all parameters had been initialized from a uniform distribution (same starting point!) as the default from PyTorch. We thus changed our initialization method to the Xavier intilization, which produced a very good distinction between objects when overfitted.

![overfitted](pictures/overfit_2.png)
<br/>_From left to right (input, prediction, actual image)_

## Tuning the "hyper-paradio" (hyper-parameters)

We ultilized a specific technique of plotting traning loss for every batch with different learning rates ([reference](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)). We plotted multiple ranges of learning rates and from this best plot of learning rates ranging from 0.1 to 10, found the optimal learning rate for our approach to be approximately 0.5 and incorporated it into our training.

![lr plot](pictures/lr_finder_10percent.png)

## Further improvements

While our current model yields decent predictions, there is much room for improvement. One such was to introduce Generative Adversarial Networks. The "game" could be formulated such that our current neural net will be the "forger", and an additional network the "detector" which tries to distinguish between an output from the "forger" and the actual image. However, due to time constraints, we did not have a chance to implement this.

## Technologies

* [PyTorch](http://pytorch.org/) (An awesome, python-first deep learning framework)
* [sk-learn](http://scikit-learn.org/) (Our casual friend)
* [sk-image](http://scikit-image.org/) (Python image magician)

## Team Members

* Vivek Kalyan
* Teo Si-Yan
* Stanley Nguyen
* Jo Hsi Keong
* Tan Li Xuan

## Acknowledgements

_SUTD 50.035 Computer Vision Instructors and TAs:_

* Prof. Ngai-Man (Man) Cheung
* Prof. Gemma Roig
* Tuan Nguyen Anh Hoang
