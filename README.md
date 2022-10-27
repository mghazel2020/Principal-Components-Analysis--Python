# Principal Components Analysis using MNIST Datset in Python

<img src="images/PCA.jpg" width="1000"/>

## 1. Objectives

The objective of this project is to demonstrate the application of the Principal Components Analysis (PCA) algorithms to represent the MNIST training data subset.

## 2.  Principal Components Analysis (PCA)

Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

Reducing the number of variables of a data set naturally comes at the expense of accuracy, but the trick in dimensionality reduction is to trade a little accuracy for simplicity. Because smaller data sets are easier to explore and visualize and make analyzing data much easier and faster for machine learning algorithms without extraneous variables to process.

Basically, the idea of PCA is simple: reduce the number of variables of a data set, while preserving as much information as possible. 

In this project, we shall illustrate how to compute and visualize the PCA components generated from the MINIST dataset.

## 3. Data

We shall illustrate the PCA representation of the  MNIST database of handwritten digits, available from this page, which has a training set of 42,000 examples, and a test set of 18,000 examples. We shall illustrate sample images from this data sets in the next section.

## 4. Development

Project: PCA Representation of the MNIST Dataset:
The objective of this project is to demonstrate the application of the PCA algorithm to visualize the MNIST dataset in two different ways:

* First, we apply the PCA using the Sckit-learn API
* Secondly, we implement the PCA algorithm from scratch.

* Author: Mohsen Ghazel (mghazel)
* Date: May 5th, 2021

### 4.1. Part 1: Python imports and global variables:

#### 4.1.1. Python imports:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Python imports and environment setup</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># opencv</span>
<span style="color:#200080; font-weight:bold; ">import</span> cv2
<span style="color:#595979; "># numpy</span>
<span style="color:#200080; font-weight:bold; ">import</span> numpy <span style="color:#200080; font-weight:bold; ">as</span> np
<span style="color:#595979; "># matplotlib</span>
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>pyplot <span style="color:#200080; font-weight:bold; ">as</span> plt

<span style="color:#595979; "># import additional functionalities</span>
<span style="color:#200080; font-weight:bold; ">from</span> __future__ <span style="color:#200080; font-weight:bold; ">import</span> print_function<span style="color:#308080; ">,</span> division
<span style="color:#200080; font-weight:bold; ">from</span> builtins <span style="color:#200080; font-weight:bold; ">import</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">,</span> <span style="color:#400000; ">input</span>

<span style="color:#595979; "># import the PCA algorithm from sklearn</span>
<span style="color:#200080; font-weight:bold; ">from</span> sklearn<span style="color:#308080; ">.</span>decomposition <span style="color:#200080; font-weight:bold; ">import</span> PCA

<span style="color:#595979; "># import shuffle  from sklearn</span>
<span style="color:#200080; font-weight:bold; ">from</span> sklearn<span style="color:#308080; ">.</span>utils <span style="color:#200080; font-weight:bold; ">import</span> shuffle

<span style="color:#595979; "># import pandas</span>
<span style="color:#200080; font-weight:bold; ">import</span> pandas <span style="color:#200080; font-weight:bold; ">as</span> pd

<span style="color:#595979; "># random number generators values</span>
<span style="color:#595979; "># seed for reproducing the random number generation</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> seed
<span style="color:#595979; "># random integers: I(0,M)</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> randint
<span style="color:#595979; "># random standard unform: U(0,1)</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> random
<span style="color:#595979; "># time</span>
<span style="color:#200080; font-weight:bold; ">import</span> datetime
<span style="color:#595979; "># I/O</span>
<span style="color:#200080; font-weight:bold; ">import</span> os
<span style="color:#595979; "># sys</span>
<span style="color:#200080; font-weight:bold; ">import</span> sys

<span style="color:#595979; "># display figure within the notebook</span>
<span style="color:#44aadd; ">%</span>matplotlib inline

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Test imports and display package versions</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Testing the OpenCV version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"OpenCV : "</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>
<span style="color:#595979; "># Testing the numpy version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Numpy : "</span><span style="color:#308080; ">,</span>np<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>

OpenCV <span style="color:#308080; ">:</span>  <span style="color:#008000; ">3.4</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">8</span>
Numpy <span style="color:#308080; ">:</span>  <span style="color:#008000; ">1.19</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">2</span>
</pre>

#### 4.1.2. Global variables:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># set the random_state seed = 100 for reproducibilty</span>
random_state_seed <span style="color:#308080; ">=</span> <span style="color:#008c00; ">100</span>

<span style="color:#595979; "># the number of visualized images</span>
num_visualized_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span>
</pre>


### 4.2. Part 2: Read the input data:

* We use the MINIST dataset, which was downloaded from the following link:
  * Kaggle: Digit REcognizer: https://www.kaggle.com/c/digit-recognizer/data
  * The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.
  * Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.
  * The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.
  * Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

#### 4.2.1. Load and normalize the training data sets:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># read the training data set</span>
train <span style="color:#308080; ">=</span> pd<span style="color:#308080; ">.</span>read_csv<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'../large_files/train.csv'</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>values<span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>float32<span style="color:#308080; ">)</span>
<span style="color:#595979; "># shuffle the training data set</span>
train <span style="color:#308080; ">=</span> shuffle<span style="color:#308080; ">(</span>train<span style="color:#308080; ">)</span>
<span style="color:#595979; "># normalize the training data to [0,1]:</span>
x_train <span style="color:#308080; ">=</span> train<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span> <span style="color:#44aadd; ">/</span> <span style="color:#008c00; ">255</span>
<span style="color:#595979; "># format the class type to integer</span>
y_train <span style="color:#308080; ">=</span> train<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>int32<span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Display a summary of the training data:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># the number of training images</span>
num_train_images <span style="color:#308080; ">=</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Training data:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"x_train.shape: "</span><span style="color:#308080; ">,</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"y_train.shape: "</span><span style="color:#308080; ">,</span> y_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Number of training images: "</span><span style="color:#308080; ">,</span> num_train_images<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Image size: "</span><span style="color:#308080; ">,</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Classes/labels:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'The target labels: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>unique<span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Training data<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">42000</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">784</span><span style="color:#308080; ">)</span>
y_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">42000</span><span style="color:#308080; ">,</span><span style="color:#308080; ">)</span>
Number of training images<span style="color:#308080; ">:</span>  <span style="color:#008c00; ">42000</span>
Image size<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">784</span><span style="color:#308080; ">,</span><span style="color:#308080; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Classes<span style="color:#44aadd; ">/</span>labels<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The target labels<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span> <span style="color:#008c00; ">1</span> <span style="color:#008c00; ">2</span> <span style="color:#008c00; ">3</span> <span style="color:#008c00; ">4</span> <span style="color:#008c00; ">5</span> <span style="color:#008c00; ">6</span> <span style="color:#008c00; ">7</span> <span style="color:#008c00; ">8</span> <span style="color:#008c00; ">9</span><span style="color:#308080; ">]</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

#### 4.2.2. Visualize some of the training images and their associated targets:

##### 4.2.2.1. First implement a visualization functionality to visualize the number of randomly selected images:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">"""</span>
<span style="color:#595979; "># A utility function to visualize multiple images:</span>
<span style="color:#595979; ">"""</span>
<span style="color:#200080; font-weight:bold; ">def</span> visualize_images_and_labels<span style="color:#308080; ">(</span>num_visualized_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span><span style="color:#308080; ">,</span> dataset_flag <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
  <span style="color:#595979; ">"""To visualize images.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keyword arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- num_visualized_images -- the number of visualized images (deafult 25)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- dataset_flag -- 1: training dataset, 2: test dataset</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- None</span>
<span style="color:#595979; ">&nbsp;&nbsp;"""</span>
  <span style="color:#595979; ">#--------------------------------------------</span>
  <span style="color:#595979; "># the suplot grid shape:</span>
  <span style="color:#595979; ">#--------------------------------------------</span>
  num_rows <span style="color:#308080; ">=</span> <span style="color:#008c00; ">5</span>
  <span style="color:#595979; "># the number of columns</span>
  num_cols <span style="color:#308080; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
  <span style="color:#595979; "># setup the subplots axes</span>
  fig<span style="color:#308080; ">,</span> axes <span style="color:#308080; ">=</span> plt<span style="color:#308080; ">.</span>subplots<span style="color:#308080; ">(</span>nrows<span style="color:#308080; ">=</span>num_rows<span style="color:#308080; ">,</span> ncols<span style="color:#308080; ">=</span>num_cols<span style="color:#308080; ">,</span> figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
  <span style="color:#595979; "># set a seed random number generator for reproducible results</span>
  seed<span style="color:#308080; ">(</span>random_state_seed<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># iterate over the sub-plots</span>
  <span style="color:#200080; font-weight:bold; ">for</span> row <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_rows<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
      <span style="color:#200080; font-weight:bold; ">for</span> col <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_cols<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># get the next figure axis</span>
        ax <span style="color:#308080; ">=</span> axes<span style="color:#308080; ">[</span>row<span style="color:#308080; ">,</span> col<span style="color:#308080; ">]</span><span style="color:#308080; ">;</span>
        <span style="color:#595979; "># turn-off subplot axis</span>
        ax<span style="color:#308080; ">.</span>set_axis_off<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># if the dataset_flag = 1: Training data set</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span> dataset_flag <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">:</span> 
          <span style="color:#595979; "># generate a random image counter</span>
          counter <span style="color:#308080; ">=</span> randint<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>num_train_images<span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the training image</span>
          image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the target associated with the image</span>
          label <span style="color:#308080; ">=</span> y_train<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">]</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># dataset_flag = 2: Test data set</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span> 
          <span style="color:#595979; "># generate a random image counter</span>
          counter <span style="color:#308080; ">=</span> randint<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>num_test_images<span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the test image</span>
          image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>x_test<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the target associated with the image</span>
          label <span style="color:#308080; ">=</span> y_test<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">]</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># display the image</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        ax<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>image<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span>plt<span style="color:#308080; ">.</span>cm<span style="color:#308080; ">.</span>gray_r<span style="color:#308080; ">,</span> interpolation<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'nearest'</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; "># set the title showing the image label</span>
        ax<span style="color:#308080; ">.</span>set_title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'y ='</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>label<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> size <span style="color:#308080; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#308080; ">)</span>
</pre>

##### 4.2.2.2. Call the function to visualize the randomly selected training images:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># the number of selected training images</span>
num_visualized_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span>
<span style="color:#595979; "># call the function to visualize the training images</span>
visualize_images_and_labels<span style="color:#308080; ">(</span>num_visualized_images<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
</pre>

<img src="images/sample-images.png" width="1000"/>

#### 4.2.3. Examine the number of images for each class of the training and testing subsets:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># create a histogram of the number of images in each class/digit:</span>
<span style="color:#200080; font-weight:bold; ">def</span> plot_bar<span style="color:#308080; ">(</span>y<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">,</span> relative<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    width <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.35</span>
    <span style="color:#200080; font-weight:bold; ">if</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">:</span>
        n <span style="color:#308080; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">0.5</span>
    <span style="color:#200080; font-weight:bold; ">elif</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'right'</span><span style="color:#308080; ">:</span>
        n <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.5</span>
     
    <span style="color:#595979; "># calculate counts per type and sort, to ensure their order</span>
    unique<span style="color:#308080; ">,</span> counts <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>unique<span style="color:#308080; ">(</span>y<span style="color:#308080; ">,</span> return_counts<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span>
    sorted_index <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>argsort<span style="color:#308080; ">(</span>unique<span style="color:#308080; ">)</span>
    unique <span style="color:#308080; ">=</span> unique<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span>
     
    <span style="color:#200080; font-weight:bold; ">if</span> relative<span style="color:#308080; ">:</span>
        <span style="color:#595979; "># plot as a percentage</span>
        counts <span style="color:#308080; ">=</span> <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">*</span>counts<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span><span style="color:#44aadd; ">/</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y<span style="color:#308080; ">)</span>
        ylabel_text <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'% count'</span>
    <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># plot counts</span>
        counts <span style="color:#308080; ">=</span> counts<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span>
        ylabel_text <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'count'</span>
         
    xtemp <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>arange<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>unique<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>bar<span style="color:#308080; ">(</span>xtemp <span style="color:#44aadd; ">+</span> n<span style="color:#44aadd; ">*</span>width<span style="color:#308080; ">,</span> counts<span style="color:#308080; ">,</span> align<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'center'</span><span style="color:#308080; ">,</span> alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">.7</span><span style="color:#308080; ">,</span> width<span style="color:#308080; ">=</span>width<span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span>xtemp<span style="color:#308080; ">,</span> unique<span style="color:#308080; ">,</span> rotation<span style="color:#308080; ">=</span><span style="color:#008c00; ">45</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'digit'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span>ylabel_text<span style="color:#308080; ">)</span>
 
plt<span style="color:#308080; ">.</span>suptitle<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Frequency of images per digit'</span><span style="color:#308080; ">)</span>
plot_bar<span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>legend<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>
    <span style="color:#1060b6; ">'train ({0} images)'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
</pre>

<img src="images/histogram-train-images.png" width="1000"/>


### 4.3. Part 3: Use Scikit-learn-API: Generate the PCA representation of the training data:

#### 4.3.1. Visualize the 2D PCA representation of the training data


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># Step 3.1.1: Compute the PCA components</span>
<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># Instantiante the PCA() algorithm</span>
pca <span style="color:#308080; ">=</span> PCA<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># apply the PCA on the training data</span>
reduced <span style="color:#308080; ">=</span> pca<span style="color:#308080; ">.</span>fit_transform<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">)</span>
<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># Step 3.1.2: plot the first 2 PCA </span>
<span style="color:#595979; ">#             components</span>
<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
<span style="color:#595979; "># plot the first 2 PCA components</span>
plt<span style="color:#308080; ">.</span>scatter<span style="color:#308080; ">(</span>reduced<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> reduced<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> s<span style="color:#308080; ">=</span><span style="color:#008c00; ">100</span><span style="color:#308080; ">,</span> c<span style="color:#308080; ">=</span>y_train<span style="color:#308080; ">,</span> alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">0.5</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># set the x-label</span>
plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'V1: First PCA Component'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># set the y-label</span>
plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'V2: Second PCA Component'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># set the title</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'2D PCA representation of the MNIST dataset'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># show the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<img src="images/Scikit-learn-PCA-2D.png" width="1000"/>

#### 4.3.2. Display the explained variance variations:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># Step 3.2.1: plot the explained variance for </span>
<span style="color:#595979; ">#             each PCA component:</span>
<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">5</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
<span style="color:#595979; "># the first subplot</span>
plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">121</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># plot the the explained variance for each PCA component</span>
plt<span style="color:#308080; ">.</span>plot<span style="color:#308080; ">(</span>pca<span style="color:#308080; ">.</span>explained_variance_ratio_<span style="color:#308080; ">)</span>
<span style="color:#595979; "># the x-label</span>
plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'PCA Dimensions'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># the y-label</span>
plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Explained Variance (Percent)'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># the figure title</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Explained variance of each PCA component'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># Step 3.2.2: plot the cumulative explained </span>
<span style="color:#595979; ">#            variance of the PCA components</span>
<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># cumulative variance</span>
<span style="color:#595979; "># choose k = number of dimensions that gives us 95-99% variance</span>
cumulative <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#308080; ">]</span>
last <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
<span style="color:#595979; "># sum-up the expplained variances for each component</span>
<span style="color:#595979; "># to comute the cumulative explained variance</span>
<span style="color:#200080; font-weight:bold; ">for</span> v <span style="color:#200080; font-weight:bold; ">in</span> pca<span style="color:#308080; ">.</span>explained_variance_ratio_<span style="color:#308080; ">:</span>
    <span style="color:#595979; "># sum the explained variance</span>
    cumulative<span style="color:#308080; ">.</span>append<span style="color:#308080; ">(</span>last <span style="color:#44aadd; ">+</span> v<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># store the last comulative variance</span>
    last <span style="color:#308080; ">=</span> cumulative<span style="color:#308080; ">[</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span>
<span style="color:#595979; "># the second subplot</span>
plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">122</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the cumulative variance</span>
plt<span style="color:#308080; ">.</span>plot<span style="color:#308080; ">(</span>cumulative<span style="color:#308080; ">)</span>
<span style="color:#595979; "># the x-label</span>
plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'PCA Dimensions'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># the y-label</span>
plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Cumulative explained variance (percent)'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># the figure title</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Cumulative explained variance vs the PCA dimensions'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># show the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<img src="images/Scikit-learn-PCA-Explianed-Variance.png" width="1000"/>


### 4.4. Part 4: From Scratch: Generate the PCA representation of the training data:

#### 4.1.1. Visualize the 2D PCA representation of the training data:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># Step 4.1.1: Compute the PCA components</span>
<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># decompose covariance</span>
covX <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>cov<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">.</span>T<span style="color:#308080; ">)</span>
<span style="color:#595979; "># compute the eigen-values and vectors of covX</span>
lambdas<span style="color:#308080; ">,</span> Q <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>linalg<span style="color:#308080; ">.</span>eigh<span style="color:#308080; ">(</span>covX<span style="color:#308080; ">)</span>

<span style="color:#595979; "># lambdas are sorted from smallest --&gt; largest</span>
<span style="color:#595979; "># some may be slightly negative due to precision</span>
idx <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>argsort<span style="color:#308080; ">(</span><span style="color:#44aadd; ">-</span>lambdas<span style="color:#308080; ">)</span>
<span style="color:#595979; "># sort in proper order</span>
lambdas <span style="color:#308080; ">=</span> lambdas<span style="color:#308080; ">[</span>idx<span style="color:#308080; ">]</span> 
<span style="color:#595979; "># get rid of negatives</span>
lambdas <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>maximum<span style="color:#308080; ">(</span>lambdas<span style="color:#308080; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span> 
Q <span style="color:#308080; ">=</span> Q<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span>idx<span style="color:#308080; ">]</span>
<span style="color:#595979; "># compute the transformed data</span>
Z <span style="color:#308080; ">=</span> x_train<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>Q<span style="color:#308080; ">)</span>
<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># Step 4.1.2: plot the first 2 PCA components</span>
<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
<span style="color:#595979; "># plot the first 2 PCA components</span>
plt<span style="color:#308080; ">.</span>scatter<span style="color:#308080; ">(</span>Z<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> Z<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> s<span style="color:#308080; ">=</span><span style="color:#008c00; ">100</span><span style="color:#308080; ">,</span> c<span style="color:#308080; ">=</span>y_train<span style="color:#308080; ">,</span> alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">0.3</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># set the x-label</span>
plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'V1: First PCA Component'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># set the y-label</span>
plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'V2: Second PCA Component'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># set the title</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'2D PCA representation of the MNIST dataset'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
<span style="color:#595979; "># shoe the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<img src="images/Implemented-PCA-2D.png" width="1000"/>

#### 4.4.2. Display the explained variance variations:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># Step 4.2.1: plot the explained variance for </span>
<span style="color:#595979; ">#             each PCA component:</span>
<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># create a figure and set its axis</span>
fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">16</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># create the figure </span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
<span style="color:#595979; "># the first subplot</span>
plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">211</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># plot the the explained variance for each PCA component</span>
plt<span style="color:#308080; ">.</span>plot<span style="color:#308080; ">(</span>lambdas<span style="color:#308080; ">)</span>
<span style="color:#595979; "># the x-label</span>
plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'PCA Dimensions'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># the y-label</span>
plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Explained variance (percent)'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># the figure title</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Explained variance of each PCA component'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># Step 4.2.2: plot the cumulative explained </span>
<span style="color:#595979; ">#            variance of the PCA components</span>
<span style="color:#595979; ">#----------------------------------------</span>
<span style="color:#595979; "># the second subplot</span>
plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">212</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display the cumulative variance</span>
plt<span style="color:#308080; ">.</span>plot<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>cumsum<span style="color:#308080; ">(</span>lambdas<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># the x-label</span>
plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'PCA Dimensions'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># the y-label</span>
plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Cumulative explained variance (Percent)'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># the figure title</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Cumulative explained variance vs. the PCA dimensions'</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># show the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<img src="images/Implemented-PCA-Explianed-Variance.png" width="1000"/>

### 4.5. Step 5: Display a successful execution message:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># display a final message</span>
<span style="color:#595979; "># current time</span>
now <span style="color:#308080; ">=</span> datetime<span style="color:#308080; ">.</span>datetime<span style="color:#308080; ">.</span>now<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>now<span style="color:#308080; ">.</span>strftime<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#1060b6; ">"</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

Program executed successfully on<span style="color:#308080; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">05</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">0</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">9</span> <span style="color:#008c00; ">15</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">38</span><span style="color:#308080; ">:</span><span style="color:#008000; ">04.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>Goodbye!
</pre>

## 5. Analysis

* In view of the presented results, we make the following observations:

  * The PCA representations of the MNIST training data sets generated using the following methods are comparable:
  * The PCA computed using the Scikit-learn API.
  * The PCA computed from scratch using the Eigen decomposition of the covariance matrix of the training data 784 features vectors
  * The slight differences can be explained as follows:
  * For PCA implementation, the orientations of the PCA 2D representation is flipped in the first PCA direction due to the fact that the eigenvector is not unique, as both V1 and -V1 are acceptable values
  * For both methods, the explained variance variance by each component decays rapidly, indicating that the first few PCA components capture most of the data variability
  * Similarly, the cumulative explained variance confirming that most of  most of the data variability lies in the first few PCA components
  * The differences in the vertical axes scaling of the variance explanation figures generated by the 2 methods are due to:
    * The PCA computed using the Scikit-learn API uses percentage scaling
    * The PCA computed from scratch using the Eigen decomposition of the covariance matrix of the training data 784 features vectors, as use frequency scaling.

## 6. Future Work

* We plan to explore the following related issues:

  * To truncate the PCA decomposition space of the MNIST dataset to K dimension, K < 784, by removing insignificant dimension, after capturing 95% of the variability
  * To classify the MNIST dataset using the K-dimensional PCA decomposition space 
  * To assess the computational complexity vs accuracy trade-off of classifying the MNIST dataset using:
  * The original training dataset
  * The reduced PCA representation.

## 7. References

1. Kaggle. Digit Recognizer: Learn computer vision fundamentals with the famous MNIST data. https://www.kaggle.com/c/digit-recognizer/data 
2. Yann LeCun et. al. THE MNIST DATABASE of handwritten digits. http://yann.lecun.com/exdb/mnist/ buitin. (May 9th, 2021). A STEP-BY-STEP EXPLANATION OF PRINCIPAL COMPONENT ANALYSIS (PCA). https://builtin.com/data-science/step-step-explanation-principal-component-analysis 
3. PennState Eberly College of Science. Principal Components Analysis (PCA). https://online.stat.psu.edu/stat505/book/export/html/670 
4. Aditya Sharma. Principal Component Analysis (PCA) in Python. https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python 
5. Python Data Science Handbook. In Depth: Principal Component Analysis. https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html 
6. Towards data science. PCA using Python (scikit-learn). https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60 
7. Usman Malik. Implementing PCA in Python with Scikit-Learn. https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/ 
8. Towards AI Team. Principal Component Analysis (PCA) with Python Examples  - Tutorial. https://pub.towardsai.net/principal-component-analysis-pca-with-python-examples-tutorial-67a917bae9aa
 
 
