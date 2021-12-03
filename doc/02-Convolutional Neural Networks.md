## Convolutional Neural Networks

> So what changes? ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. ==These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.==

卷积神经网络假设输入的都是图片，这样可以根据图片有针对性的做推流。

> *Regular Neural Nets.* As we saw in the previous chapter, Neural Networks receive an input (a single vector), and transform it through a series of *hidden layers*. Each hidden layer is made up of a set of neurons, where each neuron is fully connected to all neurons in the previous layer, and where neurons in a single layer function completely independently and do not share any connections. The last fully-connected layer is called the “output layer” and in classification settings it represents the class scores.

回顾一下普通的神经网络

> In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: **width, height, depth**. 

![image-20211125115037614](/Users/songtingyu/Library/Application Support/typora-user-images/image-20211125115037614.png)

这张图真不chuo。

> As we described above, a simple ConvNet is a sequence of layers, and every layer of a ConvNet transforms one volume of activations to another through a differentiable function. We use three main types of layers to build ConvNet architectures: ==**Convolutional Layer**, **Pooling Layer**, and **Fully-Connected Layer** (exactly as seen in regular Neural Networks). We will stack these layers to form a full ConvNet **architecture**.==

> - INPUT [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
> - CONV layer will compute the output of neurons that are connected to local regions in the input, ==each computing a dot product between their weights and a small region they are connected to in the input volume.== This may result in volume such as [32x32x12] if we decided to use 12 filters.
> - RELU layer will apply an elementwise activation function, such as the \(max(0,x)\) thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]).
> - POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
> - FC (i.e. fully-connected) layer will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10. As with ordinary Neural Networks and as the name implies, each neuron in this layer will be connected to all the numbers in the previous volume.

### 2 Convolutional layer

> Let’s first discuss what the CONV layer computes without brain/neuron analogies. The CONV layer’s parameters consist of a set of learnable filters. Every filter is small spatially (along width and height), but extends through the full depth of the input volume. For example, ==a typical filter on a first layer of a ConvNet might have size 5x5x3 (i.e. 5 pixels width and height, and 3 because images have depth 3, the color channels).== During the forward pass, we slide (more precisely, convolve) each filter across the width and height of the input volume and compute dot products between the entries of the filter and the input at any position. As we slide the filter over the width and height of the input volume we will produce a 2-dimensional activation map that gives the responses of that filter at every spatial position. Intuitively, the network will learn filters that activate when they see some type of visual feature such as an edge of some orientation or a blotch of some color on the first layer, or eventually entire honeycomb or wheel-like patterns on higher layers of the network. Now, we will have an entire set of filters in each CONV layer (e.g. 12 filters), and each of them will produce a separate 2-dimensional activation map. We will stack these activation maps along the depth dimension and produce the output volume.

1个filter可能是5 * 5 * 3.

> **Local Connectivity.** When dealing with high-dimensional inputs such as images, as we saw above it is impractical to connect neurons to all neurons in the previous volume. **Instead, we will connect each neuron to only a local region of the input volume.**  The spatial extent of this connectivity is a hyperparameter called the **receptive field** of the neuron (equivalently this is the filter size). The extent of the connectivity along the depth axis is always equal to the depth of the input volume. It is important to emphasize again this asymmetry in how we treat the spatial dimensions (width and height) and the depth dimension: ==The connections are local in 2D space (along width and height), but always full along the entire depth of the input volume.==

卷积神经网络的神经元和普通全连接的神经元有什么不同。卷积神经网络里神经元和filter有什么差别。为什么filter可以提取图片特征

[filter大小改变对图片的影响](https://setosa.io/ev/image-kernels/)

![image-20211125135909155](/Users/songtingyu/Library/Application Support/typora-user-images/image-20211125135909155.png)

> **Left:** An example input volume in red (e.g. a 32x32x3 CIFAR-10 image), and an example volume of neurons in the first Convolutional layer. Each neuron in the convolutional layer is connected only to a local region in the input volume spatially, but to the full depth (i.e. all color channels). ==Note, there are multiple neurons (5 in this example) along the depth, all looking at the same region in the input: the lines that connect this column of 5 neurons do not represent the weights== (i.e. these 5 neurons do not share the same weights, but they are associated with 5 different filters), they just indicate that these neurons are connected to or looking at the same receptive field or region of the input volume, i.e. they share the same receptive field but not the same weights. **Right:** The neurons from the Neural Network chapter remain unchanged: They still compute a dot product of their weights with the input followed by a non-linearity, but their connectivity is now restricted to be local spatially.

回顾一下，比重是用来干什么的？

> **Implementation as Matrix Multiplication**. Note that the convolution operation essentially performs dot products between the filters and local regions of the input. A common implementation pattern of the CONV layer is to take advantage of this fact and formulate the forward pass of a convolutional layer as one big matrix multiply as follows:
>
> 1. The local regions in the input image are stretched out into columns in an operation commonly called **im2col**. For example, if the input is [227x227x3] and it is to be convolved with 11x11x3 filters at stride 4, then we would take [11x11x3] blocks of pixels in the input and stretch each block into a column vector of size 11* 11* 3 = 363. Iterating this process in the input at stride of 4 gives (227-11)/4+1 = 55 locations along both width and height, leading to an output matrix `X_col` of *im2col* of size [363 x 3025], where every column is a stretched out receptive field and there are 55*55 = 3025 of them in total. Note that since the receptive fields overlap, every number in the input volume may be duplicated in multiple distinct columns.
> 2. The weights of the CONV layer are similarly stretched out into rows. For example, if there are 96 filters of size [11x11x3] this would give a matrix `W_row` of size [96 x 363].
> 3. The result of a convolution is now equivalent to performing one large matrix multiply `np.dot(W_row, X_col)`, which evaluates the dot product between every filter and every receptive field location. In our example, the output of this operation would be [96 x 3025], giving the output of the dot product of each filter at each location.
> 4. The result must finally be reshaped back to its proper output dimension [55x55x96].

先通过im2col将图像变成列，这个im2col的代码如下

```python
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1
    i0 = np.repeat(np.arange(field_height), field_width) # i0.shape = [field_height*field_width,1]
    # tile(array,brray) 把array沿brray方向展开多少次 比如 
    # a = np.array([0,1,2])
    # np.tile(a,(2,2)) = array([[0,1,2,0,1,2],[0,1,2,0,1,2]])
    i0 = np.tile(i0, C)  # i0.shape = [field_width*field_height*C,1]
    i1 = stride * np.repeat(np.arange(out_height), out_width) #i1.shape = [out_height*out_width,1]
    j0 = np.tile(np.arange(field_width), field_height * C) # j0.shape = [field_width*field_height*C,1]
    j1 = stride * np.tile(np.arange(out_width), out_height) # j1.shape = [out_width*out_height,1]
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    # 先把pad部分加入x
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
```

看到网上博客讲的一点很好，也引起我的思考，为什么需要卷积层。

> 假如我们有 M 个输入和 N个输出，那么在训练过程中，我们需要 M×N 个参数去刻画输入和输出的关系 。当 M 和 N都很大，并且再加几层的卷积网络，这个参数量将会大的离谱。
>
> 卷积运算主要通过**三个重要思想**来改进上述面来的问题：稀疏连接、参数共享、平移不变性。

而实现如上方法的原因是因为kernel核函数

1. 卷积层减少了连接数量
2. 卷积层学习一组参数集合，而不是针对每一个位置的参数进行学习。比如你在全连接层里，可能每个点都有一个权重矩阵。
3. 平移不变性，如果一个函数的输入做了改变，那么输出也会跟着改变。举个很好的例子，比如要识别一直猫，那么无论这只猫在图片的什么位置，我们都应该识别出来。

> **Spatial arrangement**. We have explained the connectivity of each neuron in the Conv Layer to the input volume, but we haven’t yet discussed how many neurons there are in the output volume or how they are arranged. Three hyperparameters control the size of the output volume: the **depth, stride** and **zero-padding**. We discuss these next:
>
> 1. First, the **depth** of the output volume is a hyperparameter: it corresponds to the number of filters we would like to use, each learning to look for something different in the input. For example, if the first Convolutional Layer takes as input the raw image, then different neurons along the depth dimension may activate in presence of various oriented edges, or blobs of color. We will refer to a set of neurons that are all looking at the same region of the input as a **depth column** (some people also prefer the term *fibre*).
> 2. Second, we must specify the **stride** with which we slide the filter. When the stride is 1 then we move the filters one pixel at a time. When the stride is 2 (or uncommonly 3 or more, though this is rare in practice) then the filters jump 2 pixels at a time as we slide them around. This will produce smaller output volumes spatially.
> 3. As we will soon see, sometimes it will be convenient to pad the input volume with zeros around the border. The size of this **zero-padding** is a hyperparameter. The nice feature of zero padding is that it will allow us to control the spatial size of the output volumes (most commonly as we’ll see soon we will use it to exactly preserve the spatial size of the input volume so the input and output width and height are the same).

![image-20211128092030842](/Users/songtingyu/Library/Application Support/typora-user-images/image-20211128092030842.png)

> *Use of zero-padding*. In the example above on left, note that the input dimension was 5 and the output dimension was equal: also 5. This worked out so because our receptive fields were 3 and we used zero padding of 1. If there was no zero-padding used, then the output volume would have had spatial dimension of only 3, because that is how many neurons would have “fit” across the original input. In general, setting zero padding to be *P*=(*F*−1)/2 when the stride is *S*=1 ensures that the input volume and output volume will have the same size spatially. It is very common to use zero-padding in this way and we will discuss the full reasons when we talk more about ConvNet architectures.

这里说的zero-padding并不是不padding，而是0填充。0填充可以使输入和输出具有相同的维度。

> *Constraints on strides*. Note again that the spatial arrangement hyperparameters have mutual constraints. For example, when the input has size *W*=10, no zero-padding is used *P*=0, and the filter size is *F*=3, then it would be impossible to use stride *S*=2, since (*W*−*F*+2*P*)/*S*+1=(10−3+0)/2+1=4.5, i.e. not an integer, indicating that the neurons don’t “fit” neatly and symmetrically across the input. Therefore, this setting of the hyperparameters is considered to be invalid, and a ConvNet library could throw an exception or zero pad the rest to make it fit, or crop the input to make it fit, or something. As we will see in the ConvNet architectures section, sizing the ConvNets appropriately so that all the dimensions “work out” can be a real headache, which the use of zero-padding and some design guidelines will significantly alleviate.

这里讲述的是strides是有限制的。

> *Real-world example*. The [Krizhevsky et al.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) architecture that won the ImageNet challenge in 2012 accepted images of size [227x227x3]. On the first Convolutional Layer, it used neurons with receptive field size *F*=11, stride *S*=4 and no zero padding *P*=0. Since (227 - 11)/4 + 1 = 55, and since the Conv layer had ==a depth of *K*=96==, the Conv layer output volume had size [55x55x96]. Each of the 55 * 55 * 96 neurons in this volume was connected to a region of size [11x11x3] in the input volume. Moreover, all 96 neurons in each depth column are connected to the same [11x11x3] region of the input, but of course with different weights. As a fun aside, if you read the actual paper it claims that the input images were 224x224, which is surely incorrect because (224 - 11)/4 + 1 is quite clearly not an integer. This has confused many people in the history of ConvNets and little is known about what happened. My own best guess is that Alex used zero-padding of 3 extra pixels that he does not mention in the paper.

这里讲述的是一个，现实中的具体例子。这个图像大小是227 * 227 * 3，然后这个人用了F=11，步长Stride为4，没有使用zero-padding。这个卷积层的深度为96，输出的结果为[55 * 55 * 96]

> **Parameter Sharing.** Parameter sharing scheme is used in Convolutional Layers to control the number of parameters. Using the real-world example above, we see that there are 55 * 55 * 96 = 290,400 neurons in the first Conv Layer, and each has 11 *  11* 3 = 363 weights and 1 bias. Together, this adds up to 290400 * 364 = 105,705,600 parameters on the first layer of the ConvNet alone. Clearly, this number is very high.
>
> It turns out that we can dramatically reduce the number of parameters by making one reasonable assumption: That if one feature is useful to compute at some spatial position (x,y), then it should also be useful to compute at a different position (x2,y2). In other words, denoting a single 2-dimensional slice of depth as a **depth slice** (e.g. a volume of size [55x55x96] has 96 depth slices, each of size [55x55]), we are going to constrain the neurons in each depth slice to use the same weights and bias. With this parameter sharing scheme, the first Conv Layer in our example would now have only 96 unique set of weights (one for each depth slice), for a total of 96*11*11*3 = 34,848 unique weights, or 34,944 parameters (+96 biases). Alternatively, all 55*55 neurons in each depth slice will now be using the same parameters. In practice during backpropagation, every neuron in the volume will compute the gradient for its weights, but these gradients will be added up across each depth slice and only update a single set of weights per slice.
>
> Notice that if all neurons in a single depth slice are using the same weight vector, then the forward pass of the CONV layer can in each depth slice be computed as a **convolution** of the neuron’s weights with the input volume (Hence the name: Convolutional Layer). This is why it is common to refer to the sets of weights as a **filter** (or a **kernel**), that is convolved with the input.

**back propagation**



### Pooling layer

>  It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. The Pooling Layer operates independently on every depth slice of the input and resizes it spatially, using the MAX operation. The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. Every MAX operation would in this case be taking a max over 4 numbers (little 2x2 region in some depth slice). The depth dimension remains unchanged. More generally, the pooling layer:
>
> * Accepts a volume of size *W*1×*H*1×*D*1
> * Requires two hyperparameters:
>   * their spatial extent *F*,
>   * the stride *S*,
> * Produces a volume of size[*W*2×*H*2×*D*2] where:
>   - *W*2=(*W*1−*F*)/*S*+1
>   - *H*2=(*H*1−*F*)/*S*+1
>   - *D*2=*D*1
> * Introduces zero parameters since it computes a fixed function of the input
> * For Pooling layers, it is not common to pad the input using zero-padding.
>
> It is worth noting that there are only two commonly seen variations of the max pooling layer found in practice: A pooling layer with *F*=3,*S*=2 (also called overlapping pooling), and more commonly *F*=2,*S*=2. Pooling sizes with larger receptive fields are too destructive.

上面讲了通常而言pooling要怎么做的问题，通常pooling的filter的大小是2*2，stride也同样为2，有max pooling，average pooling等等，

> **Backpropagation**. Recall from the backpropagation chapter that the backward pass for a max(x, y) operation has a simple interpretation as only routing the gradient to the input that had the highest value in the forward pass. Hence, during the forward pass of a pooling layer it is common to keep track of the index of the max activation (sometimes also called *the switches*) so that gradient routing is efficient during backpropagation.

Backpropagation，因为通常为max pooling，所以在前向计算时，需要对最大值做一个追踪。

> **Getting rid of pooling**. Many people dislike the pooling operation and think that we can get away without it. For example, [Striving for Simplicity: The All Convolutional Net](http://arxiv.org/abs/1412.6806) proposes to discard the pooling layer in favor of architecture that only consists of repeated CONV layers. To reduce the size of the representation they suggest using larger stride in CONV layer once in a while. Discarding pooling layers has also been found to be important in training ==good generative models==, such as variational autoencoders (VAEs) or generative adversarial networks (GANs). It seems likely that future architectures will feature very few to no pooling layers.

很多人不喜欢pooling操作，并且不适用pooling layer被证明可以用来训练一些good generative models.



### Normalization Layer

>  Many types of normalization layers have been proposed for use in ConvNet architectures, sometimes with the intentions of implementing inhibition schemes observed in the biological brain. However, these layers have since fallen out of favor because in practice their contribution has been shown to be minimal, if any. For various types of normalizations, see the discussion in Alex Krizhevsky’s [cuda-convnet library API](http://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_(same_map)).

被证明作用很小。

### Converting FC layers to Conv Layer

> It is worth noting that the only difference between FC and CONV layers is that the neurons in the CONV layer are connected only to a local region in the input, and that many of the neurons in a CONV volume share parameters. However, the neurons in both layers still compute dot products, ==so their functional form is identical==. Therefore, it turns out that it’s possible to convert between FC and CONV layers:
>
> - For any CONV layer there is an FC layer that implements the same forward function. The weight matrix would be a large matrix that is mostly zero except for at certain blocks (due to local connectivity) where the weights in many of the blocks are equal (due to parameter sharing).
> - Conversely, any FC layer can be converted to a CONV layer. For example, an FC layer with \(K = 4096\) that is looking at some input volume of size \(7 * 7 * 512\) can be equivalently expressed as a CONV layer with \(F = 7, P = 0, S = 1, K = 4096\). In other words, we are setting the filter size to be exactly the size of the input volume, and hence the output will simply be \(1 * 1 * 4096\) since only a single depth column “fits” across the input volume, giving identical result as the initial FC layer.

意思是两个层的计算函数实际上是一样的，唯一不同的是FC是全连接，Conv并没有全连接。下面举的例子是，让卷积核的大小就是图像的大小，这样做滤波的结果和全连接层是一样的。计算的过程是这样的，我们现在有volume的大小是 7 * 7 * 512， 如果用一个(F=7 P=0 S=1 K=4096)的filter来计算，然后output就会是(1,1,4096)

> W' = 1 + (W - 2*pad - F) // stride = 2 
>
> H' = 1 + (H - 2*pad - F) // stride = 2
>
> K = W' * H' * 512 = 2048?

这个例子我咋没算明白。

### Assignment 

**Fast Layers** 



**pytorch && tf**

很重要的一个区别是，tensorflow是静态图，pytorch是动态图，静态图的好处是先建图，再填数，这样如果该神经网络需要跑很多次就可以更快。而pytorch则是，动态建图，每次新建一个图。