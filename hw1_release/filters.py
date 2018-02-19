import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    hhk = int(Hk/2)
    hwk = int(Wk/2)
    out = np.zeros((Hi, Wi))
    p_image = np.zeros((Hk, Wk))
    print("--------------------------")
    for i in range(0, Hi-hhk):
        for j in range(0, Wi-hwk):
            for ki in range(0, Hk):
                for kj in range(0, Wk):
                    out[i, j] += image[i-ki+hhk,j-kj+hwk]*kernel[ki, kj]
                    if i == 0 and j == 0:
                        p_image[ki, kj] = image[i-ki+hhk,j-kj+hwk]
    print(p_image)
    print(out[0,0])
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = np.zeros((H+pad_height*2, W+pad_width*2))

    ### YOUR CODE HERE
    out[pad_height:H+pad_height, pad_width:W+pad_width] = image.copy()
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    hhk = int(Hk/2)
    hhw = int(Wk/2)
    image_pad = zero_pad(image, Hi, Wi)
    image_pad[Hi:Hi*2,0:Wi] = np.flip(image, 1)
    image_pad[0:Hi,0:Wi] = np.flip(np.flip(image, 1), 0)
    image_pad[0:Hi,Wi:Wi*2] = np.flip(image, 0)
    ### YOUR CODE HERE
    print("======================")
    for i in range(0, Hi):
        for j in range(0, Wi):
            ii = i + Hi
            ij = j + Wi
            out[i, j] = np.sum(image_pad[ii-hhk-1:ii+hhk, ij-hhw-1:ij+hhw]*kernel)
            if i == 0 and j == 0:
                print(ii-hhk-1, ii+hhk, ij-hhw-1, ij+hhw)
                print(image_pad[ii-hhk-1:ii+hhk, ij-hhw-1:ij+hhw])
    ### END YOUR CODE
    print(out[0,0])
    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
