import torch
from torch.autograd import Variable

from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F


class Im2Col(Module):
    r"""PLACEHOLDER TEMPLATE COMMENT - [TODO](JEB) needs updating
    Applies a 1D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`
    and output :math:`(N, C, L_{out})` can be precisely described as:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, k)  = \max_{{m}=0}^{{kernel\_size}-1} input(N_i, C_j, stride * k + m)
        \end{array}

    | If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
      for :attr:`padding` number of points
    | :attr:`dilation` controls the spacing between the kernel points. It is harder to describe,
      but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if True, will return the max indices along with the outputs.
                        Useful when Unpooling later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})` where
          :math:`L_{out} = floor((L_{in}  + 2 * padding - dilation * (kernel\_size - 1) - 1) / stride + 1)`

    Examples::

        >>> # pool of size=3, stride=2
        >>> m = nn.MaxPool1d(3, stride=2)
        >>> input = autograd.Variable(torch.randn(20, 16, 50))
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kH, kW, dH, dW, padH, padW, sH, sW):
        super(Im2Col, self).__init__()
        self.kH = kH
        self.kW = kW
        self.dH = dH
        self.dW = dW
        self.padH = padH
        self.padW = padW
        self.sH = sH
        self.sW = sW

    def forward(self, input):
        return F.im2col(input, self.kH, self.kW, self.dH, self.dW,
                        self.padH, self.padW, self.sH, self.sW)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'kH=' + str(self.kH) \
            + ', kW=' + str(self.kH) \
            + ', dH=' + str(self.dH) \
            + ', dW=' + str(self.dW) \
            + ', padH=' + str(self.padH) \
            + ', padW=' + str(self.padW) \
            + ', sH=' + str(self.sH) \
            + ', sW=' + str(self.sW) + ')'

