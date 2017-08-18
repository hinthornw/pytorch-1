from torch.autograd.function import Function
from torch._thnn import type2backend

from . import _all_functions
from torch.nn.modules.utils import _single, _pair, _triple

class Im2Col(Function):

    def __init__(self, kH, kW, dH, dW, padH, padW, sH, sW):
        self.kH = kH
        self.kW = kW
        self.dH = dH
        self.dW = dW
        self.padH = padH
        self.padW = padW
        self.sH = sH
        self.sW = sW

    def forward(self, input):

        input = input.contiguous()
        backend = type2backend[type(input)]
        output = input.new()
        backend.Im2Col_updateOutput(backend.library_state,
                                    input, output,
                                    self.kH, self.kW,
                                    self.dH, self.dW,
                                    self.padH, self.padW,
                                    self.sH, self.sW)
        self.save_for_backward(input)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        input = input.contiguous()

        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.Im2Col_updateGradInput(backend.library_state,
                                       input, grad_output,
                                       grad_input,
                                       self.kH, self.kW,
                                       self.dH, self.dW,
                                       self.padH, self.padW,
                                       self.sH, self.sW)
        return grad_input

#_all_functions.append(Im2Col)
