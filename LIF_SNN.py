import torch
import torch.nn as nn
import torch.nn.functional as F


# automatic differentiation

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    """"context, input """"
    def forward(ctx, input):
      # save as tensor
        ctx.save_for_backward(input)
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        #return tuple
        (input,) = ctx.saved_tensors

        # Sigmoid derivative : sigmoid = torch.sigmoid(alpha * input) ; surrogate_derivative = alpha * sigmoid * (1.0 - sigmoid) ; grad = grad_output * surrogate_derivative
        # Rectangular surrogate : surrogate_derivative = (torch.abs(input) < 1.0).float(); grad = grad_output * surrogate_derivative
        # straight-Through Estimator : grad = grad_output

        # Fast sigmoid surrogate gradient
        alpha = 10.0
        grad = grad_output / (alpha * torch.abs(input) + 1.0) ** 2

        return grad


spike_function = SurrogateSpike.apply

# nn : neural network
class LIFLayer(nn.Module):
    #initializer/constructor
    # this object itself + some defaults
    def __init__(self, input_size, output_size, beta=0.9, threshold=1.0):
      # call the constructor of the parent class (nn.Module) so initialize the nn.Module part of this object
      # PyTorch needs to prepare internal machinery for the layer
        super().__init__()

        # neural network linear layer : nn.linear
        # affine trns
        self.linear = nn.Linear(input_size, output_size)
        # decay factor/leak factor
        self.beta = beta
        self.threshold = threshold

    def forward(self, input_spikes, membrane_potential):

        input_current = self.linear(input_spikes)

        membrane_before_spike = self.beta * membrane_potential + input_current

        spikes = spike_function(membrane_before_spike - self.threshold)

        membrane_potential = membrane_before_spike - self.threshold * spikes

        return spikes, membrane_potential

