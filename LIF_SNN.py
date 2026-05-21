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

class LIFSNN(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, num_steps=20, beta=0.9, threshold=1.0):
        super().__init__()
        self.num_steps = num_steps
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList([LIFLayer(input_size=layer_sizes[i], output_size=layer_sizes[i + 1], beta=beta, threshold=threshold) for i in range(len(layer_sizes) - 1)])

    def encode_input(self, x):
        x = torch.clamp(x, 0.0, 1.0)
        random_values = torch.rand( self.num_steps, x.size(0), x.size(1), device=x.device)
        input_spikes = (random_values < x.unsqueeze(0)).float()
        return input_spikes

    def forward(self, x):
        batch_size = x.size(0)
        input_spikes_over_time = self.encode_input(x)
        membrane_potentials = [ torch.zeros(batch_size, layer.linear.out_features, device=x.device) for layer in self.layers]
        output_spikes_record = []
        all_spikes_record = []

        for t in range(self.num_steps):
            spikes = input_spikes_over_time[t]
            layer_spikes_at_t = []
            for layer_index, layer in enumerate(self.layers):
                spikes, membrane_potentials[layer_index] = layer( spikes, membrane_potentials[layer_index])
                layer_spikes_at_t.append(spikes)

            output_spikes_record.append(spikes)
            all_spikes_record.append(layer_spikes_at_t)

        output_spikes_record = torch.stack(output_spikes_record, dim=0)
        output = output_spikes_record.mean(dim=0)

        return output, output_spikes_record, all_spikes_record

