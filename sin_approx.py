import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        alpha = 10.0
        grad = grad_output / (alpha * torch.abs(input) + 1.0) ** 2
        return grad


spike_function = SurrogateSpike.apply


class LIFLayer(nn.Module):
    def __init__(self, input_size, output_size, beta=0.9, threshold=1.0):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.beta = beta
        self.threshold = threshold

    def forward(self, input_spikes, membrane_potential):
        input_current = self.linear(input_spikes)
        membrane_before_spike = self.beta * membrane_potential + input_current
        spikes = spike_function(membrane_before_spike - self.threshold)
        membrane_potential = membrane_before_spike - self.threshold * spikes
        return spikes, membrane_potential


class LIFSNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        num_steps=50,
        beta=0.9,
        threshold=1.0
    ):
        super().__init__()
        self.num_steps = num_steps

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.layers = nn.ModuleList([
            LIFLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                beta=beta,
                threshold=threshold
            )
            for i in range(len(layer_sizes) - 1)
        ])

    def encode_input(self, x):
        x = torch.clamp(x, 0.0, 1.0)

        random_values = torch.rand(
            self.num_steps,
            x.size(0),
            x.size(1),
            device=x.device
        )

        input_spikes = (random_values < x.unsqueeze(0)).float()

        return input_spikes

    def forward(self, x):
        batch_size = x.size(0)

        input_spikes_over_time = self.encode_input(x)

        membrane_potentials = [
            torch.zeros(
                batch_size,
                layer.linear.out_features,
                device=x.device
            )
            for layer in self.layers
        ]

        output_spikes_record = []
        all_spikes_record = []

        for t in range(self.num_steps):
            spikes = input_spikes_over_time[t]

            layer_spikes_at_t = []

            for layer_index, layer in enumerate(self.layers):
                spikes, membrane_potentials[layer_index] = layer(
                    spikes,
                    membrane_potentials[layer_index]
                )

                layer_spikes_at_t.append(spikes)

            output_spikes_record.append(spikes)
            all_spikes_record.append(layer_spikes_at_t)

        output_spikes_record = torch.stack(output_spikes_record, dim=0)

        output = output_spikes_record.mean(dim=0)

        return output, output_spikes_record, all_spikes_record


torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train = torch.linspace(-torch.pi, torch.pi, 300).unsqueeze(1).to(device)
y_train = torch.sin(x_train)

x_train_normalized = (x_train + torch.pi) / (2 * torch.pi)
y_train_normalized = (y_train + 1.0) / 2.0

model = LIFSNN(
    input_size=1,
    hidden_sizes=[64, 64],
    output_size=1,
    num_steps=80,
    beta=0.9,
    threshold=0.5
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_function = nn.MSELoss()

num_epochs = 1000

loss_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()

    output_normalized, output_spikes_record, all_spikes_record = model(
        x_train_normalized
    )

    loss = loss_function(output_normalized, y_train_normalized)

    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

model.eval()

with torch.no_grad():
    output_normalized, output_spikes_record, all_spikes_record = model(
        x_train_normalized
    )

    y_pred = 2.0 * output_normalized - 1.0

x_plot = x_train.cpu().numpy()
y_true_plot = y_train.cpu().numpy()
y_pred_plot = y_pred.cpu().numpy()

plt.figure(figsize=(8, 4))
plt.plot(x_plot, y_true_plot, label="True sin(x)")
plt.plot(x_plot, y_pred_plot, label="SNN approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Approximation of sin(x) using a LIF Spiking Neural Network")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Training loss")
plt.grid(True)
plt.show()

print("\nSome predictions:\n")

for value in [-3.14, -1.57, 0.0, 1.57, 3.14]:
    x_value = torch.tensor([[value]], device=device)

    x_value_normalized = (x_value + torch.pi) / (2 * torch.pi)

    with torch.no_grad():
        output_normalized, _, _ = model(x_value_normalized)
        prediction = 2.0 * output_normalized - 1.0

    true_value = torch.sin(x_value)

    print(f"x = {value: .2f}")
    print(f"true sin(x)      = {true_value.item(): .4f}")
    print(f"SNN prediction   = {prediction.item(): .4f}")
    print()
