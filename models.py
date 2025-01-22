import torch
import torch.nn as nn
import torch.nn.functional as F

############################################# cWGAN-GP Model #############################################


class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, output_dim, hidden_dim, hidden_dim2):
        super(Generator, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(noise_dim + cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 2 * output_dim * (output_dim - 1) // 2)

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)  # concatenate the condition to the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:, :, 0]

        adj = torch.zeros(x.size(0), self.output_dim, self.output_dim, device=x.device)

        idx = torch.triu_indices(self.output_dim, self.output_dim, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)

        return adj


class Discriminator(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, adj_matrix, cond):
        x = self.flatten(adj_matrix)
        x = torch.cat([x, cond], dim=1)  # concatenate the condition to the input
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


############################################# Gradient Penalty #############################################


def gradient_penalty(discriminator, real_samples, fake_samples, conditions, device):
    batch_size = real_samples.size(0)
    epsilon = torch.rand((batch_size, 50, 50), device=device)
    epsilon = epsilon.expand_as(real_samples)

    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated = interpolated.detach()
    interpolated.requires_grad = True

    prob_interpolated = discriminator(interpolated, conditions)
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty
