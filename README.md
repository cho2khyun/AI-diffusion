# AI-diffusion

1. Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

2. Set initial variables
x0 = -5
n_steps = 100
alphas = 1. - torch.linspace(0.001, 0.2, n_steps)
alphas_cumprod = torch.cumprod(alphas, axis=0)

3. Plot sqrt_alphas_cumprod and sqrt_one_minus_alphas_cumprod
plt.plot(sqrt_alphas_cumprod, label="sqrt_alphas_cumprod")
plt.plot(sqrt_one_minus_alphas_cumprod, label="sqrt_one_minus_alphas_cumprod")
plt.legend()

4. Define q_sample function
def q_sample(x_0, t, noise):
    return sqrt_alphas_cumprod.gather(-1, t) * x_0 + sqrt_one_minus_alphas_cumprod.gather(-1, t) * noise

5. Plot histograms of noised_x for different values of t
for t in [1, n_steps // 10, n_steps // 2, n_steps - 1]:
    noised_x = q_sample(x0, torch.tensor(t), torch.randn(1000))
    plt.hist(noised_x.numpy(), bins=100, alpha=0.5, label=f"t={t}");
plt.legend()

6. Define DenoiseModel class
class DenoiseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(1.))
        self.b = torch.nn.Parameter(torch.tensor(0.))
        self.c = torch.nn.Parameter(torch.tensor(0.))
        
    def forward(self, x, t):
        return self.a * x + self.b * t + self.c

7. Define p_loss function
def p_loss(x, t):
    noise = torch.randn(t.shape)
    noisy_x = q_sample(x, t, noise)
    noise_computed = denoise(noisy_x, t)
    return F.mse_loss(noise, noise_computed)

8. Initialize DenoiseModel and optimizer
denoise = DenoiseModel()
optimizer = torch.optim.SGD(denoise.parameters(),lr=0.001)

>> The previous code uses Adam Optimizer to perform learning. I expect to use other optimizers, AdaGrad & SGD optimizers, to improve performance. When the AdaGrad Optimizer and SGD Optimizer were applied, it was configured that the loss increased by 0.2 and 0.1 respectively compared to the use of the adam optimizer.

9. Set training parameters
n_epochs = 10000
batch_size = 1000

10. Train the model
for step in range(n_epochs):
    optimizer.zero_grad()
    t = torch.randint(0, n_steps, (batch_size, ))
    loss = p_loss(x0, t)
    loss.backward()
    if step % (n_epochs // 10) == 0:
        print(f"loss={loss.item():.4f}; a={denoise.a.item():.4f}, b={denoise.b.item():.4f}, c={denoise.c.item():.4f}")
    optimizer.step()
print(f"final: loss={loss.item():.4f}; a={denoise.a.item():.4f}, b={denoise.b.item():.4f}, c={denoise.c.item():.4f}")

11. Calculate posterior variance
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
posterior_variance = (1 - alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

12. Define p_sample function
def p_sample(x, t):
    alpha_t = alphas.gather(-1, t)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod.gather(-1, t)
    model_mean = torch
