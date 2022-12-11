#%%
import argparse

import torch
import torch.distributions as distributions

import matplotlib.pyplot as plt
from wasserstein_distances.estimate_wasserstein_distance import estimate_wasserstein_kantorovich_rubinstein

parser = argparse.ArgumentParser()


parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--distribution', type=str, default='normal')
parser.add_argument('--num_samples', type=int, default=250_000)

parser.add_argument('--bs', type=int, default=256)
parser.add_argument('--eval_bs', type=int, default=2**16)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--steps', type=int, default=5000)
parser.add_argument('--eval_steps', type=int, default=1000)

args = parser.parse_args()

distribution_type = args.distribution

device = torch.device(f'cuda:{args.gpu}')

if distribution_type == 'normal':
    mean1 = 2
    mean2 = 0

    std1 = 2
    std2 = 1.5

    p1 = distributions.Normal(mean1, std1)
    p2 = distributions.Normal(mean2, std2)
else:
    raise NotImplemented

#%%

#analytic wasserstein-1 distance using the formula for one-dimensional distributions
# integral_R | F1(x) - F2(x) | dx where F1 and F2 are the CDFs of the distributions

#simple numerical integration using composite trapezoidal method
#https://en.wikipedia.org/wiki/Numerical_integration

integration_x_start = -20
integration_x_end = 20

num_subdivisions = 1_000_000

def composite_trapezoidal(f, a, b, n):
    h = (b-a)/n
    inner_points = a + h * torch.arange(1, n)
    integral = h * torch.sum(f(inner_points))
    integral += h * 0.5 * torch.sum(f(torch.FloatTensor([a,b])))
    return integral

def integrand(x):
   return torch.abs(p1.cdf(x) - p2.cdf(x))

wasserstein_1_distance_integration = composite_trapezoidal(integrand, integration_x_start, integration_x_end, num_subdivisions)
print(f'Numerical integration Wasserstein 1 distance: {wasserstein_1_distance_integration:.5f}')

num_samples = args.num_samples
samples_p1 = p1.sample((num_samples,))
samples_p2 = p2.sample((num_samples,))
wasserstein_1_distance_integration_duality = estimate_wasserstein_kantorovich_rubinstein(samples_p1, samples_p2, device,
                                                                                         bs=args.bs,
                                                                                         eval_bs=args.eval_bs,
                                                                                         optim=args.optim, lr=args.lr,
                                                                                         steps=args.steps,
                                                                                         eval_steps=args.eval_steps)

#draw pdfs
x_subdivs = 1_000

xs = torch.linspace(-10, 10, x_subdivs)

pdf1 = p1.log_prob(xs).exp()
pdf2 = p2.log_prob(xs).exp()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs, pdf1, label='Distribution 1', color='red')
ax.plot(xs, pdf2, label='Distribution 2', color='black')
ax.set_xlabel('x')
ax.set_ylabel('pdf')
ax.set_title(f'Wasserstein-1 - Integration {wasserstein_1_distance_integration:.5f}'
             f' - Duality {wasserstein_1_distance_integration_duality:.5f}')
ax.legend()
plt.show()


