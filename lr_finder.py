import torch
import matplotlib.pyplot as plt

from train import main
from corolization import ColorfulColorizer, MultinomialCELoss

dset_root = './SUN2012'
batch_size = 12
num_batches = 123
num_epochs = 1
print_freq = 1
encoder = ColorfulColorizer()
criterion = MultinomialCELoss()
start_lr = 0.1
end_lr = 10

lr_mult = (end_lr/start_lr) ** (1/num_batches)
lr_lambda = lambda step: lr_mult ** step

optimizer = torch.optim.SGD(encoder.parameters(),
                            lr=start_lr,
                            momentum=0.9,
                            weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
losses = main(dset_root, batch_size, num_epochs, print_freq, encoder,
         criterion, optimizer, scheduler, step_every_iteration=True)

learning_rates = [start_lr * lr_mult ** x for x in list(range(num_batches))]

plt.ylabel("d/loss")
plt.xlabel("learning rate (log scale)")
plt.plot(learning_rates, losses[0])
plt.xscale('log')
plt.show()