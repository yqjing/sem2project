#!/usr/bin/env python
# coding: utf-8

# # 3. Training

# In[15]:

import os
from typing import List
import torch 
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
import matplotlib.pyplot as plt
import time
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.distributed as dist


from DenoisingDiffusionModel import *
from UNet import *




# # Configurations

# In[16]:


class Configs(BaseConfigs):
    """
    ## Configurations
    """
    
    device: torch.device = DeviceConfigs()

    # U-Net model for $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
    eps_model: UNet
    # [DDPM algorithm](index.html)
    diffusion: DenoiseDiffusion

    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Image size, 32
    image_size: int = 32
    # Number of channels in the initial feature map, 64
    n_channels: int = 64
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`, [1, 2, 2, 4]
    channel_multipliers: List[int] = [1, 2, 2, 4]
    # The list of booleans that indicate whether to use attention at each resolution, [False, False, False, True]
    is_attention: List[int] = [False, True, False, False]

    # Number of time steps $T$, 1_000
    n_steps: int = 1_000
    # Batch size, 128
    batch_size: int = 128
    # Number of samples to generate, 16
    n_samples: int = 16
    # Learning rate
    learning_rate: float = 2e-5
    
    # Number of training epochs, 1500
    epochs: int = 1_500
        
    # The loss of the training
    vals = []    
    # Dataset
    dataset: torch.utils.data.Dataset
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    optimizer: torch.optim.Adam
        
    
    
    
    
    def init(self):
        
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        
        
        # Create $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        net = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(device)
        
        self.eps_model = DDP(
            net, 
            device_ids=[local_rank],
            output_device=local_rank
        )
        
        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model.module,
            n_steps=self.n_steps,
            device=self.device,
        )
        
        
        # DistributedSampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset,
            shuffle=True,
        )
        
        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, 
            self.batch_size, 
            num_workers=2,
            pin_memory=True, 
            sampler=train_sampler,
        )
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.eps_model.parameters(),
            lr=self.learning_rate
        )

        # Count the total number of parameters in the model
        num_params = sum(
            p.numel() for p in self.eps_model.parameters() if p.requires_grad
        )
        print(f"Number of trainable parameters: {num_params:,}")
        
        # Image logging
        
      

    
    def sample(self):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)

            # Remove noise for $T$ steps
            for t_ in monit.iterate('Sample', self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log samples
            torch.save(x, 'samples.pt')


    def train(self):
        """
        ### Train
        """
       
        start_time = time.time()
        total_batches = len(self.data_loader)
           
        # Iterate through the dataset, batch sizes
        # for data in monit.iterate('Train', self.data_loader):
        for batch_idx, data in enumerate(self.data_loader):
            # Increment global step
            tracker.add_global_step()
            # Move data to device
            data = data.to(self.device)

            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Append the loss to a list for plotting
            self.vals.append(loss.item())
            
            
            # Compute iterations per second
            if batch_idx > 0:
                end_time = time.time()
                time_per_batch = (end_time - start_time) / batch_idx
                iter_per_second = 1 / time_per_batch
                print(f"\rIterations per second: {iter_per_second:.2f}", end="")

            # Compute percentage of data that has been learnt
            if batch_idx > 0 and batch_idx % 10 == 0:
                percent_complete = batch_idx / total_batches * 100
                print(f"\rIterations per second: {iter_per_second:.2f} | Percentage of data learnt: {percent_complete:.2f}%", end="")
                
            # Compute the time needed to train 
            if batch_idx > 0 and batch_idx % 30 == 0:
                time_per_epoch = ( total_batches / iter_per_second ) / 3600
                time_total = time_per_epoch * self.epochs
                print()
                print(f"\rTotal Time Required to train (hours): {time_total:.2f}")
                
        # Print newline after training is complete
        print()


    def run(self):
        """
        ### Training loop
        """

        # training loop, epochs
        for i in range(self.epochs):

            print('Current Epochs Number: {} / {}'.format(int(i+1), int(self.epochs)))
            # Train the model
            self.train()
            
            
            
        # plot the loss curve
        plt.plot(self.vals)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig('Loss Curve.png')
        
        # Sample some images
        self.sample()
        
        # Save the model
        models = self.eps_model.module.state_dict()
        torch.save(models, 'MNIST.pkl')   
        with open('model_parameters.txt', 'w') as f:
            for param_tensor in models:
                print(param_tensor, "\t", models[param_tensor].size(), file=f)
                print(param_tensor, "\t", models[param_tensor].size())
        



        
        
        
        
        # # CelebA HQ dataset
# Train on CelebA HQ dataset
class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, image_size: int):
        super().__init__()
        # ***CelebA images folder
        folder = lab.get_data_path() / 'celebA'
        # list of files
        self._files = [p for p in folder.glob(f'**/*.jpg')]
        # Transformations to resize the image to 32 by 32 and convert to tensor
        self._transform = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size), 
                                                                 torchvision.transforms.ToTensor(),
                                                                ])
    # get size of the dataset
    def __len__(self):
        return len(self._files)
    # get an image
    def __getitem__(self, index: int):
        img = Image.open(self._files[index])
        return self._transform(img)


# # Create CelebA dataset
@option(Configs.dataset, 'CelebA')
def celeb_dataset(c: Configs):
        return CelebADataset(c.image_size)


# # MNIST Dataset
# MNIST dataset
class MNISTDataset(torchvision.datasets.MNIST):
   def __init__(self, image_size):
       transform = torchvision.transforms.Compose([
       torchvision.transforms.Resize(image_size),
       torchvision.transforms.ToTensor(),
       ])
           
       super().__init__(str(lab.get_data_path()), train=True, download=True, transform=transform)
       
   # get an image
   def __getitem__(self, item):
       return super().__getitem__(item)[0]


# # Create MNIST dataset
# Create MNIST dataset
@option(Configs.dataset, 'MNIST')
def mnist_dataset(c: Configs):
    return MNISTDataset(c.image_size)






# # Create experiment
def main():
        # create experiment
        experiment.create(name='diffusion model 1')
        # create configurations
        configs = Configs()
        # ***Set configurations. Can be overrided by passing the values in the dictionary
        experiment.configs(configs, {
            'dataset': 'MNIST',
            'image_channels': 1,
            'epochs': 1_00, 
            'n_channels': 64,
            'channel_multipliers': [1, 2, 2, 4],
            'is_attention': [False, True, False, False],
            'n_steps': 1_000,
            'batch_size': 128,
            
        })
        # initialise
        configs.init()
        # set models for saving and loading 
        experiment.add_pytorch_models({'eps_model': configs.eps_model})
        # start and run the training loop
        with experiment.start():
            configs.run()


# # Start training!
if __name__ == '__main__':
    main()






    
    
    
    # Display the sampled images
import matplotlib.pyplot as plt
import numpy as np
import torch

x = torch.load('samples.pt')
x = x.cpu()
x = x.numpy()

fig, axs = plt.subplots(4, 4, figsize=(10, 10))
for i in range(16):
    row, col = divmod(i, 4)
    axs[row][col].imshow(x[i, 0], cmap='gray')
    axs[row][col].axis('off')
plt.savefig('samples.png')
plt.show()




    






