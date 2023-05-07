#!/usr/bin/env python
# coding: utf-8

# # 3. Training

# In[15]:

import os
from typing import List
import numpy as np
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
import random


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
        
    # GPU count
    gpu_num = int(1)
    
    
    def init(self):
        
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        self.gpu_num = int(dist.get_world_size())
        
        
        # Create $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        net = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(device)
        
        
        # Load the previous saved model (optional)
        net.load_state_dict(torch.load('/home/research/yuqijing/sem2project/Diffusion/Parameters/CIFAR10_1000.pkl'))
        
        
        
        
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
        
              
        
        
        # Sample Images from the true density
        N = self.n_samples 
    
        # Define the transform to apply to each image
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((32, 32))
        ])

        # Load the CIFAR10 dataset with the defined transform
        dataset_f = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        # Get the indices of 16 randomly sampled images from the dataset
        indices = random.sample(range(len(dataset_f)), N)

        # Create a tensor to hold the 16 images
        tensor = torch.zeros((N, 3, 32, 32))

        # Load each sampled image into the tensor
        for i, idx in enumerate(indices):
            image, _ = dataset_f[idx]
            tensor[i] = image

        torch.save(tensor, 'true_samples.pt')
        
#         # save dataset
#         torch.save(dataset_f, 'dataset.pt')

        
        
        

      

    

            
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
                for i in range(0, self.n_samples, self.batch_size):
                    batch = x[i:i+self.batch_size]
                    batch_size = batch.shape[0]
                    batch_t = x.new_full((batch_size,), t, dtype=torch.long)
                    x[i:i+self.batch_size] = self.diffusion.p_sample(batch, batch_t)

            
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
            for i in range(self.gpu_num):
                self.vals.append(loss.item())
            
            
            # Compute iterations per second
            if batch_idx > 0:
                end_time = time.time()
                time_per_batch = (end_time - start_time) / batch_idx
                iter_per_second = (1 / time_per_batch) * self.gpu_num
                print(f"\rIterations per second: {iter_per_second:.2f}", end="")

            # Compute percentage of data that has been learnt
            if batch_idx > 0 and batch_idx % 10 == 0:
                percent_complete = batch_idx / total_batches * 100
                print(f"\rIterations per second: {iter_per_second:.2f} | Percentage of data learnt: {percent_complete:.2f}% | Loss: {loss.item():.3f}", end="")
                
            # Compute the time needed to train 
            if batch_idx > 0 and batch_idx % 30 == 0:
                time_per_epoch = ( (total_batches*self.gpu_num) / iter_per_second ) / 3600
                time_total = time_per_epoch * self.epochs
                print()
                print(f"\rTotal Time Required to train (hours): {time_total:.2f}")

        # Print newline after training is complete
        print()


    def run(self):
        """
        ### Training loop
        """
#         start_time = time.time()
        
#         # Count the total number of batches in one epoch
#         local_rank = int(os.environ["LOCAL_RANK"])
#         total_batches = len(self.data_loader)
#         gpu_num = self.gpu_num
#         batch_num = gpu_num * total_batches
#         itera_total = batch_num * self.epochs
        
#         # training loop, epochs
#         for i in range(self.epochs):

#             # Print the current Epochs number
#             print('Current Epochs Number: {} / {}'.format(int(i+1), int(self.epochs)))
#             # Print the current Iteration number
#             print('Current Iterations Number: {} / {}'.format(int((i+1)*batch_num), int(self.epochs*batch_num)))
            
#             # Train the model
#             self.train()

            
#         # save self.vals
#         val_np = np.array(self.vals)
#         np.save('loss.npy', val_np)
        
#         # plot the loss curve
#         plt.plot(self.vals)
#         plt.title("Training Loss")
#         plt.xlabel("Step")
#         plt.ylabel("Loss")
#         plt.savefig('Loss Curve.png')
        
        # Sample some images, and compute sampling time
        start_time_s = time.time()
        self.sample()
        end_time_s = time.time()
        sampling_time = end_time_s - start_time_s
        
        
        
        
        # Save the model
        models = self.eps_model.module.state_dict()
        torch.save(models, 'CIFAR10_1000.pkl')   
        with open('model_parameters.txt', 'w') as f:
            for param_tensor in models:
                print(param_tensor, "\t", models[param_tensor].size(), file=f)
                print(param_tensor, "\t", models[param_tensor].size())
        
        end_time = time.time()
        training_plus_sampling_time = end_time - start_time
        print("###### training_plus_sampling_time: ", training_plus_sampling_time, " seconds")
        print("###### sampling_time: ", sampling_time, " seconds")


        
        
        
        
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


# # CIFAR10 Dataset
# CIFAR10 dataset
class CIFAR10Dataset(torchvision.datasets.CIFAR10):
   def __init__(self, image_size):
       transform = torchvision.transforms.Compose([
       torchvision.transforms.Resize(image_size),
       torchvision.transforms.ToTensor(),
       ])
           
       super().__init__(str(lab.get_data_path()), train=True, download=True, transform=transform)
       
   # get an image
   def __getitem__(self, item):
       return super().__getitem__(item)[0]


# # Create CIFAR10 dataset
# Create CIFAR10 dataset
@option(Configs.dataset, 'CIFAR10')
def cifar10_dataset(c: Configs):
    return CIFAR10Dataset(c.image_size)



# # Create experiment
def main():
        # create experiment
        experiment.create(name='diffusion model CIFAR10')
        # create configurations
        configs = Configs()
        # ***Set configurations. Can be overrided by passing the values in the dictionary
        experiment.configs(configs, {
            'dataset': 'CIFAR10',
            'image_channels': 3,
            'epochs': 1, 
            'n_channels': 64,
            'channel_multipliers': [1, 2, 2, 4],
            'is_attention': [False, True, False, False],
            'n_steps': 1_000,
            'batch_size': 128,
            'n_samples': 16,
            
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





    
    
    

# # Below are not relevant for training purpose.   

def normalize_tensor(tensor, value_range=None):
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if value_range is not None and not isinstance(value_range, tuple):
        raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    for t in tensor:
        norm_range(t, value_range)
    return tensor


tensor = torch.load('samples.pt')
norm_tensor = normalize_tensor(tensor)
torch.save(norm_tensor, 'samples.pt')









# # Display the sampled images
# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# tensor = torch.load('samples.pt')
# print('gen_shape:', tensor.shape)
# N = tensor.size(0)

# # Create a grid of images from the tensor
# grid = torchvision.utils.make_grid(tensor, nrow=int(N**(0.5)), padding=2, normalize=True)

# # Convert the grid to a numpy array and transpose the dimensions to match the expected format by matplotlib
# grid = grid.cpu().numpy().transpose((1, 2, 0))

# # Show the grid of images using matplotlib
# plt.imshow(grid)
# plt.axis('off')
# plt.savefig('gen_samples.png')
# plt.show()
# # os.remove('samples.pt')



# # # display the true images
# tensor = torch.load('true_samples.pt')
# print('true_shape:', tensor.shape)
# N = tensor.size(0)

# # Create a grid of images from the tensor
# grid = torchvision.utils.make_grid(tensor, nrow=int(N**(0.5)), padding=2, normalize=True)

# # Convert the grid to a numpy array and transpose the dimensions to match the expected format by matplotlib
# grid = grid.cpu().numpy().transpose((1, 2, 0))

# # Show the grid of images using matplotlib
# plt.imshow(grid)
# plt.axis('off')
# plt.savefig('true_samples.png')
# plt.show()
# # os.remove('true_samples.pt')


# # Compare two plots
# # Load the saved images
# gen_samples = plt.imread('gen_samples.png')
# true_samples = plt.imread('true_samples.png')

# # Create a new figure with two subplots side by side
# fig, axs = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')

# # Plot the generated samples on the left subplot
# axs[0].imshow(gen_samples)
# axs[0].set_title('Generated Samples')
# axs[0].axis('off')


# # Plot the true samples on the right subplot
# axs[1].imshow(true_samples)
# axs[1].set_title('True Samples')
# axs[1].axis('off')


# # Adjust the spacing between subplots
# plt.subplots_adjust(wspace=0.1)

# # Display the figure
# plt.savefig('Gen_vs_True_CIFAR10.png')
# plt.show()




    






