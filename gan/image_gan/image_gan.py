"""
Introduction Convolutional Neural Networks: 
    https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/
    https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-cnn-26a14c2ea29

Introduction Adversarial Networks:
    Generative Adversarial Networks: https://wiki.pathmind.com/generative-adversarial-network-gan
    Adversarial Autoencoder: https://medium.com/vitrox-publication/adversarial-auto-encoder-aae-a3fc86f71758
"""

import numpy as np
import torch
import torchvision
from pytorch_model_summary import summary
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import pickle
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image settings
image_data_path = "../../../../Data/Images"
image_size = 128
image_channels = 3

# model settings
latent_dim = 64
gen_conv_channel_counts = [ 512, 128, 32, 8 ]
gen_conv_kernel_size = 5
gen_dense_layer_sizes = [ 128 ]

crit_conv_channel_counts = [ 8, 32, 128, 512 ]
crit_conv_kernel_size = 5
crit_dense_layer_sizes = [ 128 ]

save_models = False
save_tscript = False
save_weights = True

# load model weights
load_weights = False
generator_weights_file = "results/weights/generator_weights_epoch_400"
critique_weights_file = "results/weights/critique_weights_epoch_400"

# training settings
batch_size = 16
train_percentage = 0.8 # train / test split
test_percentage  = 0.2
gen_learning_rate = 1e-4
crit_learning_rate = 1e-4
epochs = 1000
weight_save_interval = 10
save_history = False

# create dataset
transform = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size), 
                                            torchvision.transforms.ToTensor()])

full_dataset = torchvision.datasets.ImageFolder(image_data_path, transform=transform)
dataset_size = len(full_dataset)

test_size = int(test_percentage * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create Models

# Critique
class Critique(nn.Module):
    def __init__(self, image_size, image_channels, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.image_size = image_size
        self.image_channels = image_channels
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create convolutional layers
        conv_layers = []
        
        stride = (self.conv_kernel_size - 1) // 2
        padding = stride
        
        conv_layers.append(("critique_conv_0", nn.Conv2d(image_channels, self.conv_channel_counts[0], self.conv_kernel_size, stride=stride, padding=padding)))
        conv_layers.append(("critique_lrelu_0", nn.LeakyReLU(0.2)))
        conv_layers.append(("critique_bnorm_0", nn.BatchNorm2d(self.conv_channel_counts[0])))
        
        conv_layer_count = len(conv_channel_counts)
        
        for layer_index in range(1, conv_layer_count):
            conv_layers.append(("critique_conv_{}".format(layer_index), nn.Conv2d(self.conv_channel_counts[layer_index-1], self.conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding)))
            conv_layers.append(("critique_lrelu_{}".format(layer_index), nn.LeakyReLU(0.2)))
            conv_layers.append(("critique_bnorm_{}".format(layer_index), nn.BatchNorm2d(self.conv_channel_counts[layer_index])))

        self.conv_layers = nn.Sequential(OrderedDict(conv_layers))
        self.flatten = nn.Flatten(start_dim=1)
        
        # create dense layers
        dense_layers = []
        
        last_conv_layer_size = image_size // np.power(2, len(conv_channel_counts))
        
        #print("last_conv_layer_size ", last_conv_layer_size)
        
        dense_layer_input_size = conv_channel_counts[-1] * last_conv_layer_size * last_conv_layer_size
        
        #print("dense_layer_input_size ", dense_layer_input_size)
        
        dense_layers.append(("critique_dense_0", nn.Linear(dense_layer_input_size, self.dense_layer_sizes[0])))
        dense_layers.append(("critique_dense_lrelu_0", nn.LeakyReLU(0.2)))
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("critique_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("critique_dense_lrelu_{}".format(layer_index), nn.LeakyReLU(0.2)))

        dense_layers.append(("encoder_dense_{}".format(len(self.dense_layer_sizes)), nn.Linear(self.dense_layer_sizes[-1], 1)))
        dense_layers.append(("encoder_dense_sigmoid_{}".format(len(self.dense_layer_sizes)), nn.Sigmoid()))

        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
    def forward(self, x):
        
        #print("x1 s", x.shape)
        
        x = self.conv_layers(x)
        
        #print("x2 s", x.shape)
        
        x = self.flatten(x)
        
        #print("x3 s", x.shape)
        
        yhat = self.dense_layers(x)
        
        #print("yhat s", yhat.shape)
        
        return yhat

critique = Critique(image_size, image_channels, crit_conv_channel_counts, crit_conv_kernel_size, crit_dense_layer_sizes).to(device)

print(critique)

"""
test_input = torch.zeros((1, image_channels, image_size, image_size)).to(device)
test_output = critique(test_input)
"""

if save_models == True:
    critique.eval()
    
    # save using pickle
    torch.save(critique, "results/models/critique.pth")
    
    # save using onnx
    x = torch.zeros((1, image_channels, image_size, image_size)).to(device)
    torch.onnx.export(critique, x, "results/models/critique.onnx")
    
    critique.train()

if save_tscript == True:
    critique.eval()
    
    # save using TochScript
    x = torch.rand((1, image_channels, image_size, image_size), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(critique, x)
    script_module.save("results/models/critique.pt")
    
    critique.train()

if load_weights and critique_weights_file:
    critique.load_state_dict(torch.load(critique_weights_file))

# Generator 
class Generator(nn.Module):
    
    def __init__(self, latent_dim, image_size, image_channels, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.image_channels = image_channels
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create dense layers
        dense_layers = []
                
        dense_layers.append(("generator_dense_0", nn.Linear(latent_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("generator_relu_0", nn.ReLU()))
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("generator_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("generator_dense_relu_{}".format(layer_index), nn.ReLU()))
            
        last_conv_layer_size = int(image_size // np.power(2, len(conv_channel_counts)))
        preflattened_size = [conv_channel_counts[0], last_conv_layer_size, last_conv_layer_size]
        dense_layer_output_size = conv_channel_counts[0] * last_conv_layer_size * last_conv_layer_size

        print("preflattened_size ", preflattened_size)
    
        dense_layers.append(("generator_dense_{}".format(len(self.dense_layer_sizes)), nn.Linear(self.dense_layer_sizes[-1], dense_layer_output_size)))
        
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=preflattened_size)
        
        # create convolutional layers
        conv_layers = []
        
        stride = (self.conv_kernel_size - 1) // 2
        padding = stride
        output_padding = 1
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            conv_layers.append(("generator_bnorm_{}".format(layer_index), nn.BatchNorm2d(conv_channel_counts[layer_index-1])))
            conv_layers.append(("generator_conv_{}".format(layer_index), nn.ConvTranspose2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding)))
            conv_layers.append(("generator_lrelu_{}".format(layer_index), nn.LeakyReLU(0.2)))
            
        conv_layers.append(("generator_bnorm_{}".format(conv_layer_count), nn.BatchNorm2d(conv_channel_counts[-1])))
        conv_layers.append(("generator_conv_{}".format(conv_layer_count), nn.ConvTranspose2d(conv_channel_counts[-1], self.image_channels, self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding)))
        conv_layers.append(("generator_sigmoid_{}".format(conv_layer_count), nn.Sigmoid()))
        
        self.conv_layers = nn.Sequential(OrderedDict(conv_layers))

    def forward(self, x):
        
        #print("x1 s ", x.shape)
        
        x = self.dense_layers(x)
        
        #print("x2 s ", x.shape)
        
        x = self.unflatten(x)
        
        #print("x3 s ", x.shape)

        yhat = self.conv_layers(x)
        
        #print("yhat s ", yhat.shape)

        return yhat

generator = Generator(latent_dim, image_size, image_channels, gen_conv_channel_counts, gen_conv_kernel_size, gen_dense_layer_sizes).to(device)

print(generator)

"""
test_input = torch.zeros((1, latent_dim)).to(device)
test_output = generator(test_input)
"""

if save_models == True:
    generator.eval()
    
    # save using pickle
    torch.save(generator, "results/models/generator.pth")
    
    # save using onnx
    x = torch.zeros((1, latent_dim)).to(device)
    torch.onnx.export(generator, x, "results/models/generator.onnx")
    
    generator.train()

if save_tscript == True:
    generator.eval()
    
    # save using TochScript
    x = torch.rand((1, latent_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(critique, x)
    script_module.save("results/models/generator.pt")
    
    generator.train()

if load_weights and generator_weights_file:
    generator.load_state_dict(torch.load(generator_weights_file))
    
#Training

critique_optimizer = torch.optim.Adam(critique.parameters(), lr=crit_learning_rate)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=gen_learning_rate)

bce_loss = nn.BCELoss()

# crictique loss function
def crit_loss(crit_real_output, crit_fake_output):
    _real_loss = bce_loss(crit_real_output, torch.ones_like(crit_real_output).to(device))
    _fake_loss = bce_loss(crit_fake_output, torch.zeros_like(crit_fake_output).to(device))

    _loss = (_real_loss + _fake_loss) * 0.5
    return _loss

# generator loss
def gen_crit_loss(crit_fake_output):
    _loss = bce_loss(crit_fake_output, torch.ones_like(crit_fake_output).to(device))
    return _loss

def gen_loss(crit_fake_output):
    _loss = gen_crit_loss(crit_fake_output)
    return _loss

def crit_train_step(real_poses, random_encodings):
    
    critique_optimizer.zero_grad()
    
    with torch.no_grad():
        fake_output = generator(random_encodings)
    real_output = real_poses
        
    crit_real_output =  critique(real_output)
    crit_fake_output =  critique(fake_output)   
    
    _crit_loss = crit_loss(crit_real_output, crit_fake_output)
        
    _crit_loss.backward()
    critique_optimizer.step()
    
    return _crit_loss

def crit_test_step(real_poses, random_encodings):
    with torch.no_grad():
        fake_output = generator(random_encodings)
        real_output = real_poses
        
        crit_real_output =  critique(real_output)
        crit_fake_output =  critique(fake_output)   
    
        _crit_loss = crit_loss(crit_real_output, crit_fake_output)

    return _crit_loss

def gen_train_step(random_encodings):
    
    generator_optimizer.zero_grad()
    
    generated_poses = generator(random_encodings)
    
    crit_fake_output = critique(generated_poses)
    
    _gen_loss = gen_loss(crit_fake_output)
    
    _gen_loss.backward()
    generator_optimizer.step()
 
    return _gen_loss

def gen_test_step(random_encodings):
    with torch.no_grad():
        generated_poses = generator(random_encodings)
    
        crit_fake_output = critique(generated_poses)
    
        _gen_loss = gen_loss(crit_fake_output)
    
    return _gen_loss

def plot_gan_outputs(decoder, epoch, n=5):
    
    generator.eval()
    
    plt.figure(figsize=(10,4.5))
    for i in range(n):
      ax = plt.subplot(1,n,i+1)
      
      decoder.eval()
      with torch.no_grad():
         random_encoding = torch.randn((1, latent_dim)).to(device)
         gen_img  = decoder(random_encoding)    
      decoder.train()
      
      gen_img = gen_img.cpu().squeeze().numpy()
      gen_img = np.clip(gen_img, 0.0, 1.0)
      gen_img = np.moveaxis(gen_img, 0, 2)
      
      plt.imshow(gen_img)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == 0:
          ax.set_title("Epoch {}: Generated Images".format(epoch))
    plt.show()
    
    generator.train()

def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["gen train"] = []
    loss_history["gen test"] = []
    loss_history["crit train"] = []
    loss_history["crit test"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        crit_train_loss_per_epoch = []
        gen_train_loss_per_epoch = []
        
        for train_batch, _ in train_dataloader:
            train_batch = train_batch.to(device)
            
            random_encodings = torch.randn((train_batch.shape[0], latent_dim)).to(device)

            # start with critique training
            _crit_train_loss = crit_train_step(train_batch, random_encodings)
            
            _crit_train_loss = _crit_train_loss.detach().cpu().numpy()

            crit_train_loss_per_epoch.append(_crit_train_loss)
            
            # now train the generator
            for iter in range(2):
                _gen_loss = gen_train_step(random_encodings)
            
                _gen_loss = _gen_loss.detach().cpu().numpy()
            
                gen_train_loss_per_epoch.append(_gen_loss)
        
        crit_train_loss_per_epoch = np.mean(np.array(crit_train_loss_per_epoch))
        gen_train_loss_per_epoch = np.mean(np.array(gen_train_loss_per_epoch))

        crit_test_loss_per_epoch = []
        gen_test_loss_per_epoch = []
        
        for test_batch, _ in test_dataloader:
            test_batch = test_batch.to(device)
            
            random_encodings = torch.randn((train_batch.shape[0], latent_dim)).to(device)
            
            # start with critique testing
            _crit_test_loss = crit_test_step(train_batch, random_encodings)
            
            _crit_test_loss = _crit_test_loss.detach().cpu().numpy()

            crit_test_loss_per_epoch.append(_crit_test_loss)
            
            # now test the generator
            _gen_loss = gen_test_step(random_encodings)
            
            _gen_loss = _gen_loss.detach().cpu().numpy()
            
            gen_test_loss_per_epoch.append(_gen_loss)

        crit_test_loss_per_epoch = np.mean(np.array(crit_test_loss_per_epoch))
        gen_test_loss_per_epoch = np.mean(np.array(gen_test_loss_per_epoch))
        
        if epoch % weight_save_interval == 0 and save_weights == True:
            torch.save(critique.state_dict(), "results/weights/critique_weights_epoch_{}".format(epoch))
            torch.save(generator.state_dict(), "results/weights/generator_weights_epoch_{}".format(epoch))
        
        plot_gan_outputs(generator, epoch, n=5)
        

        loss_history["gen train"].append(gen_train_loss_per_epoch)
        loss_history["gen test"].append(gen_test_loss_per_epoch)
        loss_history["crit train"].append(crit_train_loss_per_epoch)
        loss_history["crit test"].append(crit_test_loss_per_epoch)

        print ('epoch {} : gen train: {:01.4f} gen test: {:01.4f} crit train {:01.4f} crit test {:01.4f} time {:01.2f}'.format(epoch + 1, gen_train_loss_per_epoch, gen_test_loss_per_epoch, crit_train_loss_per_epoch, crit_test_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_dataloader, test_dataloader, epochs)




"""
epochs = 2000

# gan only training loop
# outer loop over the training epochs
for epoch in range(epochs):
    
    critique_epoch_loss = 0
    generator_epoch_loss = 0
    
    tick = time.time()
    
    for batch_features, _ in train_dataloader:
        
        batch_features = batch_features.to(device)
    
        # image prior train step
        critique_optimizer.zero_grad()
        
        random_encodings = torch.randn((batch_features.shape[0], latent_dim)).to(device)
        
        with torch.no_grad():
            fake_output = generator(random_encodings)
        real_output = batch_features
    
        critique_real_output =  critique(real_output)
        critique_fake_output =  critique(fake_output)   
        
        critique_loss = crit_loss(critique_real_output, critique_fake_output)
        
        critique_loss.backward()
        critique_optimizer.step()
        
        critique_epoch_loss += critique_loss.item()
        
        for iter in range(2):
        
            # generator train step
            generator_optimizer.zero_grad()
            
            generated_images = generator(random_encodings)
    
            critique_fake_output = critique(generated_images)
                
            generator_loss = gen_loss(critique_fake_output)
            
            generator_loss.backward()
            generator_optimizer.step()
        
        generator_epoch_loss += generator_loss.item()
    
    # compute the epoch training loss
    critique_epoch_loss = critique_epoch_loss / len(train_dataloader)
    generator_epoch_loss = generator_epoch_loss / len(train_dataloader)
    
    tock = time.time()
    
    # display the epoch training loss
    #plot_ae_outputs(encoder,decoder, epoch, n=5)
    plot_gan_outputs(generator, epoch, n=5)
    print("epoch : {}/{}, di_loss  = {:.6f} dec_loss  = {:.6f}, time = {:.2f}".format(epoch + 1, epochs, critique_epoch_loss, generator_epoch_loss, (tock - tick)))
"""

