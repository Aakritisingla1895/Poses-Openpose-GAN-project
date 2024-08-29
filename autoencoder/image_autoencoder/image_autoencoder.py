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
ae_conv_channel_counts = [ 8, 32, 128, 512 ]
ae_conv_kernel_size = 5
ae_dense_layer_sizes = [ 128 ]

disc_prior_dense_layer_sizes = [ 128, 128 ]

save_models = False
save_tscript = False
save_weights = True

# load model weights
load_weights = False
disc_prior_weights_file = "results/weights/disc_prior_weights_epoch_400"
encoder_weights_file = "results/weights/encoder_weights_epoch_400"
decoder_weights_file = "results/weights/decoder_weights_epoch_400"

# training settings
batch_size = 16
train_percentage = 0.8 # train / test split
test_percentage  = 0.2
dp_learning_rate = 5e-3
ae_learning_rate = 1e-3
ae_rec_loss_scale = 1.0
ae_prior_loss_scale = 0.1
epochs = 100
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
class DiscriminatorPrior(nn.Module):
    def __init__(self, latent_dim, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.dense_layer_sizes = dense_layer_sizes
        
        # create dense layers
        dense_layers = []
        
        dense_layers.append(("disc_prior_dense_0", nn.Linear(latent_dim, dense_layer_sizes[0])))
        dense_layers.append(("disc_prior_elu_0", nn.ELU()))
        
        dense_layer_count = len(dense_layer_sizes)

        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("disc_prior_dense_{}".format(layer_index), nn.Linear(dense_layer_sizes[layer_index - 1], dense_layer_sizes[layer_index])))
            dense_layers.append(("disc_prior_elu_{}".format(layer_index), nn.ELU()))
    
        dense_layers.append(("disc_prior_dense_{}".format(dense_layer_count), nn.Linear(dense_layer_sizes[-1], 1)))
        dense_layers.append(("disc_prior_sigmoid_{}".format(dense_layer_count), nn.Sigmoid()))
        
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
    def forward(self, x):
        
        #print("x1 s", x.shape)
        
        yhat = self.dense_layers(x)
        
        #print("yhat s", yhat.shape)
        
        return yhat

disc_prior = DiscriminatorPrior(latent_dim, disc_prior_dense_layer_sizes).to(device)

print(disc_prior)

"""
test_input = torch.zeros((1, latent_dim)).to(device)
test_output = disc_prior(test_input)
"""

if save_models == True:
    disc_prior.eval()
    
    # save using pickle
    torch.save(disc_prior, "results/models/disc_prior.pth")
    
    # save using onnx
    x = torch.zeros((1, latent_dim)).to(device)
    torch.onnx.export(disc_prior, x, "results/models/disc_prior.onnx")
    
    disc_prior.train()

if save_tscript == True:
    disc_prior.eval()
    
    # save using TochScript
    x = torch.rand((1, latent_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(disc_prior, x)
    script_module.save("results/models/disc_prior.pt")
    
    disc_prior.train()

if load_weights and disc_prior_weights_file:
    disc_prior.load_state_dict(torch.load(disc_prior_weights_file))
    
    
# Encoder
class Encoder(nn.Module):
    
    def __init__(self, latent_dim, image_size, image_channels, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.image_channels = image_channels
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create convolutional layers
        conv_layers = []
        
        stride = (self.conv_kernel_size - 1) // 2
        padding = stride
        
        conv_layers.append(("encoder_conv_0", nn.Conv2d(self.image_channels, conv_channel_counts[0], self.conv_kernel_size, stride=stride, padding=padding)))
        conv_layers.append(("encoder_bnorm_0", nn.BatchNorm2d(conv_channel_counts[0])))
        conv_layers.append(("encoder_lrelu_0", nn.LeakyReLU(0.2)))
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            conv_layers.append(("encoder_conv_{}".format(layer_index), nn.Conv2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding)))
            conv_layers.append(("encoder_bnorm_{}".format(layer_index), nn.BatchNorm2d(conv_channel_counts[layer_index])))
            conv_layers.append(("encoder_lrelu_{}".format(layer_index), nn.LeakyReLU(0.2)))

        self.conv_layers = nn.Sequential(OrderedDict(conv_layers))
        
        self.flatten = nn.Flatten()
        
        # create dense layers
        dense_layers = []
        
        last_conv_layer_size = int(image_size // np.power(2, len(conv_channel_counts)))
        preflattened_size = [conv_channel_counts[-1], last_conv_layer_size, last_conv_layer_size]
        dense_layer_input_size = conv_channel_counts[-1] * last_conv_layer_size * last_conv_layer_size
        
        dense_layers.append(("encoder_dense_0", nn.Linear(dense_layer_input_size, self.dense_layer_sizes[0])))
        dense_layers.append(("encoder_relu_0", nn.ReLU()))
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("encoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("encoder_dense_relu_{}".format(layer_index), nn.ReLU()))
    
        dense_layers.append(("encoder_dense_{}".format(len(self.dense_layer_sizes)), nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)))
        
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))

    def forward(self, x):
        
        #print("x1 s ", x.shape)
        
        x = self.conv_layers(x)
        
        #print("x2 s ", x.shape)
        
        x = self.flatten(x)
        
        #print("x3 s ", x.shape)

        yhat = self.dense_layers(x)
        
        #print("yhat s ", yhat.shape)

        return yhat

encoder = Encoder(latent_dim, image_size, image_channels, ae_conv_channel_counts, ae_conv_kernel_size, ae_dense_layer_sizes).to(device)

print(encoder)

"""
test_input = torch.zeros((1, image_channels, image_size, image_size)).to(device)
test_output = encoder(test_input)
"""

if save_models == True:
    encoder.eval()
    
    # save using pickle
    torch.save(encoder, "results/models/encoder.pth")
    
    # save using onnx
    x = torch.zeros((1, image_channels, image_size, image_size)).to(device)
    torch.onnx.export(encoder, x, "results/models/encoder.onnx")
    
    encoder.train()

if save_tscript == True:
    encoder.eval()
    
    # save using TochScript
    x = torch.rand((1, image_channels, image_size, image_size), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(encoder, x)
    script_module.save("results/models/encoder.pt")
    
    encoder.train()

if load_weights and encoder_weights_file:
    encoder.load_state_dict(torch.load(encoder_weights_file))

# Decoder 
class Decoder(nn.Module):
    
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
                
        dense_layers.append(("decoder_dense_0", nn.Linear(latent_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("decoder_relu_0", nn.ReLU()))
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("decoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("decoder_dense_relu_{}".format(layer_index), nn.ReLU()))
            
        last_conv_layer_size = int(image_size // np.power(2, len(conv_channel_counts)))
        preflattened_size = [conv_channel_counts[0], last_conv_layer_size, last_conv_layer_size]
        dense_layer_output_size = conv_channel_counts[0] * last_conv_layer_size * last_conv_layer_size

        dense_layers.append(("decoder_dense_{}".format(len(self.dense_layer_sizes)), nn.Linear(self.dense_layer_sizes[-1], dense_layer_output_size)))
        dense_layers.append(("decoder_dense_relu_{}".format(len(self.dense_layer_sizes)), nn.ReLU()))
        
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=preflattened_size)
        
        # create convolutional layers
        conv_layers = []
        
        stride = (self.conv_kernel_size - 1) // 2
        padding = stride
        output_padding = 1
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            conv_layers.append(("decoder_bnorm_{}".format(layer_index), nn.BatchNorm2d(conv_channel_counts[layer_index-1])))
            conv_layers.append(("decoder_conv_{}".format(layer_index), nn.ConvTranspose2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding)))
            conv_layers.append(("decoder_lrelu_{}".format(layer_index), nn.LeakyReLU(0.2)))
            
        conv_layers.append(("decoder_bnorm_{}".format(conv_layer_count), nn.BatchNorm2d(conv_channel_counts[-1])))
        conv_layers.append(("decoder_conv_{}".format(conv_layer_count), nn.ConvTranspose2d(conv_channel_counts[-1], self.image_channels, self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding)))
        
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
    
ae_conv_channel_counts_reversed = ae_conv_channel_counts.copy()
ae_conv_channel_counts_reversed.reverse()
    
ae_dense_layer_sizes_reversed = ae_dense_layer_sizes.copy()
ae_dense_layer_sizes_reversed.reverse()

decoder = Decoder(latent_dim, image_size, image_channels, ae_conv_channel_counts_reversed, ae_conv_kernel_size, ae_dense_layer_sizes_reversed).to(device)

print(decoder)

"""
test_input = torch.zeros((1, latent_dim)).to(device)
test_output = generator(test_input)
"""

if save_models == True:
    decoder.eval()
    
    # save using pickle
    torch.save(decoder, "results/models/decoder.pth")
    
    # save using onnx
    x = torch.zeros((1, latent_dim)).to(device)
    torch.onnx.export(decoder, x, "results/models/decoder.onnx")
    
    decoder.train()

if save_tscript == True:
    decoder.eval()
    
    # save using TochScript
    x = torch.rand((1, latent_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(decoder, x)
    script_module.save("results/models/decoder.pt")
    
    decoder.train()

if load_weights and decoder_weights_file:
    decoder.load_state_dict(torch.load(decoder_weights_file))
    
#Training

disc_prior_optimizer = torch.optim.Adam(disc_prior.parameters(), lr=dp_learning_rate)
ae_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=ae_learning_rate)

mse_loss = torch.nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()

# function returning normal distributed random data 
# serves as reference for the discriminator to distinguish the encoders prior from
def sample_normal(shape):
    return torch.tensor(np.random.normal(size=shape), dtype=torch.float32).to(device)

# discriminator prior loss function
def disc_prior_loss(disc_real_output, disc_fake_output):
    _real_loss = bce_loss(disc_real_output, torch.ones_like(disc_real_output).to(device))
    _fake_loss = bce_loss(disc_fake_output, torch.zeros_like(disc_fake_output).to(device))

    _total_loss = (_real_loss + _fake_loss) * 0.5
    return _total_loss

def ae_loss(y, yhat, disc_pior_fake_output):
    _ae_rec_loss = mse_loss(y, yhat)
    _disc_prior_fake_loss = bce_loss(disc_pior_fake_output, torch.ones_like(disc_pior_fake_output).to(device))
    
    _total_loss = 0.0
    _total_loss += _ae_rec_loss * ae_rec_loss_scale
    _total_loss += _disc_prior_fake_loss * ae_prior_loss_scale
    
    return _total_loss, _ae_rec_loss, _disc_prior_fake_loss

def disc_prior_train_step(target_images):
    # have normal distribution and encoder produce real and fake outputs, respectively
    
    with torch.no_grad():
        encoder_output = encoder(target_images)
        
    real_output = sample_normal(encoder_output.shape)
    
    # let discriminator distinguish between real and fake outputs
    disc_real_output =  disc_prior(real_output)
    disc_fake_output =  disc_prior(encoder_output)   
    _disc_loss = disc_prior_loss(disc_real_output, disc_fake_output)
    
    # Backpropagation
    disc_prior_optimizer.zero_grad()
    _disc_loss.backward()
    disc_prior_optimizer.step()
    
    return _disc_loss

def ae_train_step(target_images):
    
    encoder_output = encoder(target_images)
    pred_images = decoder(encoder_output)

    disc_fake_output = disc_prior(encoder_output)
    
    _ae_loss, _ae_rec_loss, _disc_prior_fake_loss = ae_loss(target_images, pred_images, disc_fake_output) 

    ae_optimizer.zero_grad()
    _ae_loss.backward()
    ae_optimizer.step()
    
    return _ae_loss, _ae_rec_loss, _disc_prior_fake_loss

def ae_test_step(target_images):

    with torch.no_grad():
        encoder_output = encoder(target_images)
        pred_images = decoder(encoder_output)
        disc_fake_output = disc_prior(encoder_output)
    
        _ae_loss, _ae_rec_loss, _disc_prior_fake_loss = ae_loss(target_images, pred_images, disc_fake_output) 
    
    return _ae_loss, _ae_rec_loss, _disc_prior_fake_loss

def plot_ae_outputs(encoder, decoder, epoch, n=5):
    
    encoder.eval()
    decoder.eval()
    
    plt.figure(figsize=(10,4.5))
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[i][0].unsqueeze(0).to(device)

      with torch.no_grad():
         rec_img  = decoder(encoder(img))    
      
      img = img.cpu().squeeze().numpy()
      img = np.clip(img, 0.0, 1.0)
      img = np.moveaxis(img, 0, 2)
      
      rec_img = rec_img.cpu().squeeze().numpy()
      rec_img = np.clip(rec_img, 0.0, 1.0)
      rec_img = np.moveaxis(rec_img, 0, 2)
      
      plt.imshow(img)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img)  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title("Epoch {}: Reconstructed images".format(epoch))
    plt.show()
    #plt.savefig("epoch_{0:05d}.jpg".format(epoch))
    plt.close()
    
    decoder.train()
    decoder.train()

def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["ae train"] = []
    loss_history["ae test"] = []
    loss_history["ae rec"] = []
    loss_history["ae prior"] = []
    loss_history["disc prior"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        ae_train_loss_per_epoch = []
        ae_rec_loss_per_epoch = []
        ae_prior_loss_per_epoch = []
        disc_prior_loss_per_epoch = []
        
        for train_batch, _ in train_dataloader:
            train_batch = train_batch.to(device)

            # start with discriminator training
            _disc_prior_train_loss = disc_prior_train_step(train_batch)
            _disc_prior_train_loss = _disc_prior_train_loss.detach().cpu().numpy()
            disc_prior_loss_per_epoch.append(_disc_prior_train_loss)
            
            # now train the autoencoder
            _ae_loss, _ae_rec_loss, _ae_prior_loss = ae_train_step(train_batch)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            _ae_rec_loss = _ae_rec_loss.detach().cpu().numpy()
            _ae_prior_loss = _ae_prior_loss.detach().cpu().numpy()
            
            ae_train_loss_per_epoch.append(_ae_loss)
            ae_rec_loss_per_epoch.append(_ae_rec_loss)
            ae_prior_loss_per_epoch.append(_ae_prior_loss)
        
        ae_train_loss_per_epoch = np.mean(np.array(ae_train_loss_per_epoch))
        ae_rec_loss_per_epoch = np.mean(np.array(ae_rec_loss_per_epoch))
        ae_prior_loss_per_epoch = np.mean(np.array(ae_prior_loss_per_epoch))
        disc_prior_loss_per_epoch = np.mean(np.array(disc_prior_loss_per_epoch))
        
        ae_test_loss_per_epoch = []
        
        for test_batch, _ in test_dataloader:
            test_batch = test_batch.to(device)
            
            _ae_loss, _, _ = ae_test_step(train_batch)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            ae_test_loss_per_epoch.append(_ae_loss)
        
        ae_test_loss_per_epoch = np.mean(np.array(ae_test_loss_per_epoch))
        
        if epoch % weight_save_interval == 0 and save_weights == True:
            torch.save(disc_prior.state_dict(), "results/weights/disc_prior_weights_epoch_{}".format(epoch))
            torch.save(encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epoch))
            torch.save(decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epoch))
        
        plot_ae_outputs(encoder, decoder, epoch)
        
        loss_history["ae train"].append(ae_train_loss_per_epoch)
        loss_history["ae test"].append(ae_test_loss_per_epoch)
        loss_history["ae rec"].append(ae_rec_loss_per_epoch)
        loss_history["ae prior"].append(ae_prior_loss_per_epoch)
        loss_history["disc prior"].append(disc_prior_loss_per_epoch)
        
        print ('epoch {} : ae train: {:01.4f} ae test: {:01.4f} disc prior {:01.4f} rec {:01.4f} prior {:01.4f} time {:01.2f}'.format(epoch + 1, ae_train_loss_per_epoch, ae_test_loss_per_epoch, disc_prior_loss_per_epoch, ae_rec_loss_per_epoch, ae_prior_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_dataloader, test_dataloader, epochs)

epochs = 100

# outer loop over the training epochs
for epoch in range(epochs):
    
    disc_prior_epoch_loss = 0
    autoencoder_epoch_loss = 0
    
    tick = time.time()
    
    for batch_features, _ in train_dataloader:
        
        batch_features = batch_features.to(device)
        
        # disc prior train step
        disc_prior_optimizer.zero_grad()
        
        with torch.no_grad():
            fake_output = encoder(batch_features) 
        real_output = sample_normal(fake_output.shape)
    
        disc_prior_real_output =  disc_prior(real_output)
        disc_prior_fake_output =  disc_prior(fake_output)   
        
        _disc_prior_loss = disc_prior_loss(disc_prior_real_output, disc_prior_fake_output)
        
        _disc_prior_loss.backward()
        disc_prior_optimizer.step()
        
        disc_prior_epoch_loss += _disc_prior_loss.item()

        # autoencoder train step
        ae_optimizer.zero_grad()

        encoded_images = encoder(batch_features)
        reconstructed_images = decoder(encoded_images)

        disc_prior_fake_output = disc_prior(encoded_images)
            
        _autoencoder_loss, _, _ = ae_loss(batch_features, reconstructed_images, disc_prior_fake_output)
        
        _autoencoder_loss.backward()
        ae_optimizer.step()
        
        autoencoder_epoch_loss += _autoencoder_loss.item()
    
    # compute the epoch training loss
    disc_prior_epoch_loss = disc_prior_epoch_loss / len(train_dataloader)
    autoencoder_epoch_loss = autoencoder_epoch_loss / len(train_dataloader)
    
    tock = time.time()

    
    # display the epoch training loss
    plot_ae_outputs(encoder,decoder, epoch, n=5)
    print("epoch : {}/{}, dp_loss = {:.6f} ae_loss = {:.6f}, time = {:.2f}".format(epoch + 1, epochs, disc_prior_epoch_loss, autoencoder_epoch_loss, (tock - tick)))

"""
def plot_ae_outputs2(encoder, decoder, epoch, n=5):
    
    encoder.eval()
    decoder.eval()
    
    plt.figure(figsize=(10,4.5))
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = train_dataset[i][0].unsqueeze(0).to(device)

      with torch.no_grad():
         rec_img  = decoder(encoder(img))    
      
      img = img.cpu().squeeze().numpy()
      img = np.clip(img, 0.0, 1.0)
      img = np.moveaxis(img, 0, 2)
      
      rec_img = rec_img.cpu().squeeze().numpy()
      rec_img = np.clip(rec_img, 0.0, 1.0)
      rec_img = np.moveaxis(rec_img, 0, 2)
      
      plt.imshow(img)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img)  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title("Epoch {}: Reconstructed images".format(epoch))
    plt.show()
    #plt.savefig("epoch_{0:05d}.jpg".format(epoch))
    plt.close()
    
    decoder.train()
    decoder.train()
    

plot_ae_outputs2(encoder, decoder, 200)
""""""