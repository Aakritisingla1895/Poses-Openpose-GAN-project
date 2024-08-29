import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict

import os, sys, time, subprocess
import numpy as np
sys.path.append("../..")

from common import utils
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.quaternion import qmul, qnormalize_np, slerp
from common.pose_renderer import PoseRenderer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# mocap settings
mocap_data_path = "../../../../Data/Mocap/Muriel_Nov_2021/MUR_PolytopiaMovement_Take2_mb_proc_rh.p"
mocap_valid_frame_ranges = [ [ 860, 9500 ] ]
mocap_fps = 50

# model settings
latent_dim = 64
sequence_length = 128
ae_rnn_layer_count = 2
ae_rnn_layer_size = 512
ae_dense_layer_sizes = [ 512 ]
prior_crit_dense_layer_sizes = [ 512, 512 ]

save_models = False
save_tscript = False
save_weights = False

# load model weights
load_weights = False
disc_prior_weights_file = "results/weights/disc_prior_weights_epoch_400"
encoder_weights_file = "results/weights/encoder_weights_epoch_400"
decoder_weights_file = "results/weights/decoder_weights_epoch_400"

# training settings
sequence_offset = 2 # when creating sequence excerpts, each excerpt is offset from the previous one by this value
batch_size = 16
train_percentage = 0.8 # train / test split
test_percentage  = 0.2
dp_learning_rate = 5e-4
ae_learning_rate = 1e-4
ae_norm_loss_scale = 0.1
ae_pos_loss_scale = 0.1
ae_quat_loss_scale = 1.0
ae_prior_loss_scale = 0.01 # weight for prior distribution loss
epochs = 10
model_save_interval = 100
save_history = False

# visualization settings
view_ele = 0.0
view_azi = 0.0
view_line_width = 4.0
view_size = 8.0

# load mocap data
mocap_data = MocapDataset(mocap_data_path, fps=mocap_fps)
if device == 'cuda':
    mocap_data.cuda()
mocap_data.compute_positions()

# gather skeleton info
skeleton = mocap_data.skeleton()
skeleton_joint_count = skeleton.num_joints()
skel_edge_list = utils.get_skeleton_edge_list(skeleton)

# obtain pose sequence
subject = "S1"
action = "A1"
pose_sequence = mocap_data[subject][action]["rotations"]

pose_sequence_length = pose_sequence.shape[0]
joint_count = pose_sequence.shape[1]
joint_dim = pose_sequence.shape[2]
pose_dim = joint_count * joint_dim
pose_sequence = np.reshape(pose_sequence, (-1, pose_dim))

# gather pose sequence excerpts
pose_sequence_excerpts = []

for valid_frame_range in mocap_valid_frame_ranges:
    frame_range_start = valid_frame_range[0]
    frame_range_end = valid_frame_range[1]
    
    for seq_excerpt_start in np.arange(frame_range_start, frame_range_end - sequence_length, sequence_offset):
        #print("valid: start ", frame_range_start, " end ", frame_range_end, " exc: start ", seq_excerpt_start, " end ", (seq_excerpt_start + sequence_length) )
        pose_sequence_excerpt =  pose_sequence[seq_excerpt_start:seq_excerpt_start + sequence_length]
        pose_sequence_excerpts.append(pose_sequence_excerpt)
        
pose_sequence_excerpts = np.array(pose_sequence_excerpts)

# create dataset

sequence_excerpts_count = pose_sequence_excerpts.shape[0]

class SequenceDataset(Dataset):
    def __init__(self, sequence_excerpts):
        self.sequence_excerpts = sequence_excerpts
    
    def __len__(self):
        return self.sequence_excerpts.shape[0]
    
    def __getitem__(self, idx):
        return self.sequence_excerpts[idx, ...]
        

full_dataset = SequenceDataset(pose_sequence_excerpts)
dataset_size = len(full_dataset)

test_size = int(test_percentage * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# create models

# create discriminator model for prior distribution

class DiscriminatorPrior(nn.Module):
    def __init__(self, latent_dim, prior_crit_dense_layer_sizes):
        super(DiscriminatorPrior, self).__init__()
        
        self.latent_dim = latent_dim
        self.prior_crit_dense_layer_sizes = prior_crit_dense_layer_sizes
        
        dense_layers = []
        dense_layers.append(("disc_prior_dense_0", nn.Linear(latent_dim, prior_crit_dense_layer_sizes[0])))
        dense_layers.append(("disc_prior_elu_0", nn.ELU()))
        
        dense_layer_count = len(prior_crit_dense_layer_sizes)

        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("disc_prior_dense_{}".format(layer_index), nn.Linear(prior_crit_dense_layer_sizes[layer_index - 1], prior_crit_dense_layer_sizes[layer_index])))
            dense_layers.append(("disc_prior_elu_{}".format(layer_index), nn.ELU()))
    
        dense_layers.append(("disc_prior_dense_{}".format(dense_layer_count), nn.Linear(prior_crit_dense_layer_sizes[-1], 1)))
        dense_layers.append(("disc_prior_sigmoid_{}".format(dense_layer_count), nn.Sigmoid()))
        
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
    
    def forward(self, x):
        yhat = self.dense_layers(x)
        return yhat
        
disc_prior = DiscriminatorPrior(latent_dim, prior_crit_dense_layer_sizes).to(device)

print(disc_prior)

"""
for name, param in discriminator_prior.named_parameters():
    print(f"Layer: {name} | Size: {param.size()}")
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

# create encoder model

class Encoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(Encoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_count = rnn_layer_count
        self.rnn_layer_size = rnn_layer_size 
        self.dense_layer_sizes = dense_layer_sizes
    
        # create recurrent layers
        rnn_layers = []
        rnn_layers.append(("encoder_rnn_0", nn.LSTM(self.pose_dim, self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # create dense layers
        
        dense_layers = []
        
        dense_layers.append(("encoder_dense_0", nn.Linear(self.rnn_layer_size, self.dense_layer_sizes[0])))
        dense_layers.append(("encoder_dense_relu_0", nn.ReLU()))
        
        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("encoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("encoder_dense_relu_{}".format(layer_index), nn.ReLU()))

        dense_layers.append(("encoder_dense_{}".format(len(self.dense_layer_sizes)), nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)))
        dense_layers.append(("encoder_dense_relu_{}".format(len(self.dense_layer_sizes)), nn.ReLU()))
        
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
    def forward(self, x):
        
        #print("x 1 ", x.shape)
        
        x, (_, _) = self.rnn_layers(x)
        
        #print("x 2 ", x.shape)
        
        x = x[:, -1, :] # only last time step 
        
        #print("x 3 ", x.shape)
        
        yhat = self.dense_layers(x)
        
        #print("yhat ", yhat.shape)
 
        return yhat
    
encoder = Encoder(sequence_length, pose_dim, latent_dim, ae_rnn_layer_count, ae_rnn_layer_size, ae_dense_layer_sizes).to(device)

print(encoder)

if save_models == True:
    encoder.train()
    
    # save using pickle
    torch.save(encoder, "results/models/encoder.pth")
    
    # save using onnx
    x = torch.zeros((1, sequence_length, pose_dim)).to(device)
    torch.onnx.export(encoder, x, "results/models/encoder.onnx")
    
    encoder.test()

if save_tscript == True:
    encoder.train()
    
    # save using TochScript
    x = torch.rand((1, sequence_length, pose_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(encoder, x)
    script_module.save("results/models/encoder.pt")
    
    encoder.test()

if load_weights and encoder_weights_file:
    encoder.load_state_dict(torch.load(encoder_weights_file, map_location=device))
    
    
# create decoder model

class Decoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(Decoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_count = rnn_layer_count
        self.dense_layer_sizes = dense_layer_sizes

        # create dense layers
        dense_layers = []
        
        dense_layers.append(("decoder_dense_0", nn.Linear(latent_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("decoder_relu_0", nn.ReLU()))

        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("decoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("decoder_dense_relu_{}".format(layer_index), nn.ReLU()))
 
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        # create rnn layers
        rnn_layers = []

        rnn_layers.append(("decoder_rnn_0", nn.LSTM(self.dense_layer_sizes[-1], self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # final output dense layer
        final_layers = []
        
        final_layers.append(("decoder_dense_{}".format(dense_layer_count), nn.Linear(self.rnn_layer_size, self.pose_dim)))
        
        self.final_layers = nn.Sequential(OrderedDict(final_layers))
        
    def forward(self, x):
        #print("x 1 ", x.size())
        
        # dense layers
        x = self.dense_layers(x)
        #print("x 2 ", x.size())
        
        # repeat vector
        x = torch.unsqueeze(x, dim=1)
        x = x.repeat(1, sequence_length, 1)
        #print("x 3 ", x.size())
        
        # rnn layers
        x, (_, _) = self.rnn_layers(x)
        #print("x 4 ", x.size())
        
        # final time distributed dense layer
        x_reshaped = x.contiguous().view(-1, self.rnn_layer_size)  # (batch_size * sequence, input_size)
        #print("x 5 ", x_reshaped.size())
        
        yhat = self.final_layers(x_reshaped)
        #print("yhat 1 ", yhat.size())
        
        yhat = yhat.contiguous().view(-1, self.sequence_length, self.pose_dim)
        #print("yhat 2 ", yhat.size())

        return yhat

ae_dense_layer_sizes_reversed = ae_dense_layer_sizes.copy()
ae_dense_layer_sizes_reversed.reverse()

decoder = Decoder(sequence_length, pose_dim, latent_dim, ae_rnn_layer_count, ae_rnn_layer_size, ae_dense_layer_sizes_reversed).to(device)

print(decoder)

if save_models == True:
    decoder.eval()
    
    # save using pickle
    torch.save(decoder, "results/models/decoder_weights.pth")
    
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
    decoder.load_state_dict(torch.load(decoder_weights_file, map_location=device))
    
# Training

disc_optimizer = torch.optim.Adam(disc_prior.parameters(), lr=dp_learning_rate)
ae_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=ae_learning_rate)

cross_entropy = nn.BCELoss()

# function returning normal distributed random data 
# serves as reference for the discriminator to distinguish the encoders prior from
def sample_normal(shape):
    return torch.tensor(np.random.normal(size=shape), dtype=torch.float32).to(device)

# discriminator prior loss function
def disc_prior_loss(disc_real_output, disc_fake_output):
    ones = torch.ones_like(disc_real_output).to(device)
    zeros = torch.zeros_like(disc_fake_output).to(device)

    real_loss = cross_entropy(disc_real_output, ones)
    fake_loss = cross_entropy(disc_fake_output, zeros)

    total_loss = (real_loss + fake_loss) * 0.5
    return total_loss

def ae_norm_loss(yhat):
    
    _yhat = yhat.view(-1, 4)
    _norm = torch.norm(_yhat, dim=1)
    _diff = (_norm - 1.0) ** 2
    _loss = torch.mean(_diff)
    return _loss

def ae_pos_loss(y, yhat):
    # y and yhat shapes: batch_size, seq_length, pose_dim

    # normalize tensors
    _yhat = yhat.view(-1, 4)

    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    _y_rot = y.view((y.shape[0], y.shape[1], -1, 4))
    _yhat_rot = _yhat.view((y.shape[0], y.shape[1], -1, 4))

    zero_trajectory = torch.zeros((y.shape[0], y.shape[1], 3), dtype=torch.float32, requires_grad=True).to(device)

    _y_pos = skeleton.forward_kinematics(_y_rot, zero_trajectory)
    _yhat_pos = skeleton.forward_kinematics(_yhat_rot, zero_trajectory)

    _pos_diff = torch.norm((_y_pos - _yhat_pos), dim=3)
    
    _loss = torch.mean(_pos_diff)

    return _loss

def ae_quat_loss(y, yhat):
    # y and yhat shapes: batch_size, seq_length, pose_dim
    
    # normalize quaternion
    
    _y = y.view((-1, 4))
    _yhat = yhat.view((-1, 4))

    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    
    # inverse of quaternion: https://www.mathworks.com/help/aeroblks/quaternioninverse.html
    _yhat_inv = _yhat_norm * torch.tensor([[1.0, -1.0, -1.0, -1.0]], dtype=torch.float32).to(device)

    # calculate difference quaternion
    _diff = qmul(_yhat_inv, _y)
    # length of complex part
    _len = torch.norm(_diff[:, 1:], dim=1)
    # atan2
    _atan = torch.atan2(_len, _diff[:, 0])
    # abs
    _abs = torch.abs(_atan)
    _loss = torch.mean(_abs)   
    return _loss

# autoencoder loss function
def ae_loss(y, yhat, disc_fake_output):
    # function parameters
    # y: encoder input
    # yhat: decoder output (i.e. reconstructed encoder input)
    # disc_fake_output: discriminator output for encoder generated prior
    
    _norm_loss = ae_norm_loss(yhat)
    _pos_loss = ae_pos_loss(y, yhat)
    _quat_loss = ae_quat_loss(y, yhat)


    # discrimination loss
    _fake_loss = cross_entropy(torch.zeros_like(disc_fake_output), disc_fake_output)
    
    _total_loss = 0.0
    _total_loss += _norm_loss * ae_norm_loss_scale
    _total_loss += _pos_loss * ae_pos_loss_scale
    _total_loss += _quat_loss * ae_quat_loss_scale
    _total_loss += _fake_loss * ae_prior_loss_scale
    
    return _total_loss, _norm_loss, _pos_loss, _quat_loss, _fake_loss

def disc_prior_train_step(target_poses):
    # have normal distribution and encoder produce real and fake outputs, respectively
    
    with torch.no_grad():
        encoder_output = encoder(target_poses)
        
    real_output = sample_normal(encoder_output.shape)
    
    # let discriminator distinguish between real and fake outputs
    disc_real_output =  disc_prior(real_output)
    disc_fake_output =  disc_prior(encoder_output)   
    _disc_loss = disc_prior_loss(disc_real_output, disc_fake_output)
    
    # Backpropagation
    disc_optimizer.zero_grad()
    _disc_loss.backward()
    disc_optimizer.step()
    
    return _disc_loss

def ae_train_step(target_poses):
    
    #print("train step target_poses ", target_poses.shape)
 
    # let autoencoder preproduce target_poses (decoder output) and also return encoder output
    encoder_output = encoder(target_poses)
    pred_poses = decoder(encoder_output)
    
    # let discriminator output its fake assessment of the encoder ouput
    with torch.no_grad():
        disc_fake_output = disc_prior(encoder_output)
    
    _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_prior_loss = ae_loss(target_poses, pred_poses, disc_fake_output) 

    #print("_ae_pos_loss ", _ae_pos_loss)
    
    # Backpropagation
    ae_optimizer.zero_grad()
    _ae_loss.backward()
    ae_optimizer.step()
    
    return _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_prior_loss

def ae_test_step(target_poses):
    with torch.no_grad():
        # let autoencoder preproduce target_poses (decoder output) and also return encoder output
        encoder_output = encoder(target_poses)
        pred_poses = decoder(encoder_output)
        
        # let discriminator output its fake assessment of the encoder ouput
        disc_fake_output =  disc_prior(encoder_output)
    
        _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_prior_loss = ae_loss(target_poses, pred_poses, disc_fake_output) 
    
    return _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_prior_loss

def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["ae train"] = []
    loss_history["ae test"] = []
    loss_history["ae norm"] = []
    loss_history["ae pos"] = []
    loss_history["ae quat"] = []
    loss_history["ae prior"] = []
    loss_history["disc prior"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        ae_train_loss_per_epoch = []
        ae_norm_loss_per_epoch = []
        ae_pos_loss_per_epoch = []
        ae_quat_loss_per_epoch = []
        ae_prior_loss_per_epoch = []
        disc_prior_loss_per_epoch = []
        
        for train_batch in train_dataloader:
            train_batch = train_batch.to(device)

            # start with discriminator training
            _disc_prior_train_loss = disc_prior_train_step(train_batch)
            
            _disc_prior_train_loss = _disc_prior_train_loss.detach().cpu().numpy()
            
            #print("_disc_prior_train_loss ", _disc_prior_train_loss)
            
            disc_prior_loss_per_epoch.append(_disc_prior_train_loss)
            
            # now train the autoencoder
            _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_prior_loss = ae_train_step(train_batch)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            _ae_norm_loss = _ae_norm_loss.detach().cpu().numpy()
            _ae_pos_loss = _ae_pos_loss.detach().cpu().numpy()
            _ae_quat_loss = _ae_quat_loss.detach().cpu().numpy()
            _ae_prior_loss = _ae_prior_loss.detach().cpu().numpy()
            
            #print("_ae_prior_loss ", _ae_prior_loss)
            
            ae_train_loss_per_epoch.append(_ae_loss)
            ae_norm_loss_per_epoch.append(_ae_norm_loss)
            ae_pos_loss_per_epoch.append(_ae_pos_loss)
            ae_quat_loss_per_epoch.append(_ae_quat_loss)
            ae_prior_loss_per_epoch.append(_ae_prior_loss)

        ae_train_loss_per_epoch = np.mean(np.array(ae_train_loss_per_epoch))
        ae_norm_loss_per_epoch = np.mean(np.array(ae_norm_loss_per_epoch))
        ae_pos_loss_per_epoch = np.mean(np.array(ae_pos_loss_per_epoch))
        ae_quat_loss_per_epoch = np.mean(np.array(ae_quat_loss_per_epoch))
        ae_prior_loss_per_epoch = np.mean(np.array(ae_prior_loss_per_epoch))
        disc_prior_loss_per_epoch = np.mean(np.array(disc_prior_loss_per_epoch))

        ae_test_loss_per_epoch = []
        
        for test_batch in test_dataloader:
            test_batch = test_batch.to(device)
            
            _ae_loss, _, _, _, _ = ae_test_step(train_batch)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            ae_test_loss_per_epoch.append(_ae_loss)
        
        ae_test_loss_per_epoch = np.mean(np.array(ae_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(disc_prior.state_dict(), "results/weights/disc_prior_weights_epoch_{}".format(epoch))
            torch.save(encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epoch))
            torch.save(decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epoch))
 
        loss_history["ae train"].append(ae_train_loss_per_epoch)
        loss_history["ae test"].append(ae_test_loss_per_epoch)
        loss_history["ae norm"].append(ae_norm_loss_per_epoch)
        loss_history["ae pos"].append(ae_pos_loss_per_epoch)
        loss_history["ae quat"].append(ae_quat_loss_per_epoch)
        loss_history["ae prior"].append(ae_prior_loss_per_epoch)
        loss_history["disc prior"].append(disc_prior_loss_per_epoch)
        
        print ('epoch {} : ae train: {:01.4f} ae test: {:01.4f} disc prior {:01.4f} norm {:01.4f} pos {:01.4f} quat {:01.4f} prior {:01.4f} time {:01.2f}'.format(epoch + 1, ae_train_loss_per_epoch, ae_test_loss_per_epoch, disc_prior_loss_per_epoch, ae_norm_loss_per_epoch, ae_pos_loss_per_epoch, ae_quat_loss_per_epoch, ae_prior_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_dataloader, test_dataloader, epochs)

# save history
utils.save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
utils.save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

# save model weights
torch.save(disc_prior.state_dict(), "results/weights/disc_prior_weights_epoch_{}".format(epochs))
torch.save(encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epochs))
torch.save(decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epochs))

# inference and rendering 

skel_edge_list = utils.get_skeleton_edge_list(skeleton)
poseRenderer = PoseRenderer(skel_edge_list)

def create_ref_sequence_anim(seq_index, file_name):
    sequence_excerpt = pose_sequence_excerpts[seq_index]
    sequence_excerpt = np.reshape(sequence_excerpt, (sequence_length, joint_count, joint_dim))
    
    sequence_excerpt = torch.tensor(np.expand_dims(sequence_excerpt, axis=0)).to(device)
    zero_trajectory = torch.tensor(np.zeros((1, sequence_length, 3), dtype=np.float32)).to(device)
    
    skel_sequence = skeleton.forward_kinematics(sequence_excerpt, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

def create_rec_sequence_anim(seq_index, file_name):
    sequence_excerpt = pose_sequence_excerpts[seq_index]
    sequence_excerpt = np.expand_dims(sequence_excerpt, axis=0)
    
    sequence_excerpt = torch.from_numpy(sequence_excerpt).to(device)
    
    with torch.no_grad():
        sequence_enc = encoder(sequence_excerpt)
        pred_sequence = decoder(sequence_enc)
        
    pred_sequence = torch.squeeze(pred_sequence)
    pred_sequence = pred_sequence.view((-1, 4))
    pred_sequence = nn.functional.normalize(pred_sequence, p=2, dim=1)
    pred_sequence = pred_sequence.view((1, sequence_length, joint_count, joint_dim))

    zero_trajectory = torch.tensor(np.zeros((1, sequence_length, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)

    skel_sequence = skeleton.forward_kinematics(pred_sequence, zero_trajectory)

    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)    

    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

def encode_sequences(frame_indices):
    
    encoder.eval()
    
    latent_vectors = []
    
    seq_excerpt_count = len(frame_indices)

    for excerpt_index in range(seq_excerpt_count):
        excerpt_start_frame = frame_indices[excerpt_index]
        excerpt_end_frame = excerpt_start_frame + sequence_length
        excerpt = pose_sequence[excerpt_start_frame:excerpt_end_frame]
        excerpt = np.expand_dims(excerpt, axis=0)
        excerpt = torch.from_numpy(excerpt).to(device)
        
        with torch.no_grad():
            latent_vector = encoder(excerpt)
            
        latent_vector = torch.squeeze(latent_vector)
        latent_vector = latent_vector.detach().cpu().numpy()

        latent_vectors.append(latent_vector)
        
    encoder.train()
        
    return latent_vectors

def decode_sequence_encodings(sequence_encodings, file_name):
    
    decoder.eval()
    
    rec_sequences = []
    
    for seq_encoding in sequence_encodings:
        seq_encoding = np.expand_dims(seq_encoding, axis=0)
        seq_encoding = torch.from_numpy(seq_encoding).to(device)

        with torch.no_grad():
            rec_seq = decoder(seq_encoding)
            
        rec_seq = torch.squeeze(rec_seq)
        rec_seq = rec_seq.view((-1, 4))
        rec_seq = nn.functional.normalize(rec_seq, p=2, dim=1)
        rec_seq = rec_seq.view((-1, joint_count, joint_dim))

        rec_sequences.append(rec_seq)
    
    rec_sequences = torch.cat(rec_sequences, dim=0)
    rec_sequences = torch.unsqueeze(rec_sequences, dim=0)
    
    print("rec_sequences s ", rec_sequences.shape)

    zero_trajectory = torch.tensor(np.zeros((1, len(sequence_encodings) * sequence_length, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    skel_sequence = skeleton.forward_kinematics(rec_sequences, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

    decoder.train()
    
# create original sequence

seq_index = 100

create_ref_sequence_anim(seq_index, "results/anims/orig_sequence_{}.gif".format(seq_index))

# recontruct original sequence

seq_index = 100

create_rec_sequence_anim(seq_index, "results/anims/rec_sequence_{}.gif".format(seq_index))

# reconstruct original pose sequences

start_seq_index = 1000
end_seq_index = 2000
seq_indices = [ seq_index for seq_index in range(start_seq_index, end_seq_index, sequence_length)]

seq_encodings = encode_sequences(seq_indices)
decode_sequence_encodings(seq_encodings, "results/anims/rec_sequences_{}-{}.gif".format(start_seq_index, end_seq_index))
