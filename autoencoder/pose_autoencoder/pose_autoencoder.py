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
mocap_data_path = "../../../../Data/Mocap/Muriel_Nov_2021/MUR_Fluidity_Body_Take1_mb_proc_rh.p"
mocap_valid_frame_ranges = [ [ 500, 6500 ] ]
mocap_fps = 50

# model settings
latent_dim = 8
ae_dense_layer_sizes = [ 64, 16 ]
prior_crit_dense_layer_sizes = [ 32, 32 ]

save_models = False
save_tscript = False
save_weights = False

# load model weights
load_weights = True
disc_prior_weights_file = "results/weights/disc_prior_weights_epoch_400"
encoder_weights_file = "results/weights/encoder_weights_epoch_400"
decoder_weights_file = "results/weights/decoder_weights_epoch_400"

# training settings
batch_size = 16
train_percentage = 0.8 # train / test split
test_percentage  = 0.2
dp_learning_rate = 5e-4
ae_learning_rate = 1e-4
ae_norm_loss_scale = 0.1
ae_pos_loss_scale = 0.1
ae_quat_loss_scale = 1.0
ae_prior_loss_scale = 0.01 # weight for prior distribution loss
epochs = 400
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

# gather poses
subject = "S1"
action = "A1"
pose_sequence = mocap_data[subject][action]["rotations"]

poses = []
for valid_frame_range in mocap_valid_frame_ranges:
    frame_range_start = valid_frame_range[0]
    frame_range_end = valid_frame_range[1]
    poses += [pose_sequence[frame_range_start:frame_range_end]]
poses = np.concatenate(poses, axis=0)

pose_count = poses.shape[0]
joint_count = poses.shape[1]
joint_dim = poses.shape[2]
pose_dim = joint_count * joint_dim

poses = np.reshape(poses, (-1, pose_dim))

# create dataset

class PoseDataset(Dataset):
    def __init__(self, poses):
        self.poses = poses
    
    def __len__(self):
        return self.poses.shape[0]
    
    def __getitem__(self, idx):
        return self.poses[idx, ...]

full_dataset = PoseDataset(poses)
dataset_size = len(full_dataset)

test_size = int(test_percentage * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# create models

# create discriminator model for prior distribution

class DiscriminatorPrior(nn.Module):
    def __init__(self, latent_dim, dense_layer_sizes):
        super(DiscriminatorPrior, self).__init__()
        
        self.latent_dim = latent_dim
        self.dense_layer_sizes = dense_layer_sizes
        
        dense_layers = []
        
        dense_layer_count = len(self.dense_layer_sizes)
        
        dense_layers.append(("disc_prior_dense_0", nn.Linear(latent_dim, dense_layer_sizes[0])))
        dense_layers.append(("disc_prior_elu_0", nn.ELU()))
        
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("disc_prior_dense_{}".format(layer_index), nn.Linear(dense_layer_sizes[layer_index - 1], dense_layer_sizes[layer_index])))
            dense_layers.append(("disc_prior_elu_{}".format(layer_index), nn.ELU()))
        
        dense_layers.append(("disc_prior_dense_{}".format(dense_layer_count), nn.Linear(prior_crit_dense_layer_sizes[-1], 1)))
        dense_layers.append(("disc_prior_sigmoid_{}".format(dense_layer_count), nn.Sigmoid()))
        
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
    
    def forward(self, x):
        yhat = self.dense_layers(x)
        return yhat
        
disc_prior = DiscriminatorPrior(latent_dim, prior_crit_dense_layer_sizes).to(device)

print(disc_prior)

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
    def __init__(self, pose_dim, latent_dim, dense_layer_sizes):
        super(Encoder, self).__init__()
        
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.dense_layer_sizes = dense_layer_sizes
    
        # create dense layers
        
        dense_layers = []
        
        dense_layers.append(("encoder_dense_0", nn.Linear(self.pose_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("encoder_dense_relu_0", nn.ReLU()))
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("encoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("encoder_dense_relu_{}".format(layer_index), nn.ReLU()))

        dense_layers.append(("encoder_dense_{}".format(len(self.dense_layer_sizes)), nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)))
        dense_layers.append(("encoder_dense_relu_{}".format(len(self.dense_layer_sizes)), nn.ReLU()))
        
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
    def forward(self, x):
        
        #print("x 1 ", x.shape)
        
        yhat = self.dense_layers(x)
        
        #print("yhat ", yhat.shape)
 
        return yhat
        
encoder = Encoder(pose_dim, latent_dim, ae_dense_layer_sizes).to(device)

print(encoder)

if save_models == True:
    disc_prior.eval()
    
    # save using pickle
    torch.save(encoder, "results/models/encoder.pth")
    
    # save using onnx
    x = torch.zeros((1, pose_dim)).to(device)
    torch.onnx.export(encoder, x, "results/models/encoder.onnx")
    
    disc_prior.train()

if save_tscript == True:
    encoder.eval()
    
    # save using TochScript
    x = torch.rand((1, pose_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(encoder, x)
    script_module.save("results/models/encoder.pt")
    
    encoder.train()

if load_weights and encoder_weights_file:
    encoder.load_state_dict(torch.load(encoder_weights_file))
    
# create decoder model

class Decoder(nn.Module):
    def __init__(self, pose_dim, latent_dim, dense_layer_sizes):
        super(Decoder, self).__init__()
        
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.dense_layer_sizes = dense_layer_sizes

        # create dense layers
        dense_layers = []
        
        dense_layers.append(("decoder_dense_0", nn.Linear(latent_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("decoder_relu_0", nn.ReLU()))

        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("decoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("decoder_dense_relu_{}".format(layer_index), nn.ReLU()))
            
        dense_layers.append(("encoder_dense_{}".format(len(self.dense_layer_sizes)), nn.Linear(self.dense_layer_sizes[-1], self.pose_dim)))
 
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
    def forward(self, x):
        #print("x 1 ", x.size())
        
        # dense layers
        yhat = self.dense_layers(x)
        #print("yhat  ", yhat.size())
        

        return yhat
    
ae_dense_layer_sizes_reversed = ae_dense_layer_sizes.copy()
ae_dense_layer_sizes_reversed.reverse()

decoder = Decoder(pose_dim, latent_dim, ae_dense_layer_sizes_reversed).to(device)

print(decoder)

if save_models == True:
    disc_prior.eval()
    
    # save using pickle
    torch.save(decoder, "results/models/decoder.pth")
    
    # save using onnx
    x = torch.zeros((1, latent_dim)).to(device)
    torch.onnx.export(decoder, x, "results/models/decoder.onnx")
    
    disc_prior.train()

if save_tscript == True:
    decoder.eval()
    
    # save using TochScript
    x = torch.rand((1, latent_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(decoder, x)
    script_module.save("results/models/decoder.pt")
    
    decoder.train()

if load_weights and decoder_weights_file:
    decoder.load_state_dict(torch.load(decoder_weights_file))
    
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

# define AE Loss Functions

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
    _y_rot = y.view((y.shape[0], 1, -1, 4))
    _yhat_rot = _yhat.view((y.shape[0], 1, -1, 4))

    zero_trajectory = torch.zeros((y.shape[0], 1, 3), dtype=torch.float32, requires_grad=True).to(device)

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
            disc_prior.save_weights("disc_prior_weights epoch_{}".format(epoch))
            encoder.save_weights("ae_encoder_weights epoch_{}".format(epoch))
            decoder.save_weights("ae_decoder_weights epoch_{}".format(epoch))
        
        """
        if epoch % vis_save_interval == 0 and save_vis == True:
            create_epoch_visualisations(epoch)
        """
        
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

def create_ref_pose_image(pose_index, file_name):
    pose = poses[pose_index]
    pose = torch.tensor(np.reshape(pose, (1, 1, joint_count, joint_dim))).to(device)
    zero_trajectory = torch.tensor(np.zeros((1, 1, 3), dtype=np.float32)).to(device)
    skel_pose = skeleton.forward_kinematics(pose, zero_trajectory)
    skel_pose = skel_pose.detach().cpu().numpy()
    skel_pose = np.reshape(skel_pose, (joint_count, 3))
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_pose)
    pose_image = poseRenderer.create_pose_image(skel_pose, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    pose_image.save(file_name, optimize=False)

def create_rec_pose_image(pose_index, file_name):
    encoder.eval()
    decoder.eval()
    
    pose = poses[pose_index]
    pose = torch.tensor(np.expand_dims(pose, axis=0)).to(device)
    
    with torch.no_grad():
        pose_enc = encoder(pose)
        rec_pose = decoder(pose_enc)
        
    rec_pose = torch.squeeze(rec_pose)
    rec_pose = rec_pose.view((-1, 4))
    rec_pose = nn.functional.normalize(rec_pose, p=2, dim=1)
    rec_pose = rec_pose.view((1, 1, joint_count, joint_dim))

    zero_trajectory = torch.tensor(np.zeros((1, 1, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)

    skel_pose = skeleton.forward_kinematics(rec_pose, zero_trajectory)

    skel_pose = skel_pose.detach().cpu().numpy()
    skel_pose = np.squeeze(skel_pose)    

    view_min, view_max = utils.get_equal_mix_max_positions(skel_pose)
    pose_image = poseRenderer.create_pose_image(skel_pose, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    pose_image.save(file_name, optimize=False)
    
    encoder.train()
    decoder.train()
    
def encode_poses(pose_indices):
    
    encoder.eval()
    
    pose_encodings = []
    
    for pose_index in pose_indices:
        pose = poses[pose_index]
        pose = np.expand_dims(pose, axis=0)
        pose = torch.from_numpy(pose).to(device)
        
        with torch.no_grad():
            pose_enc = encoder(pose)
            
        pose_enc = torch.squeeze(pose_enc)
        pose_enc = pose_enc.detach().cpu().numpy()

        pose_encodings.append(pose_enc)
        
    encoder.train()
        
    return pose_encodings

def decode_pose_encodings(pose_encodings, file_name):
    
    decoder.eval()
    
    rec_poses = []
    
    for pose_encoding in pose_encodings:
        pose_encoding = np.expand_dims(pose_encoding, axis=0)
        pose_encoding = torch.from_numpy(pose_encoding).to(device)

        with torch.no_grad():
            rec_pose = decoder(pose_encoding)
            
        rec_pose = torch.squeeze(rec_pose)
        rec_pose = rec_pose.view((-1, 4))
        rec_pose = nn.functional.normalize(rec_pose, p=2, dim=1)
        rec_pose = rec_pose.view((1, joint_count, joint_dim))

        rec_poses.append(rec_pose)
    
    rec_poses = torch.cat(rec_poses, dim=0)
    rec_poses = torch.unsqueeze(rec_poses, dim=0)

    zero_trajectory = torch.tensor(np.zeros((1, len(pose_encodings), 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    skel_poses = skeleton.forward_kinematics(rec_poses, zero_trajectory)
    
    skel_poses = skel_poses.detach().cpu().numpy()
    skel_poses = np.squeeze(skel_poses)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_poses)
    pose_images = poseRenderer.create_pose_images(skel_poses, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    pose_images[0].save(file_name, save_all=True, append_images=pose_images[1:], optimize=False, duration=33.0, loop=0) 
    
    decoder.train()
    
# create single original pose

pose_index = 100

create_ref_pose_image(pose_index, "results/images/orig_pose_{}.gif".format(pose_index))

# recontruct single pose

pose_index = 100

create_rec_pose_image(pose_index, "results/images/rec_pose_{}.gif".format(pose_index))

# reconstruct original pose sequence

start_pose_index = 100
end_pose_index = 500
pose_indices = [ pose_index for pose_index in range(start_pose_index, end_pose_index)]

pose_encodings = encode_poses(pose_indices)
decode_pose_encodings(pose_encodings, "results/images/rec_pose_sequence_{}-{}.gif".format(start_pose_index, end_pose_index))

# random walk

start_pose_index = 100
pose_count = 500

pose_indices = [start_pose_index]

pose_encodings = encode_poses(pose_indices)

for index in range(0, pose_count - 1):
    random_step = np.random.random((latent_dim)).astype(np.float32) * 2.0
    pose_encodings.append(pose_encodings[index] + random_step)

decode_pose_encodings(pose_encodings, "results/images/rec_poses_randwalk_{}_{}.gif".format(start_pose_index, pose_count))

# pose sequence offset following

start_pose_index = 100
end_pose_index = 500
    
pose_indices = [ pose_index for pose_index in range(start_pose_index, end_pose_index)]

pose_encodings = encode_poses(pose_indices)

offset_pose_encodings = []

for index in range(len(pose_encodings)):
    sin_value = np.sin(index / (len(pose_encodings) - 1) * np.pi * 4.0)
    offset = np.ones(shape=(latent_dim), dtype=np.float32) * sin_value * 4.0
    offset_pose_encoding = pose_encodings[index] + offset
    offset_pose_encodings.append(offset_pose_encoding)
    
decode_pose_encodings(offset_pose_encodings, "results/images/rec_pose_sequence_offset_{}-{}.gif".format(start_pose_index, end_pose_index))

# interpolate two original pose sequences

start_pose1_index = 100
end_pose1_index = 500

start_pose2_index = 1100
end_pose2_index = 1500

pose1_indices = [ pose_index for pose_index in range(start_pose1_index, end_pose1_index)]
pose2_indices = [ pose_index for pose_index in range(start_pose2_index, end_pose2_index)]

pose1_encodings = encode_poses(pose1_indices)
pose2_encodings = encode_poses(pose2_indices)

mixed_pose_encodings = []

for index in range(len(pose1_indices)):
    mix_factor = index / (len(pose1_indices) - 1)
    mixed_pose_encoding = pose1_encodings[index] * (1.0 - mix_factor) + pose2_encodings[index] * mix_factor
    mixed_pose_encodings.append(mixed_pose_encoding)

decode_pose_encodings(mixed_pose_encodings, "results/images/rec_pose_sequence_mix_{}-{}_{}-{}.gif".format(start_pose1_index, end_pose2_index, start_pose2_index, end_pose2_index))


