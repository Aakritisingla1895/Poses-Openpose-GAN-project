"""
simple sequence generation example
network input and output are joint rotations (quaternions)
two loss functions:
    normalization loss applied to quaternions
    difference loss applied to quaternions
two layer rnn
permanent teacher forcing
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict

import os, sys, time, subprocess
import numpy as np
import math
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
sequence_length = 128
ar_rnn_layer_count = 2
ar_rnn_layer_size = 512
ar_dense_layer_sizes = [ ]

save_models = False
save_tscript = False
save_weights = False

# load model weights
load_weights = False
autoreg_weights_file = "results/weights/autoregr_weights_epoch_400"

# training settings
sequence_offset = 2 # when creating sequence excerpts, each excerpt is offset from the previous one by this value
batch_size = 16
train_percentage = 0.8 # train / test split
test_percentage  = 0.2
ar_learning_rate = 1e-4
ar_norm_loss_scale = 0.1
ar_quat_loss_scale = 0.9



epochs = 100
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

# prepare training data
# split data into input sequence(s) and output pose(s)
input_pose_sequences = []
output_poses = []

for valid_frame_range in mocap_valid_frame_ranges:
    frame_range_start = valid_frame_range[0]
    frame_range_end = valid_frame_range[1]
    
    for seq_excerpt_start in np.arange(frame_range_start, frame_range_end - sequence_length - 1, sequence_offset):
        #print("valid: start ", frame_range_start, " end ", frame_range_end, " exc: start ", seq_excerpt_start, " end ", (seq_excerpt_start + sequence_length) )
        
        input_pose_sequences.append(pose_sequence[seq_excerpt_start:seq_excerpt_start+sequence_length])
        output_poses.append(pose_sequence[seq_excerpt_start+sequence_length:seq_excerpt_start+sequence_length + 1])

input_pose_sequences = np.array(input_pose_sequences)
output_poses = np.array(output_poses)

# create dataset

class SequencePoseDataset(Dataset):
    def __init__(self, input_poses_sequences, output_poses):
        self.input_poses_sequences = input_poses_sequences
        self.output_poses = output_poses
    
    def __len__(self):
        return self.input_poses_sequences.shape[0]
    
    def __getitem__(self, idx):
        return self.input_poses_sequences[idx, ...], self.output_poses[idx, ...]

full_dataset = SequencePoseDataset(input_pose_sequences, output_poses)

dataset_size = len(full_dataset)

test_size = int(test_percentage * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
batch = next(iter(train_dataloader))

batch_mocap = batch[0]
batch_labels = batch[1]

print("batch_mocap s", batch_mocap.shape)
print("batch_labels s", batch_labels.shape)
"""

# create model

class AutoRegressor(nn.Module):
    def __init__(self, pose_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(AutoRegressor, self).__init__()
        
        self.pose_dim = pose_dim
        self.rnn_layer_count = rnn_layer_count
        self.rnn_layer_size = rnn_layer_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create recurrent layers
        rnn_layers = []
        
        rnn_layers.append(("autoreg_rnn_0", nn.LSTM(self.pose_dim, self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # create dense layers
        dense_layers = []
        dense_layer_count = len(self.dense_layer_sizes)
        
        if dense_layer_count > 0:
            dense_layers.append(("autoreg_dense_0", nn.Linear(self.rnn_layer_size, self.dense_layer_sizes[0])))
            dense_layers.append(("autoregr_dense_relu_0", nn.ReLU()))

            for layer_index in range(1, dense_layer_count):
                dense_layers.append(("autoreg_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
                dense_layers.append(("autoregr_dense_relu_{}".format(layer_index), nn.ReLU()))
        
            dense_layers.append(("autoregr_dense_{}".format(len(self.dense_layer_sizes)), nn.Linear(self.dense_layer_sizes[-1], self.pose_dim)))
        else:
            dense_layers.append(("autoreg_dense_0", nn.Linear(self.rnn_layer_size, self.pose_dim)))
        
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
     
autoreg = AutoRegressor(pose_dim, ar_rnn_layer_count, ar_rnn_layer_size, ar_dense_layer_sizes).to(device)

print(autoreg)

"""
test_input = torch.zeros((1, sequence_length, pose_dim)).to(device)
test_output = autoreg(test_input)
"""

if save_models == True:
    autoreg.train()
    
    # save using pickle
    torch.save(autoreg, "results/models/autoreg.pth")
    
    # save using onnx
    x = torch.zeros((1, sequence_length, pose_dim)).to(device)
    torch.onnx.export(autoreg, x, "results/models/autoreg.onnx")
    
    autoreg.test()

if save_tscript == True:
    autoreg.train()
    
    # save using TochScript
    x = torch.rand((1, sequence_length, pose_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(autoreg, x)
    script_module.save("results/models/autoreg.pt")
    
    autoreg.test()

if load_weights and autoreg_weights_file:
    autoreg.load_state_dict(torch.load(autoreg_weights_file, map_location=device))
    
# Training
ar_optimizer = torch.optim.Adam(autoreg.parameters(), lr=ar_learning_rate)

def ar_norm_loss(yhat):
    _yhat = yhat.view(-1, 4)
    _norm = torch.norm(_yhat, dim=1)
    _diff = (_norm - 1.0) ** 2
    _loss = torch.mean(_diff)
    return _loss

def ar_quat_loss(y, yhat):
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
def ar_loss(y, yhat):
    _norm_loss = ar_norm_loss(yhat)
    _quat_loss = ar_quat_loss(y, yhat)
    
    _total_loss = 0.0
    _total_loss += _norm_loss * ar_norm_loss_scale
    _total_loss += _quat_loss * ar_quat_loss_scale
    
    return _total_loss, _norm_loss, _quat_loss

def ar_train_step(pose_sequences, target_poses):

    pred_poses = autoreg(pose_sequences)

    _ar_loss, _ar_norm_loss, _ar_quat_loss = ar_loss(target_poses, pred_poses) 

    #print("_ae_pos_loss ", _ae_pos_loss)
    
    # Backpropagation
    ar_optimizer.zero_grad()
    _ar_loss.backward()

    ar_optimizer.step()
    
    return _ar_loss, _ar_norm_loss, _ar_quat_loss

def ar_test_step(pose_sequences, target_poses):
    
    autoreg.eval()
 
    with torch.no_grad():
        pred_poses = autoreg(pose_sequences)
        _ar_loss, _ar_norm_loss, _ar_quat_loss = ar_loss(target_poses, pred_poses) 
    
    autoreg.train()
    
    return _ar_loss, _ar_norm_loss, _ar_quat_loss

def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["ar train"] = []
    loss_history["ar test"] = []
    loss_history["ar norm"] = []
    loss_history["ar quat"] = []

    for epoch in range(epochs):
        start = time.time()
        
        ar_train_loss_per_epoch = []
        ar_norm_loss_per_epoch = []
        ar_quat_loss_per_epoch = []

        for train_batch in train_dataloader:
            input_pose_sequences = train_batch[0].to(device)
            target_poses = train_batch[1].to(device)
            
            _ar_loss, _ar_norm_loss, _ar_quat_loss = ar_train_step(input_pose_sequences, target_poses)
            
            _ar_loss = _ar_loss.detach().cpu().numpy()
            _ar_norm_loss = _ar_norm_loss.detach().cpu().numpy()
            _ar_quat_loss = _ar_quat_loss.detach().cpu().numpy()
            
            ar_train_loss_per_epoch.append(_ar_loss)
            ar_norm_loss_per_epoch.append(_ar_norm_loss)
            ar_quat_loss_per_epoch.append(_ar_quat_loss)

        ar_train_loss_per_epoch = np.mean(np.array(ar_train_loss_per_epoch))
        ar_norm_loss_per_epoch = np.mean(np.array(ar_norm_loss_per_epoch))
        ar_quat_loss_per_epoch = np.mean(np.array(ar_quat_loss_per_epoch))

        ar_test_loss_per_epoch = []
        
        for test_batch in test_dataloader:
            input_pose_sequences = train_batch[0].to(device)
            target_poses = train_batch[1].to(device)
            
            _ar_loss, _, _ = ar_train_step(input_pose_sequences, target_poses)
            
            _ar_loss = _ar_loss.detach().cpu().numpy()
            
            ar_test_loss_per_epoch.append(_ar_loss)
        
        ar_test_loss_per_epoch = np.mean(np.array(ar_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            autoreg.save_weights("results/weights/autoreg_weights_epoch_{}".format(epoch))
        
        loss_history["ar train"].append(ar_train_loss_per_epoch)
        loss_history["ar test"].append(ar_test_loss_per_epoch)
        loss_history["ar norm"].append(ar_norm_loss_per_epoch)
        loss_history["ar quat"].append(ar_quat_loss_per_epoch)
        
        print ('epoch {} : ar train: {:01.4f} ar test: {:01.4f} norm {:01.4f} quat {:01.4f} time {:01.2f}'.format(epoch + 1, ar_train_loss_per_epoch, ar_test_loss_per_epoch, ar_norm_loss_per_epoch, ar_quat_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_dataloader, test_dataloader, epochs)

# save history
utils.save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
utils.save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

# save model weights
torch.save(autoreg.state_dict(), "results/weights/autoreg_weights_epoch_{}".format(epochs))

# inference and rendering 
skel_edge_list = utils.get_skeleton_edge_list(skeleton)
poseRenderer = PoseRenderer(skel_edge_list)

# create ref pose sequence
def create_ref_sequence_anim(start_pose_index, pose_count, file_name):
    
    start_pose_index = max(start_pose_index, sequence_length)
    pose_count = min(pose_count, pose_sequence_length - start_pose_index)
    
    sequence_excerpt = pose_sequence[start_pose_index:start_pose_index + pose_count, :]
    sequence_excerpt = np.reshape(sequence_excerpt, (pose_count, joint_count, joint_dim))

    sequence_excerpt = torch.tensor(np.expand_dims(sequence_excerpt, axis=0)).to(device)
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence = skeleton.forward_kinematics(sequence_excerpt, zero_trajectory)

    skel_sequence = np.squeeze(skel_sequence.cpu().numpy())
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)


def create_pred_sequence_anim(start_pose_index, pose_count, file_name):
    autoreg.eval()
    
    start_pose_index = max(start_pose_index, sequence_length)
    pose_count = min(pose_count, pose_sequence_length - start_pose_index)
    
    start_seq = pose_sequence[start_pose_index - sequence_length:start_pose_index, :]
    start_seq = torch.from_numpy(start_seq).to(device)
    
    next_seq = start_seq
    
    pred_poses = []
    
    for i in range(pose_count):
        with torch.no_grad():
            pred_pose = autoreg(torch.unsqueeze(next_seq, axis=0))
    
        # normalize pred pose
        pred_pose = torch.squeeze(pred_pose)
        pred_pose = pred_pose.view((-1, 4))
        pred_pose = nn.functional.normalize(pred_pose, p=2, dim=1)
        pred_pose = pred_pose.view((1, pose_dim))

        pred_poses.append(pred_pose)
    
        #print("next_seq s ", next_seq.shape)
        #print("pred_pose s ", pred_pose.shape)

        next_seq = torch.cat([next_seq[1:,:], pred_pose], axis=0)
    
        print("predict time step ", i)

    pred_poses = torch.cat(pred_poses, dim=0)
    pred_poses = pred_poses.view((1, pose_count, joint_count, joint_dim))


    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    skel_poses = skeleton.forward_kinematics(pred_poses, zero_trajectory)
    
    skel_poses = skel_poses.detach().cpu().numpy()
    skel_poses = np.squeeze(skel_poses)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_poses)
    pose_images = poseRenderer.create_pose_images(skel_poses, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    pose_images[0].save(file_name, save_all=True, append_images=pose_images[1:], optimize=False, duration=33.0, loop=0) 

    autoreg.train()


seq_start_pose_index = 1000
seq_pose_count = 200

create_ref_sequence_anim(seq_start_pose_index, seq_pose_count, "ref_{}_{}.gif".format(seq_start_pose_index, seq_pose_count))
create_pred_sequence_anim(seq_start_pose_index, seq_pose_count, "pred_{}_{}.gif".format(seq_start_pose_index, seq_pose_count))
