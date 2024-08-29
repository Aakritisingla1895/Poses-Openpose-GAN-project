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
gen_rnn_layer_count = 2
gen_rnn_layer_size = 512
gen_dense_layer_sizes = [ 512 ]
crit_rnn_layer_count = 2
crit_rnn_layer_size = 512
crit_dense_layer_sizes = [ 512 ]

save_models = False
save_tscript = False
save_weights = True

# load model weights
load_weights = True
generator_weights_file = "results/weights/generator_weights_epoch_150"
critique_weights_file = "results/weights/critique_weights_epoch_150"

# training settings
sequence_offset = 2 # when creating sequence excerpts, each excerpt is offset from the previous one by this value
batch_size = 16
train_percentage = 0.8 # train / test split
test_percentage  = 0.2
gen_learning_rate = 1e-4
crit_learning_rate = 1e-4
gen_norm_loss_scale = 0.1
gen_crit_loss_scale = 1.0
epochs = 500
weight_save_interval = 10
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

# create critique model

class Critique(nn.Module):
    def __init__(self, sequence_length, pose_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(Critique, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.rnn_layer_count = rnn_layer_count
        self.rnn_layer_size = rnn_layer_size 
        self.dense_layer_sizes = dense_layer_sizes
    
        # create recurrent layers
        rnn_layers = []
        rnn_layers.append(("critique_rnn_0", nn.LSTM(self.pose_dim, self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # create dense layers
        
        dense_layers = []
        
        dense_layers.append(("critique_dense_0", nn.Linear(self.rnn_layer_size, self.dense_layer_sizes[0])))
        dense_layers.append(("critique_dense_relu_0", nn.ReLU()))
        
        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("critique_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("critique_dense_relu_{}".format(layer_index), nn.ReLU()))

        dense_layers.append(("critique_dense_{}".format(len(self.dense_layer_sizes)), nn.Linear(self.dense_layer_sizes[-1], 1)))
        dense_layers.append(("critique_dense_sigmoid_{}".format(len(self.dense_layer_sizes)), nn.Sigmoid()))
        
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
    
critique = Critique(sequence_length, pose_dim, crit_rnn_layer_count, crit_rnn_layer_size, crit_dense_layer_sizes).to(device)

print(critique)

if save_models == True:
    critique.train()
    
    # save using pickle
    torch.save(critique, "results/models/critique.pth")
    
    # save using onnx
    x = torch.zeros((1, sequence_length, pose_dim)).to(device)
    torch.onnx.export(critique, x, "results/models/critique.onnx")
    
    critique.test()

if save_tscript == True:
    critique.train()
    
    # save using TochScript
    x = torch.rand((1, sequence_length, pose_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(critique, x)
    script_module.save("results/models/critique.pt")
    
    critique.test()

if load_weights and critique_weights_file:
    critique.load_state_dict(torch.load(critique_weights_file, map_location=device))

# create generator model

class Generator(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(Generator, self).__init__()
        
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
    
generator = Generator(sequence_length, pose_dim, latent_dim, gen_rnn_layer_count, gen_rnn_layer_size, gen_dense_layer_sizes).to(device)

print(generator)

if save_models == True:
    generator.eval()
    
    # save using pickle
    torch.save(generator, "results/models/generator_weights.pth")
    
    # save using onnx
    x = torch.zeros((1, latent_dim)).to(device)
    torch.onnx.export(generator, x, "results/models/generator.onnx")
    
    generator.train()

if save_tscript == True:
    generator.eval()
    
    # save using TochScript
    x = torch.rand((1, latent_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(generator, x)
    script_module.save("results/models/generator.pt")
    
    generator.train()

if load_weights and generator_weights_file:
    generator.load_state_dict(torch.load(generator_weights_file, map_location=device))

# Training

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

def gen_norm_loss(yhat):
    
    _yhat = yhat.view(-1, 4)
    _norm = torch.norm(_yhat, dim=1)
    _diff = (_norm - 1.0) ** 2
    _loss = torch.mean(_diff)
    return _loss

def gen_loss(yhat, crit_fake_output):
    _norm_loss = gen_norm_loss(yhat)
    _crit_loss = gen_crit_loss(crit_fake_output)
    
    _total_loss = 0.0
    _total_loss += _norm_loss * gen_norm_loss_scale
    _total_loss += _crit_loss * gen_crit_loss_scale
    
    return _total_loss, _norm_loss, _crit_loss

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
    
    _gen_loss, _norm_loss, _crit_loss = gen_loss(generated_poses, crit_fake_output)
    
    _gen_loss.backward()
    generator_optimizer.step()
 
    return _gen_loss, _norm_loss, _crit_loss

def gen_test_step(random_encodings):
    with torch.no_grad():
        generated_poses = generator(random_encodings)
    
        crit_fake_output = critique(generated_poses)
    
        _gen_loss, _norm_loss, _crit_loss = gen_loss(generated_poses, crit_fake_output)
    
    return _gen_loss, _norm_loss, _crit_loss

def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["gen train"] = []
    loss_history["gen test"] = []
    loss_history["crit train"] = []
    loss_history["crit test"] = []
    loss_history["gen crit"] = []
    loss_history["gen norm"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        crit_train_loss_per_epoch = []
        gen_train_loss_per_epoch = []
        gen_norm_loss_per_epoch = []
        gen_crit_loss_per_epoch = []
        
        for train_batch in train_dataloader:
            train_batch = train_batch.to(device)
            
            random_encodings = torch.randn((train_batch.shape[0], latent_dim)).to(device)

            # start with critique training
            _crit_train_loss = crit_train_step(train_batch, random_encodings)
            
            _crit_train_loss = _crit_train_loss.detach().cpu().numpy()

            crit_train_loss_per_epoch.append(_crit_train_loss)
            
            # now train the generator
            for iter in range(2):
                _gen_loss, _gen_norm_loss, _gen_crit_loss = gen_train_step(random_encodings)
            
                _gen_loss = _gen_loss.detach().cpu().numpy()
                _gen_norm_loss = _gen_norm_loss.detach().cpu().numpy()
                _gen_crit_loss = _gen_crit_loss.detach().cpu().numpy()
            
                gen_train_loss_per_epoch.append(_gen_loss)
                gen_norm_loss_per_epoch.append(_gen_norm_loss)
                gen_crit_loss_per_epoch.append(_gen_crit_loss)

        
        crit_train_loss_per_epoch = np.mean(np.array(crit_train_loss_per_epoch))
        gen_train_loss_per_epoch = np.mean(np.array(gen_train_loss_per_epoch))
        gen_norm_loss_per_epoch = np.mean(np.array(gen_norm_loss_per_epoch))
        gen_crit_loss_per_epoch = np.mean(np.array(gen_crit_loss_per_epoch))

        crit_test_loss_per_epoch = []
        gen_test_loss_per_epoch = []
        
        for test_batch in test_dataloader:
            test_batch = test_batch.to(device)
            
            random_encodings = torch.randn((train_batch.shape[0], latent_dim)).to(device)
            
            # start with critique testing
            _crit_test_loss = crit_test_step(train_batch, random_encodings)
            
            _crit_test_loss = _crit_test_loss.detach().cpu().numpy()

            crit_test_loss_per_epoch.append(_crit_test_loss)
            
            # now test the generator
            _gen_loss, _, _ = gen_test_step(random_encodings)
            
            _gen_loss = _gen_loss.detach().cpu().numpy()
            
            gen_test_loss_per_epoch.append(_gen_loss)

        crit_test_loss_per_epoch = np.mean(np.array(crit_test_loss_per_epoch))
        gen_test_loss_per_epoch = np.mean(np.array(gen_test_loss_per_epoch))
        
        if epoch % weight_save_interval == 0 and save_weights == True:
            torch.save(critique.state_dict(), "results/weights/critique_weights_epoch_{}".format(epoch))
            torch.save(generator.state_dict(), "results/weights/generator_weights_epoch_{}".format(epoch))
        
        loss_history["gen train"].append(gen_train_loss_per_epoch)
        loss_history["gen test"].append(gen_test_loss_per_epoch)
        loss_history["crit train"].append(crit_train_loss_per_epoch)
        loss_history["crit test"].append(crit_test_loss_per_epoch)
        loss_history["gen crit"].append(gen_crit_loss_per_epoch)
        loss_history["gen norm"].append(gen_norm_loss_per_epoch)

        print ('epoch {} : gen train: {:01.4f} gen test: {:01.4f} crit train {:01.4f} crit test {:01.4f} gen norm {:01.4f} gen crit {:01.4f} time {:01.2f}'.format(epoch + 1, gen_train_loss_per_epoch, gen_test_loss_per_epoch, crit_train_loss_per_epoch, crit_test_loss_per_epoch, gen_norm_loss_per_epoch, gen_crit_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_dataloader, test_dataloader, epochs)

# save history
utils.save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
utils.save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

# save model weights
torch.save(critique.state_dict(), "results/weights/critique_weights_epoch_{}".format(epochs))
torch.save(generator.state_dict(), "results/weights/generator_weights_epoch_{}".format(epochs))

# inference and rendering 
skel_edge_list = utils.get_skeleton_edge_list(skeleton)
poseRenderer = PoseRenderer(skel_edge_list)

def create_ref_sequence_anim(seq_index, file_name):
    sequence_excerpt = pose_sequence_excerpts[seq_index]
    sequence_excerpt = np.reshape(sequence_excerpt, (sequence_length, joint_count, joint_dim))
    
    sequence_excerpt = torch.tensor(np.expand_dims(sequence_excerpt, axis=0))
    zero_trajectory = torch.tensor(np.zeros((1, sequence_length, 3), dtype=np.float32))
    
    skel_sequence = skeleton.forward_kinematics(sequence_excerpt, zero_trajectory)
    skel_sequence = np.squeeze(skel_sequence.numpy())
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

def create_gen_sequence_anim(file_name):
    generator.eval()
    
    random_encoding = torch.randn((1, latent_dim)).to(device)
    
    with torch.no_grad():
        gen_sequence = generator(random_encoding)
        
    gen_sequence = torch.squeeze(gen_sequence)
    gen_sequence = gen_sequence.view((-1, 4))
    gen_sequence = nn.functional.normalize(gen_sequence, p=2, dim=1)
    gen_sequence = gen_sequence.view((1, sequence_length, joint_count, joint_dim))

    zero_trajectory = torch.tensor(np.zeros((1, sequence_length, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)

    skel_sequence = skeleton.forward_kinematics(gen_sequence, zero_trajectory)

    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)    

    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)
    
    generator.train()

create_gen_sequence_anim("test.gif")
