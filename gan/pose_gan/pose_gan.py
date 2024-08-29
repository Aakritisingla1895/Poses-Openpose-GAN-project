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
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# mocap settings
mocap_data_path = "../../../../Data/Mocap/Muriel_Nov_2021/MUR_PolytopiaMovement_Take2_mb_proc_rh.p"
mocap_valid_frame_ranges = [ [ 500, 6500 ] ]
mocap_fps = 50

# model settings
latent_dim = 8
gen_dense_layer_sizes = [ 16, 64, 128 ]
crit_dense_layer_sizes = [ 128, 64, 16 ]

save_models = False
save_tscript = False
save_weights = False

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
gen_norm_loss_scale = 0.1
gen_crit_loss_scale = 1.0
epochs = 200
weight_save_interval = 1
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

# inference and rendering 
skel_edge_list = utils.get_skeleton_edge_list(skeleton)
poseRenderer = PoseRenderer(skel_edge_list)

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

class Critique(nn.Module):
    def __init__(self, pose_dim, dense_layer_sizes):
        super().__init__()
        
        self.pose_dim = pose_dim
        self.dense_layer_sizes = dense_layer_sizes
        
        # create dense layers
        dense_layers = []
        
        dense_layers.append(("encoder_dense_0", nn.Linear(self.pose_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("encoder_dense_relu_0", nn.ReLU()))
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("encoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("encoder_dense_relu_{}".format(layer_index), nn.ReLU()))

        dense_layers.append(("encoder_dense_{}".format(len(self.dense_layer_sizes)), nn.Linear(self.dense_layer_sizes[-1], 1)))
        dense_layers.append(("encoder_dense_sigmoid_{}".format(len(self.dense_layer_sizes)), nn.Sigmoid()))
        
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
    def forward(self, x):
        #print("x 1 ", x.shape
        yhat = self.dense_layers(x)
        #print("yhat ", yhat.shape)
        return yhat

critique = Critique(pose_dim, crit_dense_layer_sizes).to(device)

print(critique)

"""
test_input = torch.zeros((1, pose_dim)).to(device)
test_output = critique(test_input)
"""

if save_models == True:
    critique.eval()
    
    # save using pickle
    torch.save(critique, "results/models/critique.pth")
    
    # save using onnx
    x = torch.zeros((1, pose_dim)).to(device)
    torch.onnx.export(critique, x, "results/models/critique.onnx")
    
    critique.train()

if save_tscript == True:
    critique.eval()
    
    # save using TochScript
    x = torch.rand((1, pose_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(critique, x)
    script_module.save("results/models/critique.pt")
    
    critique.train()

if load_weights and critique_weights_file:
    critique.load_state_dict(torch.load(critique_weights_file))

# create generator model

class Generator(nn.Module):
    def __init__(self, pose_dim, latent_dim, dense_layer_sizes):
        super(Generator, self).__init__()
        
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.dense_layer_sizes = dense_layer_sizes

        # create dense layers
        dense_layers = []
        
        dense_layers.append(("generator_dense_0", nn.Linear(latent_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("generator_relu_0", nn.ReLU()))

        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("generator_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("generator_dense_relu_{}".format(layer_index), nn.ReLU()))
            
        dense_layers.append(("generator_dense_{}".format(len(self.dense_layer_sizes)), nn.Linear(self.dense_layer_sizes[-1], self.pose_dim)))
 
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
    def forward(self, x):
        #print("x 1 ", x.size())
        
        # dense layers
        yhat = self.dense_layers(x)
        #print("yhat  ", yhat.size())
        

        return yhat

generator = Generator(pose_dim, latent_dim, gen_dense_layer_sizes).to(device)
    
print(generator)

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
    script_module = torch.jit.trace(generator, x)
    script_module.save("results/models/generator.pt")
    
    generator.train()

if load_weights and generator_weights_file:
    generator.load_state_dict(torch.load(generator_weights_file))
    
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

def plot_gan_outputs(generator, epoch, n=5):
    generator.eval()
    
    plt.figure(figsize=(10,4.5))
    
    zero_trajectory = torch.tensor(np.zeros((1, 1, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    for i in range(n):
        ax = plt.subplot(1,n,i+1)
        
        random_encoding = torch.randn((1, latent_dim)).to(device)
        
        with torch.no_grad():
            gen_pose  = generator(random_encoding)
            
        gen_pose = torch.squeeze(gen_pose)
        gen_pose = gen_pose.view((-1, 4))
        gen_pose = nn.functional.normalize(gen_pose, p=2, dim=1)
        gen_pose = gen_pose.view((1, 1, joint_count, joint_dim))
        
        skel_pose = skeleton.forward_kinematics(gen_pose, zero_trajectory)
        skel_pose = skel_pose.detach().cpu().numpy()
        skel_pose = np.reshape(skel_pose, (1, joint_count, 3))

        view_min, view_max = utils.get_equal_mix_max_positions(skel_pose)
        pose_image = poseRenderer.create_pose_images(skel_pose, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
        
        plt.imshow(pose_image[0])
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
        
        plot_gan_outputs(generator, epoch, n=5)

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

epochs = 400

# save history
utils.save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
utils.save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

# save model weights
torch.save(critique.state_dict(), "results/weights/critique_weights_epoch_{}".format(epochs))
torch.save(generator.state_dict(), "results/weights/generator_weights_epoch_{}".format(epochs))

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

def create_gen_pose_image(file_name):
    generator.eval()
    
    random_encoding = torch.randn((1, latent_dim)).to(device)
    
    with torch.no_grad():
        gen_pose = generator(random_encoding)
        
    gen_pose = torch.squeeze(gen_pose)
    gen_pose = gen_pose.view((-1, 4))
    gen_pose = nn.functional.normalize(gen_pose, p=2, dim=1)
    gen_pose = gen_pose.view((1, 1, joint_count, joint_dim))

    zero_trajectory = torch.tensor(np.zeros((1, 1, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)

    skel_pose = skeleton.forward_kinematics(gen_pose, zero_trajectory)

    skel_pose = skel_pose.detach().cpu().numpy()
    skel_pose = np.squeeze(skel_pose)    

    view_min, view_max = utils.get_equal_mix_max_positions(skel_pose)
    pose_image = poseRenderer.create_pose_image(skel_pose, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    pose_image.save(file_name, optimize=False)
    
    generator.train()

def decode_pose_encodings(pose_encodings, file_name):
    
    generator.eval()
    
    gen_poses = []
    
    for pose_encoding in pose_encodings:
        pose_encoding = np.expand_dims(pose_encoding, axis=0)
        pose_encoding = torch.from_numpy(pose_encoding).to(device)

        with torch.no_grad():
            gen_pose = gen_poses(pose_encoding)
            
        gen_pose = torch.squeeze(gen_pose)
        gen_pose = gen_pose.view((-1, 4))
        gen_pose = nn.functional.normalize(gen_pose, p=2, dim=1)
        gen_pose = gen_pose.view((1, joint_count, joint_dim))

        gen_poses.append(gen_pose)
    
    gen_poses = torch.cat(gen_poses, dim=0)
    gen_poses = torch.unsqueeze(gen_poses, dim=0)

    zero_trajectory = torch.tensor(np.zeros((1, len(pose_encodings), 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    skel_poses = skeleton.forward_kinematics(gen_poses, zero_trajectory)
    
    skel_poses = skel_poses.detach().cpu().numpy()
    skel_poses = np.squeeze(skel_poses)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_poses)
    pose_images = poseRenderer.create_pose_images(skel_poses, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    pose_images[0].save(file_name, save_all=True, append_images=pose_images[1:], optimize=False, duration=33.0, loop=0) 
    
    generator.train()
    
# create single original pose image

pose_index = 100

create_ref_pose_image(pose_index, "results/images/orig_pose_{}.gif".format(pose_index))

# generate single pose image
create_gen_pose_image("results/images/gen_pose_2.gif")
