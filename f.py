import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from collections import OrderedDict
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from PIL import Image
import os, sys, time

# Define utility functions
def get_skeleton_edge_list(skeleton):
    # Get joint connections from the Skeleton class
    connections = skeleton.get_joint_connections()
    edges = []
    for start_joint, connected_joints in connections.items():
        for end_joint in connected_joints:
            if (start_joint, end_joint) not in edges and (end_joint, start_joint) not in edges:
                edges.append((start_joint, end_joint))
    return edges

# Define Skeleton class
class Skeleton:
    def __init__(self, joints, edges, joint_positions):
        self.joints = joints
        self.edges = edges
        self.joint_positions = joint_positions

    def num_joints(self):
        return len(self.joints)

    def forward_kinematics(self, poses, trajectories):
        # Placeholder for forward kinematics calculations
        return poses  # Implement actual forward kinematics here

    def get_joint_connections(self):
        """Returns a dictionary where each joint maps to a list of its connected joints."""
        connections = {joint: [] for joint in self.joints}
        for start, end in self.edges:
            connections[start].append(end)
            connections[end].append(start)  # Assuming bidirectional connections
        return connections

    def get_joint_position(self, joint):
        """Returns the position of the specified joint."""
        return self.joint_positions.get(joint, None)



# Define MocapDataset class
class MocapDataset:
    def __init__(self, file_path, fps):
        self.file_path = file_path
        self.fps = fps
        self.data = self._load_data()

    def _load_data(self):
        try:
            with h5py.File(self.file_path, 'r') as f:
                return f['data'][:]  # Adjust path according to your data structure
        except OSError:
            print(f"Failed to open HDF5 file. Attempting to load from pickle.")
            try:
                with open(self.file_path.replace('.h5', '.pkl'), 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error opening file: {e}")
                raise

    def __getitem__(self, idx):
        return self.data[idx]

    def compute_positions(self):
        # Placeholder
        pass

# Define quaternion operations
def qmul(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return r1 * r2

def qnormalize_np(q):
    r = R.from_quat(q)
    return r.as_quat()

# Define PoseRenderer class
class PoseRenderer:
    def __init__(self, edges):
        self.edges = edges

    def create_pose_image(self, skeleton, view_min, view_max, elevation, azimuth, line_width, size, view_size):
        plt.figure(figsize=(size, size))
        for edge in self.edges:
            start, end = edge
            start_pos = skeleton.get_joint_position(start)
            end_pos = skeleton.get_joint_position(end)
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k-', linewidth=line_width)
        plt.xlim(view_min[0], view_max[0])
        plt.ylim(view_min[1], view_max[1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('temp_image.png')
        plt.close()
        return Image.open('temp_image.png')


# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# mocap settings
mocap_data_path = "D:/Work/Living Scope Health/Test Reports Indiviual parsers - dev integration/healthbuddy-aiml/cbc-parser/mock_mocap_data.p"
mocap_valid_frame_ranges = [[500, 6500]]
mocap_fps = 50

# model settings
latent_dim = 8
gen_dense_layer_sizes = [16, 64, 128]
crit_dense_layer_sizes = [128, 64, 16]

save_models = False
save_tscript = False
save_weights = False

# load model weights
load_weights = False
generator_weights_file = "results/weights/generator_weights_epoch_400"
critique_weights_file = "results/weights/critique_weights_epoch_400"

# training settings
batch_size = 16
train_percentage = 0.8
test_percentage = 0.2
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

# Load mocap data
mocap_data = MocapDataset(mocap_data_path, fps=mocap_fps)
mocap_data.compute_positions()

# Function to update skeleton joint positions based on pose
def update_skeleton_with_pose(skeleton, pose):
    # Check if pose size matches the expected dimensions
    if pose.size == len(skeleton.joints) * 2 or pose.size == len(skeleton.joints) * 3:
        joint_positions = np.reshape(pose, (len(skeleton.joints), -1))
        for idx, joint in enumerate(skeleton.joints):
            if idx < len(joint_positions):
                skeleton.joint_positions[joint] = tuple(joint_positions[idx][:2])  # Use [:3] for 3D poses
    else:
        print(f'Pose size mismatch: {pose.size} expected {len(skeleton.joints) * 2 or len(skeleton.joints) * 3}')

# Update the Skeleton definition and the rest of the code accordingly.

# Gather skeleton info
# Dummy skeleton example
# Example joints and edges (adjust according to your data)
# Define the joint names and edges
# Example skeleton
joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # Replace with actual joint names
edges = [('joint1', 'joint2'), ('joint2', 'joint3'), ('joint3', 'joint4'), ('joint4', 'joint5'), ('joint5', 'joint6')]  # Replace with actual edges

# Example joint positions dictionary
joint_positions = {
    'joint1': (0.1, 0.2),
    'joint2': (0.4, 0.5),
    'joint3': (0.7, 0.8),
    'joint4': (1.0, 1.1),
    'joint5': (1.3, 1.4),
    'joint6': (1.6, 1.7)
}

# Initialize the Skeleton with joints, edges, and positions
skeleton = Skeleton(joints, edges, joint_positions)
skel_edge_list = get_skeleton_edge_list(skeleton)
print(skel_edge_list)

skeleton_joint_count = skeleton.num_joints()
skel_edge_list = get_skeleton_edge_list(skeleton)



# Inference and rendering
poseRenderer = PoseRenderer(skel_edge_list)

# Gather poses
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

# Create dataset
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

# Create models
class Critique(nn.Module):
    def __init__(self, pose_dim, dense_layer_sizes):
        super().__init__()

        self.pose_dim = pose_dim
        self.dense_layer_sizes = dense_layer_sizes

        # Create dense layers
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
        yhat = self.dense_layers(x)
        return yhat

critique = Critique(pose_dim, crit_dense_layer_sizes).to(device)

print(critique)

if save_models:
    critique.eval()
    torch.save(critique, "results/models/critique.pth")
    x = torch.zeros((1, pose_dim)).to(device)
    torch.onnx.export(critique, x, "results/models/critique.onnx")
    critique.train()

if save_tscript:
    critique.eval()
    x = torch.rand((1, pose_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(critique, x)
    script_module.save("results/models/critique.pt")
    critique.train()

if load_weights and critique_weights_file:
    critique.load_state_dict(torch.load(critique_weights_file))

# Create generator model
class Generator(nn.Module):
    def __init__(self, pose_dim, latent_dim, dense_layer_sizes):
        super(Generator, self).__init__()

        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.dense_layer_sizes = dense_layer_sizes

        # Create dense layers
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
        yhat = self.dense_layers(x)
        return yhat

generator = Generator(pose_dim, latent_dim, gen_dense_layer_sizes).to(device)

print(generator)

if save_models:
    generator.eval()
    torch.save(generator, "results/models/generator.pth")
    x = torch.zeros((1, latent_dim)).to(device)
    torch.onnx.export(generator, x, "results/models/generator.onnx")
    generator.train()

if save_tscript:
    generator.eval()
    x = torch.rand((1, latent_dim), dtype=torch.float32).to(device)
    script_module = torch.jit.trace(generator, x)
    script_module.save("results/models/generator.pt")
    generator.train()

if load_weights and generator_weights_file:
    generator.load_state_dict(torch.load(generator_weights_file))

# Define loss functions and optimizers
criterion = nn.BCELoss()

gen_optimizer = torch.optim.Adam(generator.parameters(), lr=gen_learning_rate)
crit_optimizer = torch.optim.Adam(critique.parameters(), lr=crit_learning_rate)

# Training loop
def train_one_epoch():
    generator.train()
    critique.train()

    for batch_idx, data in enumerate(train_dataloader):
        # Generate data
        z = torch.randn(batch_size, latent_dim, device=device)
        generated_data = generator(z)

        # Compute Critique loss
        critique_output = critique(generated_data)
        critique_loss = criterion(critique_output, torch.ones(batch_size, 1, device=device))

        # Backpropagation for Critique
        crit_optimizer.zero_grad()  # Use crit_optimizer here
        critique_loss.backward()
        crit_optimizer.step()

        # Generate data again
        z = torch.randn(batch_size, latent_dim, device=device)
        generated_data = generator(z)

        # Compute Generator loss
        critique_output = critique(generated_data)
        generator_loss = criterion(critique_output, torch.ones(batch_size, 1, device=device))

        # Backpropagation for Generator
        gen_optimizer.zero_grad()
        generator_loss.backward(retain_graph=True)  # Retain the graph for another backward pass
        gen_optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)}] \
                  Loss: {generator_loss.item():.6f} Critique Loss: {critique_loss.item():.6f}')

# Run training
for epoch in range(epochs):
    start_time = time.time()
    train_one_epoch()
    end_time = time.time()
    print(f'Epoch {epoch+1}/{epochs} - Time: {end_time - start_time:.2f}s')

    if save_weights and (epoch + 1) % weight_save_interval == 0:
        torch.save(generator.state_dict(), f"results/weights/generator_weights_epoch_{epoch+1}")
        torch.save(critique.state_dict(), f"results/weights/critique_weights_epoch_{epoch+1}")

# Visualize poses
# Process a subset of poses
num_poses_to_visualize = min(10, len(poses))  # Example: Visualize up to 10 poses

for idx in range(num_poses_to_visualize):
    pose = poses[idx]
    # Update skeleton joint positions with the current pose
    update_skeleton_with_pose(skeleton, pose)
    
    # Create the image with updated skeleton
    pose_image = poseRenderer.create_pose_image(
        skeleton,  # Pass the updated skeleton object
        view_min=[-1, -1], view_max=[1, 1],
        elevation=view_ele, azimuth=view_azi,
        line_width=view_line_width, size=view_size,
        view_size=view_size
    )
    
    # Save the image to disk instead of showing it (for large number of images)
    pose_image.save(f'pose_{idx}.png')

    # Optionally, display one image for verification
    if idx == 0:
        pose_image.show()