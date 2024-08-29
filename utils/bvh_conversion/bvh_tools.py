"""
important note: rotation conversion to quaternion currently only workss correctly 
for the euler rotation sequence: xrot, yrot, zrot
"""

from bvh_parsers import BVH_Parser
import pandas
import math
import numpy as np
import transforms3d as t3d

class SkeletonJoint:

    def __init__(self, name, offset):
        self.name = name
        self.local_offset = offset
        self.local_translation = np.array([0, 0, 0])
        self.local_rotation = t3d.quaternions.qeye()

        self.local_transformation = np.identity(4)
        self.world_transformation = np.identity(4)
        
        self.world_rotation = t3d.quaternions.qeye()
        self.world_position = np.array([0, 0, 0])
        
        self.parent = None
        self.children = list()

class Skeleton:
    
    def __init__(self):
        self.root_joint = None
        self.joints = list()

class BVH_Tools:
    def __init__(self):
        self.parser = BVH_Parser()
        self.bvh_data = None
        self.skeletons = []
        self.skeletons_frames = []
        self.euler_sequence = [0, 1, 2] # xyz
    
    # gather all root joint names
    # each root joint corresponds to a skeleton
    def _get_root_joint_names(self):
        bvh_skeleton = self.bvh_data.skeleton
        root_joint_names = list()
        for joint_name in bvh_skeleton:
            if bvh_skeleton[joint_name]["parent"] == None:
                root_joint_names.append(joint_name)
        return root_joint_names
    
    # traverse joint hiararchy
    def _traverse_create_joint_hierarchy(self, parent_joint_name, joint_hierarchy):
        bvh_skeleton = self.bvh_data.skeleton
        children_joint_names = bvh_skeleton[parent_joint_name]["children"]
        joint_hierarchy[parent_joint_name] = children_joint_names
    
        for child_joint_name in children_joint_names:
            self._traverse_create_joint_hierarchy(child_joint_name, joint_hierarchy)
    
        return joint_hierarchy

    # create joint hierarchy
    def _create_joint_hierarchy(self, root_joint_name):
        joint_names_hierarchy = dict()
        self._traverse_create_joint_hierarchy(root_joint_name, joint_names_hierarchy)
        return joint_names_hierarchy
    
    def _traverse_create_skeleton(self, skel_parent_joint, joint_hierarchy, skeleton):
        bvh_skeleton = self.bvh_data.skeleton
        children_joint_names = joint_hierarchy[skel_parent_joint.name]

        for child_joint_name in children_joint_names:
        
            children_joint_offset = np.array(bvh_skeleton[child_joint_name]["offsets"])
            skel_child_joint = SkeletonJoint(child_joint_name, children_joint_offset)
        
            skel_parent_joint.children.append(skel_child_joint)
            skel_child_joint.parent = skel_parent_joint
                
            skeleton.joints.append(skel_child_joint)
    
            self._traverse_create_skeleton(skel_child_joint, joint_hierarchy, skeleton)
    
    def _create_skeleton(self, root_joint_name, joint_hierarchy):
        bvh_skeleton = self.bvh_data.skeleton
        skeleton = Skeleton()
    
        root_joint_offset = np.array(bvh_skeleton[root_joint_name]["offsets"])
        skel_root_joint = SkeletonJoint(root_joint_name, root_joint_offset)
    
        skeleton.root_joint = skel_root_joint
        skeleton.joints.append(skel_root_joint)
    
        self._traverse_create_skeleton(skel_root_joint, joint_hierarchy, skeleton)
    
        return skeleton
    
    def _get_skeleton_frames(self, skeleton):
        bvh_frames = self.bvh_data.values
        bvh_frames_column_names = [ column for column in self.bvh_data.values.columns ]
        bvh_framecount = bvh_frames.shape[0]
        bvh_channels = set(self.bvh_data.channel_names)
        bvh_channel_joint_names = set([channel[0] for channel in bvh_channels])
        bvh_channel_value_names = ["Xposition", "Yposition", "Zposition", "Xrotation", "Yrotation", "Zrotation"]
    
        joint_frames = list()
    
        for joint in skeleton.joints:
            joint_name = joint.name
            if joint_name in bvh_channel_joint_names:
                joint_frames_combined = []      
            
                for i, value_name in enumerate(bvh_channel_value_names):
                    column_name = joint.name + "_" + value_name
                
                    if column_name in bvh_frames_column_names:
                        joint_frames_combined.append(np.array(bvh_frames[column_name]))
                    
                        #print("colname ", column_name, " values ", np.array(bvh_frames[column_name])[0])
                    
                    else:
                        joint_frames_combined.append(np.zeros(bvh_framecount))
                    

                joint_translations = joint_frames_combined[:3]
                joint_rotations = joint_frames_combined[3:]
            
                joint_translations = np.array(joint_translations)
                joint_rotations = np.array(joint_rotations)

                joint_translations = np.transpose(joint_translations)
                joint_rotations = np.transpose(joint_rotations)
            
                joint_frames.append( [joint_name, joint_translations, joint_rotations] )
            else:
                joint_frames.append( [joint_name] )
            
        return joint_frames
 
    def _skeleton_traverse_transformations(self, joint, parent_joint):
   
        # calculate local translation vector and rotation matrix
        _trans = joint.local_offset + joint.local_translation
        _rot = t3d.quaternions.quat2mat(joint.local_rotation)
    
        # create local transformation matrix
        joint.local_transformation = np.identity(4)
        joint.local_transformation[0:3, 0:3] = _rot
        joint.local_transformation[0:3, 3] = _trans
    
        # calculate world transformation matrix
        joint.world_transformation = np.matmul(parent_joint.world_transformation, joint.local_transformation)

        # calculate absolute joint position
        joint.world_position = np.matmul(joint.world_transformation, np.array([0, 0, 0, 1]))
        joint.world_position = joint.world_position[:3]
        
        # calculate abolute joint rotation
        joint.world_rotation = t3d.quaternions.mat2quat(joint.world_transformation[0:3, 0:3])
    
        #print("joint ", joint.name ," wpos ", joint.world_position)
    
        for child_joint in joint.children:
            self._skeleton_traverse_transformations(child_joint, joint)

    def _skeleton_update_transformations(self, skeleton):
        joint = skeleton.root_joint
    
        # calculate local translation vector and rotation matrix
        _trans = joint.local_offset + joint.local_translation
        _rot = t3d.quaternions.quat2mat(joint.local_rotation)
    
        # create local transformation matrix
        joint.local_transformation = np.identity(4)
        joint.local_transformation[0:3, 0:3] = _rot
        joint.local_transformation[0:3, 3] = _trans
    
        # for root node, local and world transformation matrix are identical
        joint.world_transformation = np.copy(joint.local_transformation)
    
        # calculate absolute joint position
        joint.world_position = np.matmul(joint.world_transformation, np.array([0, 0, 0, 1]))
        joint.world_position = joint.world_position[:3]
        
        # calculate abolute joint rotation
        joint.world_rotation = t3d.quaternions.mat2quat(joint.world_transformation[0:3, 0:3])

    
        #print("joint ", joint.name ," wpos ", joint.world_position)

        for child_joint in joint.children:
            self._skeleton_traverse_transformations(child_joint, joint)
   
    def _skeleton_set_frame(self, skeleton, skeleton_frame, frame_index):
        for joint_index, joint in enumerate(skeleton.joints):
            if len(skeleton_frame[joint_index]) > 1: # check if the frame contains transfomation info
                #print("joint ", joint.name, " trans ", joint.local_translation)
            
                # get local translation
                joint.local_translation = np.copy(skeleton_frame[joint_index][1][frame_index])
            
                # get local rotation in euler angles and degrees
                rel_rotation_euler = np.copy(skeleton_frame[joint_index][2][frame_index])

                # convert degrees to radians
                rel_rotation_euler[0] = rel_rotation_euler[0]/180.0 * math.pi;
                rel_rotation_euler[1] = rel_rotation_euler[1]/180.0 * math.pi;
                rel_rotation_euler[2] = rel_rotation_euler[2]/180.0 * math.pi;

                # convert euler rotation to quaternion
                joint.local_rotation = t3d.quaternions.qeye()

                quat_x = t3d.quaternions.axangle2quat([1, 0, 0], rel_rotation_euler[0])
                quat_y = t3d.quaternions.axangle2quat([0, 1, 0], rel_rotation_euler[1])
                quat_z = t3d.quaternions.axangle2quat([0, 0, 1], rel_rotation_euler[2])
                
                rotations = [quat_x, quat_y, quat_z]
                for rot_index in self.euler_sequence:
                    joint.local_rotation = t3d.quaternions.qmult(joint.local_rotation, rotations[rot_index])

                """
                print("update joint ", joint.name, " rel quat\n", joint.local_rotation)
                """
 
    def parse_bvh_file(self, file_name):
        parser = BVH_Parser()
        self.bvh_data = parser.parse(file_name)
        bvh_root_joint_names = self._get_root_joint_names()
        
        for root_joint_name in bvh_root_joint_names:
            bvh_joint_hierarchy = self._create_joint_hierarchy(root_joint_name)
            skeleton = self._create_skeleton(bvh_root_joint_names[0], bvh_joint_hierarchy)
            
            self.skeletons.append(skeleton)
        
        for skeleton in self.skeletons:
            skeleton_frames = self._get_skeleton_frames(skeleton)
            self.skeletons_frames.append(skeleton_frames)
        
        return self.skeletons, self.skeletons_frames

    def write_bvh_file(self, skeleton, frames, fps, file_name):
        
        with open(file_name, "w") as file:
            file.write("HIERARCHY\n")
            self._write_bvh_hierarchy(skeleton.root_joint, indent="", file=file)
            file.write("MOTION\n")
            file.write("Frames:	{}\n".format(frames[0][1].shape[0]))
            file.write("Frame Time:	{}\n".format(1.0 / fps))
            self._write_bvh_frames(frames, file=file)
    
    def _write_bvh_hierarchy(self, joint, indent, file):
        if joint.parent == None:
            file.write("{}ROOT {}\n".format(indent, joint.name))
        elif len(joint.children) > 0:
            file.write("{}JOINT {}\n".format(indent, joint.name))
        else:
            file.write("{}End Site\n".format(indent))
        
        file.write("{}".format(indent) + "{\n")
        file.write("  {}OFFSET {} {} {}\n".format(indent, joint.local_offset[0], joint.local_offset[1], joint.local_offset[2]))
        
        if len(joint.children) > 0:
            file.write("  {}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n".format(indent))

        for child in joint.children:
            self._write_bvh_hierarchy(child, "{}  ".format(indent), file)
        
        file.write("{}".format(indent) + "}\n")
          
    def _write_bvh_frames(self, frames, file):
        jointcount = len(frames)
        framecount = frames[0][1].shape[0]
        
        for frame in range(framecount):
            for joint in range(jointcount):
                if len(frames[joint]) == 1: # Nub
                    continue
                joint_rotations = frames[joint][1]
                joint_positions = frames[joint][2]

                joint_rotation = joint_rotations[frame]
                joint_position = joint_positions[frame]

                file.write("{} {} {} ".format(joint_rotation[0], joint_rotation[1], joint_rotation[2]))
                file.write("{} {} {} ".format(joint_position[self.euler_sequence[0]], joint_position[self.euler_sequence[1]], joint_position[self.euler_sequence[2]]))
                
            file.write("\n")

    def set_frame(self, frame_index):
        for skeleton, skeleton_frames in zip(self.skeletons, self.skeletons_frames):
            self._skeleton_set_frame(skeleton, skeleton_frames, frame_index)
            self._skeleton_update_transformations(skeleton)

    def create_datasets(self, start_frame_index=-1, end_frame_index=-1):
        
        if start_frame_index == -1:
            start_frame_index = 0
        if end_frame_index == -1:
            end_frame_index = self.bvh_data.values.shape[0]
    
        frameCount = end_frame_index - start_frame_index
        
        datasets = dict()
        
        for skeleton_index in range(len(self.skeletons)):
   
            dataset = dict()   
            datasets["S{}".format(skeleton_index + 1)] = dataset
   
            skeleton = self.skeletons[skeleton_index]
            joint_count = len(skeleton.joints)

            joint_names = list()
            joint_parents = list()
            joint_children = list()
            joints_offsets = np.zeros((joint_count, 3), dtype=np.float32)
    
            joint_index_map = dict()
            for joint_index, joint in enumerate(skeleton.joints):
                joint_index_map[joint] = joint_index
                
            for joint_index, joint in enumerate(skeleton.joints):

                joint_names.append(joint.name)
                joints_offsets[joint_index] = joint.local_offset          

                if joint.parent:
                    joint_parent_index = joint_index_map[joint.parent]
                    joint_parents.append(joint_parent_index)
                else:
                    joint_parents.append(-1)
                
                joint_children.append(list())
                
                for joint_child in joint.children:
                    joint_child_index = joint_index_map[joint_child]
                    
                    joint_children[joint_index].append(joint_child_index)
            
            dataset["names"] = joint_names
            dataset["offsets"] = joints_offsets
            dataset["parents"] = joint_parents
            dataset["children"] = joint_children
            
            skeleton_frames = self.skeletons_frames[skeleton_index]
            joints_pos_local = np.zeros((frameCount, joint_count, 3), dtype=np.float32)
            joints_pos_world = np.zeros((frameCount, joint_count, 3), dtype=np.float32)
            joints_rot_local = np.zeros((frameCount, joint_count, 4), dtype=np.float32)
            joints_rot_world = np.zeros((frameCount, joint_count, 4), dtype=np.float32)
            
            for frame_index in range(start_frame_index, end_frame_index):
                self.set_frame(frame_index)
                
                rel_frame_index = frame_index - start_frame_index
                
                for joint_index, joint in enumerate(skeleton.joints):
                     
                    joints_pos_local[rel_frame_index][joint_index][:] = joint.local_offset + joint.local_translation
                    joints_pos_world[rel_frame_index][joint_index][:] = joint.world_position
                    joints_rot_local[rel_frame_index][joint_index][:] = joint.local_rotation
                    joints_rot_world[rel_frame_index][joint_index][:] = joint.world_rotation
            
            dataset["pos_local"] = joints_pos_local
            dataset["pos_world"] = joints_pos_world
            dataset["rot_local"] = joints_rot_local
            dataset["rot_world"] = joints_rot_world
            
        return datasets