import pickle
import json
import copy
import numpy as np
from numpy.core.umath_tests import inner1d

class DatasetTools:
    def __init__(self):
        self.dataset = None
    
    def load_dataset(self, file_path):
        if file_path.endswith(".p"):
            self._load_pickle(file_path)
        elif file_path.endswith(".json"):
            self._load_json(file_path)
        else:
            print("file type not recognized")
            
    def _load_pickle(self, file_path):
        with open(file_path, 'rb') as file:
            self.dataset = pickle.load(file)
    
    def _load_json(self, file_path):
        with open(file_path, 'r') as file:
            conv_dataset = json.load(file)
            self.dataset = self._convert_list_to_np(conv_dataset)
    
    def save_dataset(self, file_path):
        if file_path.endswith(".p"):
            self._save_pickle(file_path)
        elif file_path.endswith(".json"):
            self._save_json(file_path)
        else:
            print("file type not recognized")
    
    def _save_pickle(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump( self.dataset, file )
    
    def _save_json(self, file_path):
        conv_dataset = self._convert_np_to_list()
        with open(file_path, 'w') as file:
            json.dump(conv_dataset, file)
    
    def _convert_np_to_list(self):
        conv_dataset = copy.deepcopy(self.dataset)
        for subject_name, subject_dict in conv_dataset.items():
            for data_name, data in subject_dict.items():
                if isinstance(data, np.ndarray):
                    subject_dict[data_name] = data.tolist()
        return conv_dataset
        
    def _convert_list_to_np(self, dataset):
        conv_dataset = copy.deepcopy(dataset)
        # go through all data and check if data is a list
        # if yes, then check if first innermost value of list is a float
        # if yes, assume all values in the list are floats and convert list to numpy array
        for subject_name, subject_dict in conv_dataset.items():
            for data_name, data in subject_dict.items():
                if isinstance(data, list):
                    list_value = data
                    while( isinstance(list_value, list)):
                        list_value = list_value[0]
                    if isinstance(list_value, float):
                        #print("convert data ", data_name, " to array")
                        subject_dict[data_name] = np.array(data)
        
        return conv_dataset

    # calculate relative positions of all joints with respect to the position of a reference joint at a particular reference frame
    # arg ref_joint_name: name of the reference joint
    # arg ref_frame: index of the refefence frame (typically 0)
    # arg abs_pos_data_name: name of data containing absolute joint positions
    # arg rel_pos_data_name: name of data where the relative joint positions will be written to
    def remove_ref_position(self, ref_joint_name, ref_frame, abs_pos_data_name, rel_pos_data_name):
        assert(self.dataset != None)

        for subject_name, subject_dict in self.dataset.items():
            ref_joint_index = subject_dict["names"].index(ref_joint_name)
            #print("ref_joint_index ", ref_joint_index)
            abs_pos_data = subject_dict[abs_pos_data_name]
            abs_ref_pos = abs_pos_data[ref_frame, ref_joint_index, :]
            #print("abs_ref_pos ", abs_ref_pos)
            
            rel_pos_data = np.copy(abs_pos_data)
            rel_pos_data -= abs_ref_pos
            
            subject_dict[rel_pos_data_name] = rel_pos_data
    
    # calculate relative positions of all joints with respect to the directions of three reference joints at a particular reference frame
    # arg ref_joint_names: three joint names, typically this is: Hips, LeftUpLeg, Spine
    # arg ref_frame: index of the reference frame (typically 0)
    # arg abs_pos_data_name: name of data containing absolute joint positions
    # arg relrot_pos_data_name: name of data where the rotated joint positions will be written to
   
    def remove_ref_orientation(self, ref_joint_names, ref_frame, abs_pos_data_name, relrot_pos_data_name):
        assert(self.dataset != None)
        assert(len(ref_joint_names) == 3)
        
        for subject_name, subject_dict in self.dataset.items():
            ref1_joint_index = subject_dict["names"].index(ref_joint_names[0])
            ref2_joint_index = subject_dict["names"].index(ref_joint_names[1])
            ref3_joint_index = subject_dict["names"].index(ref_joint_names[2])
            
            abs_pos_data = subject_dict[abs_pos_data_name]
            abs_ref1_pos = abs_pos_data[ref_frame, ref1_joint_index, :]
            abs_ref2_pos = abs_pos_data[ref_frame, ref2_joint_index, :]
            abs_ref3_pos = abs_pos_data[ref_frame, ref3_joint_index, :]
            
            vecX = abs_ref2_pos - abs_ref1_pos
            vecY = abs_ref3_pos - abs_ref1_pos
            vecX /= np.linalg.norm(vecX)
            vecY /= np.linalg.norm(vecY)
            vecZ = np.cross(vecX, vecY)
            
            ref_matrix = np.zeros(shape=(3, 3), dtype=np.float32)
            ref_matrix[0, :] = vecX
            ref_matrix[1, :] = vecY
            ref_matrix[2, :] = vecZ
            
            inv_matrix = np.linalg.inv(ref_matrix)
            
            relrot_pos_data = np.copy(abs_pos_data)
            relrot_pos_data = np.matmul(relrot_pos_data, inv_matrix)

            subject_dict[relrot_pos_data_name] = relrot_pos_data
    
    # calculate the angle between joints
    # for three joints: angle between (j1 - j2) and (j3 - j2)
    # for four joints: angle between (j1 - j2) and (j4 - j3)
    # arg joint_names : three or four joint names
    # arg pos_data_name : name of data containing joint positions
    # arg angle_data_name : name of data to write angles to
    
    def _calc_angle_j3(self, joint_names, pos_data_name, angle_data_name):
        
        for subject_name, subject_dict in self.dataset.items():
            joint1_index = subject_dict["names"].index(joint_names[0])
            joint2_index = subject_dict["names"].index(joint_names[1])
            joint3_index = subject_dict["names"].index(joint_names[2])
            
            pos_data = subject_dict[pos_data_name]
            joint1_pos_data = pos_data[:, joint1_index, :].copy()
            joint2_pos_data = pos_data[:, joint2_index, :].copy()
            joint3_pos_data = pos_data[:, joint3_index, :].copy()
            
            joint21_dir = joint1_pos_data - joint2_pos_data
            joint23_dir = joint3_pos_data - joint2_pos_data
            
            joint21_len = np.expand_dims(np.linalg.norm(joint21_dir, axis=1), axis=1)
            joint21_dir /= joint21_len
            joint23_len = np.expand_dims(np.linalg.norm(joint23_dir, axis=1), axis=1)
            joint23_dir /= joint23_len

            joint_angle = inner1d(joint21_dir, joint23_dir)
            
            subject_dict[angle_data_name] = joint_angle
    
    def _calc_angle_j4(self, joint_names, pos_data_name, angle_data_name):
        
        for subject_name, subject_dict in self.dataset.items():
            joint1_index = subject_dict["names"].index(joint_names[0])
            joint2_index = subject_dict["names"].index(joint_names[1])
            joint3_index = subject_dict["names"].index(joint_names[2])
            joint4_index = subject_dict["names"].index(joint_names[3])
            
            pos_data = subject_dict[pos_data_name]
            joint1_pos_data = pos_data[:, joint1_index, :].copy()
            joint2_pos_data = pos_data[:, joint2_index, :].copy()
            joint3_pos_data = pos_data[:, joint3_index, :].copy()
            joint4_pos_data = pos_data[:, joint4_index, :].copy()
            
            joint21_dir = joint1_pos_data - joint2_pos_data
            joint43_dir = joint4_pos_data - joint3_pos_data
            
            joint21_len = np.expand_dims(np.linalg.norm(joint21_dir, axis=1), axis=1)
            joint21_dir /= joint21_len
            joint43_len = np.expand_dims(np.linalg.norm(joint43_dir, axis=1), axis=1)
            joint43_dir /= joint43_len

            joint_angle = inner1d(joint21_dir, joint43_dir)
            
            subject_dict[angle_data_name] = joint_angle
    
    def calc_angle(self, joint_names, pos_data_name, angle_data_name):
        assert(self.dataset != None)
        assert(len(joint_names) == 3 or len(joint_names) == 4)
        
        if len(joint_names) == 3:
            self._calc_angle_j3(joint_names, pos_data_name, angle_data_name)
        else:
            self._calc_angle_j4(joint_names, pos_data_name, angle_data_name)
        
            
            
            
            
            
            
            
            
                
            
            
            
    