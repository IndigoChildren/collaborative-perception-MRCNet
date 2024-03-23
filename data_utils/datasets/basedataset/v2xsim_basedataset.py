# Author: Yangheng Zhao <zhaoyangheng-sjtu@sjtu.edu.cn>
import os
import pickle
from collections import OrderedDict
from typing import Dict
from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset

from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
import random
class V2XSIMBaseDataset(Dataset):
    """
        First version.
        Load V2X-sim 2.0 using yifan lu's pickle file. 
        Only support LiDAR data.
    """

    def __init__(self,
                 params: Dict,
                 visualize: bool = False,
                 train: bool = True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.data_augmentor = DataAugmentor(params['data_augment'], train)
        # temporal mode
        self.temporal_mode = True if ('temporal_setting' in params and params['temporal_setting']) \
                                        else False
        self.frames_adj = params['temporal_setting']['frames_adj'] if self.temporal_mode else None  # the number of historical frames
        # supervise flag
        self.supervise_single_flag = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
        self.prediction = True if ('prediction' in params['model']['args'] and params['model']['args']['prediction']) \
                                        else False       
        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        self.root_dir = root_dir

        print("Dataset dir:", root_dir)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        assert self.label_type in ['lidar', 'camera']

        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                            else self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center

        # false
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False
        
        with open(self.root_dir, 'rb') as f:
            dataset_info = pickle.load(f)
        self.dataset_info_pkl = dataset_info

        # TODO param: one as ego or all as ego?
        self.ego_mode = 'one'  # "all"

        self.reinitialize()

    def reinitialize(self):
        # Add the sensor information in the order of data, initially create a datadict
        # where the lidar is the file address, and then go to the address to fetch the data at the time of retrive_data
        self.scene_database = OrderedDict()
        if self.ego_mode == 'one':
            self.len_record = len(self.dataset_info_pkl)
        else:
            raise NotImplementedError(self.ego_mode)

        for i, scene_info in enumerate(self.dataset_info_pkl):
            self.scene_database.update({i: OrderedDict()})
            cav_num = scene_info['agent_num']
            assert cav_num > 0

            if self.train:
                cav_ids = 1 + np.random.permutation(cav_num)
            else:
                cav_ids = list(range(1, cav_num + 1))
            

            for j, cav_id in enumerate(cav_ids):
                if j > self.max_cav - 1:
                    print('too many cavs reinitialize')
                    break

                self.scene_database[i][cav_id] = OrderedDict()

                self.scene_database[i][cav_id]['ego'] = j==0

                self.scene_database[i][cav_id]['lidar'] = scene_info[f'lidar_path_{cav_id}']
                # need to delete this line is running in /GPFS
                self.scene_database[i][cav_id]['lidar'] = \
                    self.scene_database[i][cav_id]['lidar'].replace("/GPFS/rhome/yifanlu/workspace/dataset/v2xsim2-complete", "dataset/V2X-Sim-2.0")

                self.scene_database[i][cav_id]['params'] = OrderedDict()

                self.scene_database[i][cav_id][
                    'params']['lidar_pose'] = tfm_to_pose(
                        scene_info[f"lidar_pose_{cav_id}"]
                    )  # [x, y, z, roll, pitch, yaw]
                self.scene_database[i][cav_id]['params'][
                    'vehicles'] = scene_info[f'labels_{cav_id}'][
                        'gt_boxes_global']
                self.scene_database[i][cav_id]['params'][
                    'object_ids'] = scene_info[f'labels_{cav_id}'][
                        'gt_object_ids'].tolist()

                # add the supervision of the ego car : lidar、param、lidar_pose
                cav_content = self.scene_database[i][cav_id]
                timestamp_index = i  # the current timestamp's index
                current_timestamp = self.dataset_info_pkl[timestamp_index]['timestamp']

                if self.supervise_single_flag and cav_content['ego']:
                    supervise_index = timestamp_index
                    self.scene_database[i]['supervise'] = OrderedDict() 
                    supervise_params = self.dataset_info_pkl[supervise_index][f'lidar_pose_{cav_id}']
                    self.scene_database[i]['supervise']['params'] = OrderedDict()
    
                    self.scene_database[i]['supervise'][
                        'params']['lidar_pose'] = tfm_to_pose(supervise_params)# [x, y, z, roll, pitch, yaw]
                    self.scene_database[i]['supervise']['params'][
                        'vehicles'] = self.dataset_info_pkl[supervise_index][f'labels_{cav_id}'][
                            'gt_boxes_global']
                    self.scene_database[i]['supervise']['params'][
                        'object_ids'] = self.dataset_info_pkl[supervise_index][f'labels_{cav_id}'][
                            'gt_object_ids'].tolist()
                    
                if self.prediction and cav_content['ego']:
                    # get the next frame's data
                    next_frame_index = min(timestamp_index+1, len(self.dataset_info_pkl)-1)
                    next_timestamp = self.dataset_info_pkl[next_frame_index]['timestamp']
                    # loss mask
                    has_future_frame = True if next_timestamp == (current_timestamp + 1) else False
                    next_frame_index = timestamp_index + 1 if has_future_frame else timestamp_index

                    # next frame's BEV feature
                    self.scene_database[i]['next_frame'] = OrderedDict() 
                    self.scene_database[i]['next_frame']['lidar'] = self.dataset_info_pkl[next_frame_index][f'lidar_path_{cav_id}']

                    self.scene_database[i]['next_frame']['params'] = OrderedDict()
                    self.scene_database[i]['next_frame'][
                        'params']['lidar_pose'] = tfm_to_pose(
                            self.dataset_info_pkl[next_frame_index][f"lidar_pose_{cav_id}"]
                        )  # [x, y, z, roll, pitch, yaw]
                    self.scene_database[i]['next_frame']['has_future_frame'] = has_future_frame
                    
                if self.temporal_mode and cav_content['ego']:
                    no_his_flag = True if current_timestamp == 6 else False

                    if not no_his_flag:
                        # the max number of hisrotical frames can be extacted
                        num_his_frames = current_timestamp - 6
                        his_frames = min(num_his_frames, self.frames_adj)
                        for index in range(his_frames, 0, -1):
                            his_timestamp_index = timestamp_index - index
                            self.scene_database[i][f'his_t_{index}'] = OrderedDict() 
                            self.scene_database[i][f'his_t_{index}']['lidar'] = self.dataset_info_pkl[his_timestamp_index][f'lidar_path_{cav_id}']

                            self.scene_database[i][f'his_t_{index}']['params'] = OrderedDict()
                            self.scene_database[i][f'his_t_{index}'][
                                'params']['lidar_pose'] = tfm_to_pose(
                                    self.dataset_info_pkl[his_timestamp_index][f"lidar_pose_{cav_id}"]
                                )  # [x, y, z, roll, pitch, yaw]
    def __len__(self) -> int:
        return self.len_record

    @abstractmethod
    def __getitem__(self, index):
        pass

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """

        data = OrderedDict()
        # {
        #     'cav_id0':{
        #         'ego': bool,
        #         'params': {
        #           'lidar_pose': [x, y, z, roll, pitch, yaw],
        #           'vehicles':{
        #                   'id': {'angle', 'center', 'extent', 'location'},
        #                   ...
        #               }
        #           },
        #         'camera_data':,
        #         'depth_data':,
        #         'lidar_np':,
        #         ...
        #     }
        #     'cav_id1': ,
        #     ...
        # }
        scene = self.scene_database[idx]
        if self.supervise_single_flag:
            supervise_info = scene['supervise']
            scene.pop('supervise')
        for cav_id, cav_content in scene.items():
            data[f'{cav_id}'] = OrderedDict()
            data[f'{cav_id}']['ego'] = cav_content['ego'] if 'ego' in cav_content else False

            data[f'{cav_id}']['params'] = cav_content['params']
            if 'has_future_frame' in cav_content:
                data[f'{cav_id}']['has_future_frame'] = cav_content['has_future_frame']
            # load the corresponding data into the dictionary
            nbr_dims = 4  # x,y,z,intensity
            scan = np.fromfile(cav_content['lidar'], dtype='float32')
            points = scan.reshape((-1, 5))[:, :nbr_dims]
            data[f'{cav_id}']['lidar_np'] = points
        if self.supervise_single_flag:
            data['supervise'] = supervise_info
        return data

    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Notice: it is a wrap of postprocessor function

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        return self.post_processor.generate_object_center_v2x(
            cav_contents, reference_lidar_pose)

    def generate_object_center_camera(self, cav_contents, reference_lidar_pose):
        raise NotImplementedError()

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask