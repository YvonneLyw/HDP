from typing import Dict, List, Sequence
import torch
import numpy as np
import h5py
from tqdm import tqdm
import json
import copy
import hiera_diffusion_policy.common.transformation as tf
from hiera_diffusion_policy.common.robot import get_subgoals_stage_robomimic, get_subgoals_realtime_robomimic
from hiera_diffusion_policy.common.visual import visual_subgoals_v6, visual_pcd, getFingersPos, getGripperPos
from hiera_diffusion_policy.common.pytorch_util import dict_apply
from hiera_diffusion_policy.dataset.base_dataset import BasePcdDataset
from hiera_diffusion_policy.model.common.normalizer import Normalizer
from hiera_diffusion_policy.model.common.rotation_transformer import RotationTransformer
from hiera_diffusion_policy.common.replay_buffer import ReplayBuffer
from hiera_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from hiera_diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)
import robomimic.utils.file_utils as FileUtils
import random



class RobomimicReplayDataset(BasePcdDataset):
    def __init__(self,
            dataset_path: str,
            observation_history_num=2,
            use_subgoal=True,
            horizon=1,  # 16
            pad_before=0,   # 1
            pad_after=0,    # 7
            obs_keys: List[str]=[
                'object', 
                'robot0_eef_pos', 
                'robot0_eef_quat', 
                'robot0_gripper_qpos'],
            max_train_episodes=None,
            abs_action=False,   # True
            rotation_rep='rotation_6d',
            seed=42,
            Tr=1,
            val_ratio=0.02,
            use_image=False,
            image_keys: List[str]=None,
            image_size: Sequence[int]=(84, 84),
            d3p_query_every=1,
            d3p_action_chunk_len=None,
            qpos_normalize=True
        ):
        obs_keys = list(obs_keys)
        if image_keys is None:
            image_keys = ['agentview_image', 'robot0_eye_in_hand_image']
        image_keys = list(image_keys)

        rotation_transformer = RotationTransformer(             ## -> action rotation_6d
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file['data']

            if abs_action and 'absactions' in demos['demo_0']:  ## 只关乎key的名称，实际demo中的key是action
                action_key = 'absactions'
            else:
                action_key = 'actions'

            scene_pcd = demos['demo_0']['scene_pcd'][:].astype(np.float32)
            object_pcd = demos['demo_0']['object_pcd'][:].astype(np.float32)
            ## 遍历轨迹（每个 demo_i/episode），获取并整理episode data，连接所有episode塞进 ReplayBuffer
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):           ## demo_i
                demo = demos[f'demo_{i}']
                # get state / action    ##把原始 obs/actions 转成统一格式：data
                data = _data_to_obs(
                    raw_obs=demo['obs'],
                    obj_pcd=object_pcd,
                    scene_pcd=scene_pcd,
                    raw_actions=demo[action_key][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                    Tr=Tr,
                    use_image=use_image,
                    image_keys=image_keys)

                if use_subgoal:
                    #! stage subgoal                               ## 4 Algorithms
                    subgoals = get_subgoals_stage_robomimic(
                        demo['obs'],
                        object_pcd,
                        fin_rad=0.008,
                        sim_thresh=[0.02, 10./180*np.pi],
                        reward_mode='only_success',
                        Tr=Tr)                                    ##'subgoal'，'next_subgoal'，'reward'
                    #! realtime subgoal
                    # subgoals = get_subgoals_realtime_robomimic(
                    #     demo['obs'],
                    #     object_pcd,
                    #     fin_rad=0.008,
                    #     reward_mode='only_success',
                    #     Tr=Tr)
                    data.update(subgoals)

                    #* 可视化每个状态的子目标
                    # visual_subgoals_v6(
                    #     state=data['state'],
                    #     subgoal=subgoals['subgoal'],
                    #     reward=subgoals['reward'],
                    #     object_pcd=object_pcd, 
                    #     scene_pcd=scene_pcd)
                    
                replay_buffer.scene_pcd = scene_pcd
                replay_buffer.object_pcd = object_pcd
                replay_buffer.add_episode(data)   ######传入scene_pcd和object_pcd？每个episode各一份？？？？？
                
        val_mask = get_val_mask(    # shape=(n_episodes,)  用于测试的episode为1，用于训练的为0
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask      ##shape=(n_episodes,)  用于测试的episode为0，用于训练的为1

        if max_train_episodes is not None:  ##限制训练用的demo 数量，减少训练时间（可选）
            print(f'Use {max_train_episodes} demos to train!')
        else:
            print('Use all demos to train!')
        train_mask = downsample_mask(   # train_mask的shape不变，1的数量等于max_train_episodes  ##函数内部负责判断max_train_episodes决定是否 downsample
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        ##截取窗口
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            abs_action=abs_action,
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.replay_buffer = replay_buffer
        self.use_subgoal = use_subgoal
        self.observation_history_num = observation_history_num
        self.sampler = sampler
        self.abs_action = abs_action
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.Tr = Tr
        self.use_image = use_image
        self.image_keys = image_keys
        self.image_size = tuple(int(v) for v in image_size)
        self.d3p_query_every = int(d3p_query_every)
        if self.use_image and (not self.use_subgoal):
            raise ValueError("use_image=True requires use_subgoal=True for d3p_subgoal_pair.")
        if d3p_action_chunk_len is None:
            d3p_action_chunk_len = horizon
        self.d3p_action_chunk_len = int(d3p_action_chunk_len)
        
        self.qpos_normalize = bool(qpos_normalize)
        self.qpos_mean = None
        self.qpos_std = None
        if self.use_image and ('qpos_state' in self.replay_buffer):
            qpos_stat = self.replay_buffer['qpos_state']
            self.qpos_mean = np.mean(qpos_stat, axis=0).astype(np.float32)
            self.qpos_std = np.std(qpos_stat, axis=0).astype(np.float32)
            self.qpos_std = np.clip(self.qpos_std, 1e-2, np.inf)

    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            abs_action=self.abs_action,
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask
        return val_set


    def get_normalizer(self) -> Normalizer:
        normalizer = Normalizer()
        # action
        action_stat = array_to_stats(self.replay_buffer['action'])
        # if self.abs_action:
        action_params = robomimic_abs_action_only_normalizer_from_stat(action_stat)
        # else:
        #     # already normalized
        #     action_params = get_identity_normalizer_from_stat(action_stat)
        normalizer.params_dict['action'] = action_params
        
        # state
        # offset为0，scale的所有元素相同
        state_stat = array_to_stats(self.replay_buffer['state']) # 归一化参数去除物体位姿
        normalizer.params_dict['state'] = normalizer_from_stat(state_stat)

        return normalizer
    
    def __len__(self):                                              ## len(dataset)
        return len(self.sampler)

    def _sample_to_data(self, sample, i):
        data = {
            'id': np.array([i,]),

            'scene_pcd': self.replay_buffer.scene_pcd,
            'object_pcd': self.replay_buffer.object_pcd,
                                                                        ## n即observation_history_num
            'pcd': sample['data']['pcd'][:self.observation_history_num],   # (n, 1024, 3)
            'state': sample['data']['state'][:self.observation_history_num],  # (n, 27)
            'action': sample['data']['action'],

            'next_pcd': sample['data']['next_pcd'][:self.observation_history_num], # (n, 1024, 3)
            'next_state': sample['data']['next_state'][:self.observation_history_num],  # (n, 27)
            'next_action': sample['data']['next_action'],
        }

        if self.use_subgoal:
            subgoal_data = {
                'subgoal': sample['data']['subgoal'][self.observation_history_num-1],   # (8,)
                'next_subgoal': sample['data']['next_subgoal'][self.observation_history_num-1],   # (8,)
                'reward': sample['data']['reward'][self.observation_history_num-1:self.observation_history_num],
            }
            data.update(subgoal_data)

        if self.use_image:
            data.update(self._build_d3p_image_payload(sample, i))

        return data

    def _build_d3p_image_payload(self, sample, idx):
        seq = sample['data']
        seq_len = seq['front_image'].shape[0]

        # HDP current-t anchor keeps existing semantics with observation history.
        current_seq_idx = max(0, min(self.observation_history_num - 1, seq_len - 1)) ## 窗口内部的0是哪个内部idx）
        target_seq_idx = min(current_seq_idx + self.d3p_query_every, seq_len - 1)    ## +h
        doser_target_seq_idx = min(current_seq_idx + self.Tr, seq_len - 1)

        ## 图像处理：归一化/255 -> float32 -> CHW
        front_curr = _hwc_uint8_to_chw_float01(seq['front_image'][current_seq_idx], self.image_size)
        wrist_curr = _hwc_uint8_to_chw_float01(seq['wrist_image'][current_seq_idx], self.image_size)
        front_next = _hwc_uint8_to_chw_float01(seq['front_image'][target_seq_idx], self.image_size)
        wrist_next = _hwc_uint8_to_chw_float01(seq['wrist_image'][target_seq_idx], self.image_size)
        front_doser_next = _hwc_uint8_to_chw_float01(seq['front_image'][doser_target_seq_idx], self.image_size)
        wrist_doser_next = _hwc_uint8_to_chw_float01(seq['wrist_image'][doser_target_seq_idx], self.image_size)

        image = np.stack([
            np.stack([front_curr, wrist_curr], axis=0),
            np.stack([front_next, wrist_next], axis=0),
        ], axis=0).astype(np.float32)
        doser_image_pair = np.stack([
            np.stack([front_curr, wrist_curr], axis=0),
            np.stack([front_doser_next, wrist_doser_next], axis=0),
        ], axis=0).astype(np.float32)

        qpos = np.stack([
            seq['qpos_state'][current_seq_idx],
            seq['qpos_state'][target_seq_idx],
        ], axis=0).astype(np.float32)
        doser_qpos_pair = np.stack([
            seq['qpos_state'][current_seq_idx],
            seq['next_qpos_state'][current_seq_idx],
        ], axis=0).astype(np.float32)
        if self.qpos_normalize and (self.qpos_mean is not None):            ##标准化
            qpos = (qpos - self.qpos_mean[None, :]) / self.qpos_std[None, :]
            doser_qpos_pair = (
                doser_qpos_pair - self.qpos_mean[None, :]
            ) / self.qpos_std[None, :]

        d3p_subgoal_pair = np.stack([
            seq['subgoal'][current_seq_idx],
            seq['subgoal'][target_seq_idx],
        ], axis=0).astype(np.float32)

        """
        episode_x               |0|1|2|3|4|5|6|7|
        window[idx]             | | |x|x|x|x|x| |
        current_seq_idx=1       | | |0|1| | | |5|   target_seq_idx=5
        start_idx=2             | | |2| | | | | |
        current_episode_idx=3   | | | |3| | | |7|   target_episode_idx=7
        """

        ## pad
        episode_idx, episode_length, start_idx, _ = self.sampler.indices[idx]   ##start_idx：这个窗口在 episode 里的起点（可为负，表示左侧 pad）
        current_episode_idx = start_idx + current_seq_idx                       ## 窗口起始时间在episode 内时间索引 t
        target_episode_idx = current_episode_idx + self.d3p_query_every         ## t+h

        episode_action = self._get_episode_action_array(episode_idx)    ## 一整条episode的['action']
        current_act, current_act_is_pad = self._build_d3p_action_chunk(
            episode_action, current_episode_idx
        )
        target_act, target_act_is_pad = self._build_d3p_action_chunk(
            episode_action, target_episode_idx
        )
        d3p_action_pair = np.stack([current_act, target_act], axis=0).astype(np.float32)
        act_is_pad_pair = np.stack([current_act_is_pad, target_act_is_pad], axis=0).astype(np.bool_)
        # obs_is_pad = act_is_pad_pair[:, 0]

        payload = {
            'image': image,                     # (2, 2, C, H, W)
            'doser_image_pair': doser_image_pair, # (2, 2, C, H, W), aligned to t and t+Tr
            'qpos': qpos,                       # (2, 9)
            'doser_qpos_pair': doser_qpos_pair, # (2, 9), aligned to t and t+Tr
            'd3p_action_pair': d3p_action_pair, # (2, L, 10)
            'act_is_pad_pair': act_is_pad_pair, # (2, L)
            'd3p_subgoal_pair': d3p_subgoal_pair, # (2, subgoal_dim)
            # 'obs_is_pad': obs_is_pad,           # (2,)
        }
        return payload

    def _get_episode_action_array(self, episode_idx):
        if episode_idx == 0:
            start = 0
        else:
            start = self.replay_buffer.meta['episode_ends'][episode_idx-1]
        end = self.replay_buffer.meta['episode_ends'][episode_idx]
        return self.replay_buffer.data['action'][start:end]

    def _build_d3p_action_chunk(self, episode_action, timestep_idx):
        episode_len = episode_action.shape[0]
        chunk = np.zeros(
            (self.d3p_action_chunk_len, episode_action.shape[-1]),
            dtype=np.float32
        )
        is_pad = np.zeros((self.d3p_action_chunk_len,), dtype=np.bool_)

        for i in range(self.d3p_action_chunk_len):
            idx = timestep_idx + i
            if idx < 0:
                src_idx = 0
                is_pad[i] = True
            elif idx > (episode_len - 1):
                src_idx = episode_len - 1
                is_pad[i] = True
            else:
                src_idx = idx
            chunk[i] = episode_action[src_idx].astype(np.float32)
        return chunk, is_pad


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:     ## dataset[index]
        sample = self.sampler.sample_sequence(idx)      ## 使用窗口索引切片sampler.indices[idx]去切replay_buffer里的实值
        data = self._sample_to_data(sample, idx)        ## 处理各种key的数据（加入窗口id,'scene_pcd''object_pcd'，截短观测数据
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return Normalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
    

## def _data_to_obs(raw_obs, obj_pcd, scene_pcd, raw_actions, obs_keys, abs_action, rotation_transformer, Tr):
def _data_to_obs(raw_obs, obj_pcd, scene_pcd, raw_actions, obs_keys, abs_action,
                 rotation_transformer, Tr, use_image=False, image_keys=None):
    """
    args:
        raw_obs: h5py dict {
            - object  
            - robot0_eef_pos
            - robot0_eef_quat
            - robot0_gripper_qpos
            - agentview_image
            - robot0_eye_in_hand_image
            - robot0_joint_pos
        }
        raw_actions: np.ndarray shape=(N, A) N为当前轨迹长度，A为action维度
        obs_keys: list(), 需要的观测, 是raw_obs.keys()的子集合

    return: Dict
        `state`: (N, S) S为需要的观测合并的维度： 物体位姿/机械臂末端位姿/两个手指的位置
        `action`: (N, A) 其中的旋转分量转换成了rotation_6d，即连续的旋转表示

    """
    # get finger pos        ##输出gripper左右手指末端的世界坐标fl_pos, fr_pos
    fs_pos = list()
    for step in range(raw_obs['object'].shape[0]):
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0]+0.0145/2,   ##夹爪坐标系里两根手指 “沿 y 轴”的位移 +—基准点修正到指尖
            raw_obs['robot0_gripper_qpos'][step, 1]-0.0145/2,
            )
        fs_pos.append( np.concatenate((fl_pos, fr_pos), axis=0) )
        
    ##拼出来（obs_keys）的状态向量：['object'7+7, 'robot0_eef_pos'3, 'robot0_eef_quat'4]+左手指 xyz + 右手指 xyz
    obs = np.concatenate(
        [raw_obs[key] for key in obs_keys[:-1]] + [np.array(fs_pos),], 
        axis=-1).astype(np.float32)

    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]  
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)         ## -> action rotation_6d
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
    
    # 构建点云 ##每步的obj点云（由 object_pcd 经过物体位姿变换得到
    obj_pcd_batch = np.expand_dims(obj_pcd, axis=0).repeat(obs.shape[0], axis=0)    ##单帧点云复制成 N 帧的 batch
    obj_pcd_state = tf.transPts_tq_npbatch(obj_pcd_batch, obs[:, :3], obs[:, 3:7])  # (N, 1024, 3)
    # scene_pcd_batch = np.expand_dims(scene_pcd, axis=0).repeat(obs.shape[0], axis=0)    # (N, 1024, 3)
    # pcd_state = np.concatenate((obj_pcd_state, scene_pcd_batch), axis=1)    # (N, 2048, 3)

    if Tr > 0:
        curr_slice = slice(None, -Tr)   ##[:-Tr]
        next_slice = slice(Tr, None)    ##[Tr:]
    else:
        curr_slice = slice(None)
        next_slice = slice(None)

    data = {
        'pcd': obj_pcd_state[curr_slice], ## [:-Tr] ##每步的obj点云
        'state': obs[curr_slice],                   ##拼出来的状态向量：['object'14, 'robot0_eef_pos'3, 'robot0_eef_quat'4]+左手指 xyz + 右手指 xyz
        'action': raw_actions[curr_slice],          ##动作（rotation_6d）
        ##（通过 Tr 形成 (s_t, a_t) → (s_{t+Tr}, a_{t+Tr})）
        'next_pcd': obj_pcd_state[next_slice], ## [Tr:]
        'next_state': obs[next_slice],
        'next_action': raw_actions[next_slice],
    }

    if use_image:
        if image_keys is None:
            raise ValueError("image_keys must be provided when use_image=True")
        front_key, wrist_key = image_keys
        qpos = np.concatenate(
            [raw_obs['robot0_joint_pos'], raw_obs['robot0_gripper_qpos']],
            axis=-1
        ).astype(np.float32)
        data.update({
            'front_image': np.asarray(raw_obs[front_key][curr_slice]),
            'wrist_image': np.asarray(raw_obs[wrist_key][curr_slice]),
            'qpos_state': qpos[curr_slice],
            'next_qpos_state': qpos[next_slice],
        })

    return data


## 图像处理：归一化/255 -> float32 -> CHW
def _hwc_uint8_to_chw_float01(image_hwc, image_size):
    image = np.asarray(image_hwc)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image.shape}")
    h, w = image.shape[:2]
    expected_h, expected_w = image_size
    if (h, w) != (expected_h, expected_w):
        raise ValueError(
            f"Unexpected image size {(h, w)}. Config expects {(expected_h, expected_w)}"
        )
    image = image.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))
