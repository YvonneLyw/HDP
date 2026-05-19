from typing import List, Dict, Optional
import numpy as np
import gym
from gym.spaces import Box
from robomimic.envs.env_robosuite import EnvRobosuite
from hiera_diffusion_policy.common.visual import getFingersPos


def updateState(raw_obs, obs_keys):
    """将 robot0_gripper_qpos 替换为 finger pos"""
    fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'], 
            raw_obs['robot0_eef_quat'], 
            raw_obs['robot0_gripper_qpos'][0]+0.0145/2,
            raw_obs['robot0_gripper_qpos'][1]-0.0145/2
            )
    # obs = np.concatenate(
    #     [raw_obs[key][:7] for key in obs_keys[:-1]] + [fl_pos, fr_pos], 
    #     axis=0)
    
    obs = np.concatenate(
        [raw_obs[key] for key in obs_keys[:-1]] + [fl_pos, fr_pos], 
        axis=0)
    
    # obs = np.concatenate(
    #     [raw_obs[key] for key in obs_keys
    #     ], axis=0)

    # obs = np.concatenate(
    #     [raw_obs[key][:7] for key in obs_keys], 
    #     axis=0)

    return obs


class RobomimicPcdWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        obs_keys: List[str]=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'],
        image_keys: Optional[List[str]]=None,   ############ 新增image ################
        init_state: Optional[np.ndarray]=None,          ## 如果不为空，reset 时固定到这个状态
        render_hw=(256,256),                            ## 渲染图像大小 (128,128)
        render_camera_name='agentview'                  ## 用哪个相机视角渲染
        ):

        self.env = env
        self.obs_keys = obs_keys
        if image_keys is None:              ############ 新增image ################
            image_keys = ['agentview_image', 'robot0_eye_in_hand_image']
        self.image_keys = list(image_keys)
        if len(self.image_keys) != 2:
            raise ValueError(f"Expected exactly 2 image keys for rollout extras, got {self.image_keys}")
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = dict()                    ## 缓存“seed 对应的初始状态”
        self._seed = None                               ## 下次 reset 要使用的 seed
        self._last_raw_obs = None           ##################保留“未加工”的obs快照################################################
        
        # setup spaces      ## 环境action space设置成标准 gym.spaces.Box：dim：env.action_dimension，每一维范围 [-1, 1]
        low = np.full(env.action_dimension, fill_value=-1)      ## env.action_dimension：从底层robomimic环境获取，最初是从demo的env_meta（中的controller配置）定义环境
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
        obs_example = self.get_observation()     ## 真的去引擎拿一份 observation 样本（并整理）， 设置 observation space
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        # 在Nonprehensile任务中，改成维度为包括 object pose/gripper pose/finger position
        self.observation_space = Box(
            low=low,          ## 不一定真实状态归一化到[-1,1]
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
  
    def pcd_goal(self):
        return self.env.pcd_goal()


    def get_observation(self):      ## 获取obs = [object, eef_pos, eef_quat, fl_pos, fr_pos]
        """
        获取flatten的观测数据
        """
        raw_obs = self.env.get_observation()
        self._last_raw_obs = raw_obs    ############新增#########
        obs = updateState(raw_obs, self.obs_keys)       ## 将 robot0_gripper_qpos 替换为 finger pos
        return obs

    def seed(self, seed=None):     ## 设置 numpy 随机种子, 把 seed 暂存在 self._seed, 真正使用是在下一次 reset()
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:     ## 按固定 init_state reset, 每次 rollout 从完全相同状态开始
            # always reset to the same state to be compatible with gym
            self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:        ## 按 seed reset
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:     ##若此seed已经生成过一个初始状态，直接恢复缓存初始状态
                # env.reset is expensive, use cache
                self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        # return obs
        obs = self.get_observation()    ## 获取obs = [object, eef_pos, eef_quat, fl_pos, fr_pos]
        return obs
    
    def step(self, action):     ## 执行一步 action
        raw_obs, reward, done, info = self.env.step(action)
        self._last_raw_obs = raw_obs    ############新增#########

        obs = updateState(raw_obs, self.obs_keys)       ##obs = [object, eef_pos, eef_quat, fl_pos, fr_pos]
        return obs, reward, done, info

    ########################### B分支输入量 ##################################################################
    def get_policy_extras(self):
        """
        Return image / proprio payload with the same semantics as the training dataset.
        提取原始 front_image、wrist_image、qpos_state
        """
        raw = self._last_raw_obs
        if raw is None:
            raw = self.env.get_observation()
            self._last_raw_obs = raw

        required_keys = list(self.image_keys) + [
            'robot0_joint_pos',
            'robot0_gripper_qpos',
        ]
        missing = [key for key in required_keys if key not in raw]
        if missing:
            raise KeyError(
                f"Missing rollout keys {missing}. Available observation keys: {sorted(raw.keys())}"
            )
        qpos_state = np.concatenate(
            [raw['robot0_joint_pos'], raw['robot0_gripper_qpos']],
            axis=-1
        ).astype(np.float32)

        return {
            'front_image': np.array(raw[self.image_keys[0]], copy=True),
            'wrist_image': np.array(raw[self.image_keys[1]], copy=True),
            'qpos_state': np.array(qpos_state, copy=True),
        }
    
    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, 
            height=h, width=w, 
            camera_name=self.render_camera_name)
