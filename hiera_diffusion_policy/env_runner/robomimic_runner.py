import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from hiera_diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from hiera_diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from hiera_diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from hiera_diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from hiera_diffusion_policy.model.common.rotation_transformer import RotationTransformer

from hiera_diffusion_policy.policy.base_pcd_policy import BasePcdPolicy
from hiera_diffusion_policy.common.pytorch_util import dict_apply
from hiera_diffusion_policy.env_runner.base_pcd_runner import BasePcdRunner
# from hiera_diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from hiera_diffusion_policy.env.robomimic.robomimic_pcd_wrapper import RobomimicPcdWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from hiera_diffusion_policy.common.replay_buffer import ReplayBuffer
import hiera_diffusion_policy.common.transformation as tf
from hiera_diffusion_policy.common.visual import visual_subgoals_tilt_v44_1, visual_subgoals_tilt_v44_2, visual_pcd
import cv2


def create_env(env_meta, lowdim_keys, enable_render=True, image_keys=None):
    ## 告诉 robomimic: rollout 期间需要保留哪些 low-dimensional observation
    modality_mapping = {
        'low_dim': list(lowdim_keys)
    }
    if image_keys:
        modality_mapping['rgb'] = list(image_keys)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
    ## 根据dataset里的环境元信息env_meta恢复仿真环境
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,                   ## 不是实时弹窗显示 onscreen render
        # only way to not show collision geometry is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=enable_render, # 原始为False
        #use_image_obs=False,            ## policy 输入不是图像 observation，而是 low-dim state + 额外构造的点云/subgoal
        use_image_obs=True,    ##use_image_obs=False,            ## policy 输入不是图像 observation，而是 low-dim state + 额外构造的点云/subgoal
    )
    return env


def _stack_rollout_rgb_obs(extras, image_key, expected_hw):
    """
    Rollout env images are expected to already be robomimic-processed:
      - float32 in [0, 1]
      - CHW layout
    """
    image_batch = np.stack([e[image_key] for e in extras], axis=0).astype(np.float32)
    if image_batch.ndim != 4 or image_batch.shape[1] != 3:
        raise RuntimeError(
            f"Rollout {image_key} must be BCHW with 3 channels, got shape={image_batch.shape}"
        )
    if tuple(image_batch.shape[-2:]) != tuple(expected_hw):
        raise RuntimeError(
            f"Rollout {image_key} size mismatch: expected {tuple(expected_hw)}, got {tuple(image_batch.shape[-2:])}"
        )
    if image_batch.min() < -1e-6 or image_batch.max() > 1.0 + 1e-6:
        raise RuntimeError(
            f"Rollout {image_key} value range mismatch: expected [0,1], "
            f"got min={float(image_batch.min()):.4f}, max={float(image_batch.max()):.4f}"
        )
    return image_batch


class RobomimicRunner(BasePcdRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,                                             ##传入
            dataset_path,                                       #
            replay_buffer: ReplayBuffer,                            ##传入
            obs_keys,                                           #
            n_train=10,                                 # 6
            n_train_vis=3,                                      #2
            train_start_idx=0,                                  #
            n_test=22,                                  # 50
            n_test_vis=6,                                       #4
            test_start_seed=10000,                              #
            max_steps=400,                                      #500
            use_subgoal=True,                                   #{use_subgoal}
            use_pcd=True,                                       #${use_pcd}
            observation_history_num=2,                          #
            n_action_steps=8,                        ## AC长    # ${n_action_steps}
            n_latency_steps=0,                       ## inference长度         ##policy 仍然预测一整段动作序列；但前 n_latency_steps 个动作丢掉
            # 渲染参数
            render_hw=(256,256),                                #(128,128)
            render_camera_name='agentview',
            image_keys=None,
            qpos_normalize=True,
            fps=10,                                             #
            crf=22,                                             #
            past_action=False,                                  #
            abs_action=False,   # true                          #true
            tqdm_interval_sec=5.0,
            n_envs=None,                                # 28
            test_run=False                                      # ${test_run}
        ):
        """
        Assuming:
        observation_history_num=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:    ## rollout_0     ## step表示控制周期     ## 前 n_latency_steps 个a丢掉
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|   ## 执行
        Batch t+1   ## rollout_1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test
        if image_keys is None:
            image_keys = ['agentview_image', 'robot0_eye_in_hand_image']
        image_keys = list(image_keys)
        rollout_lowdim_keys = list(obs_keys)
        if 'robot0_joint_pos' not in rollout_lowdim_keys:
            rollout_lowdim_keys.append('robot0_joint_pos')

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = observation_history_num + n_latency_steps         ## 环境 wrapper 需要返回多少个 obs         ## 数值怎么给到环境？
        env_n_action_steps = n_action_steps                                 ## 环境 wrapper 一次接收多少步 action（实际执行时丢掉前latency步）

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20                              ##robosuite 仿真env的控制频率 / step 频率（每个 step 大约 0.05 秒）
        steps_per_render = max(robosuite_fps // fps, 1) ##仿真环境每走多少个 step，录一帧视频：环境每走 2 step，录 1 帧视频（视频想要 10fps）

        # read from dataset                             ## demo的环境配置
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # ## 确认环境配置里有没有启用对应 camera
        # print("=" * 80)
        # print("env_meta keys =", env_meta.keys())
        # print("env_meta =", env_meta)
        # print("env_kwargs keys =", env_meta.get("env_kwargs", {}).keys())
        # print("env_kwargs =", env_meta.get("env_kwargs", {}))
        # print("=" * 80)

        # ## 临时测试：强行打开 camera obs
        # env_meta["env_kwargs"]["use_camera_obs"] = True
        # env_meta["env_kwargs"]["has_offscreen_renderer"] = True
        # env_meta["env_kwargs"]["camera_names"] = ["agentview", "robot0_eye_in_hand"]
        # env_meta["env_kwargs"]["camera_heights"] = 84
        # env_meta["env_kwargs"]["camera_widths"] = 84
        # env_meta["env_kwargs"]["camera_depths"] = False
        
        rotation_transformer = None
        if abs_action:
            try:
                env_meta['env_kwargs']['controller_configs']['control_delta'] = False       ##修改环境 controller 配置，把输入动作当作绝对目标
            except:
                env_meta['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():       ## 单个包装好的环境
            ## 创建原始 robosuite / robomimic 环境（根据demo的环境元信息）
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    lowdim_keys=rollout_lowdim_keys,
                    image_keys=image_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            ## MultiStepWrapper 本身不处理 latency： 一次执行传进来的整段AC（8步），返回最近 n_obs_steps（2+latency）个 observation
            return MultiStepWrapper(                        
                    VideoRecordingWrapper(                      ## 给环境加视频录制功能。
                        RobomimicPcdWrapper(                        ## 单步的标准 gym 环境，并且统一获得的obs格式
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            image_keys=image_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,            ## 2+latency
                    n_action_steps=env_n_action_steps,      ## 8
                    max_episode_steps=max_steps             ## 500
                )

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        ## 正式环境用 env_fn()，可以渲染； 初始化 spaces 用 dummy_env_fn()“样板环境”，关闭 render，避免 OpenGL 冲突。
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    lowdim_keys=rollout_lowdim_keys,
                    image_keys=image_keys,
                    enable_render=False         ## 这里
                )
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicPcdWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            image_keys=image_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )


        """构建 rollout 任务列表 (6+50个)"""
        env_fns = [env_fn] * n_envs ## *28      ## 28个包装好的环境
        env_seeds = list()                      ## 任务表[train_idx0,...,5,（test的）seed0，...,49]     在下面代码登记
        env_prefixs = list()                    ## 标签['train/','train/',...,'test/','test/',... ]
        env_init_fn_dills = list()              ## 把初始化函数用drill序列化后存起来，等 rollout 时再发送给 worker env

        # train     ## 在一部分 train demos 对应的初始条件上 rollout 一遍                           ##没有启用！！
        with h5py.File(dataset_path, 'r') as f:     ## 从训练dataset里挑 n_train=6 个 demo / 初始条件，作为 rollout 评测任务
            for i in range(n_train):
                train_idx = train_start_idx + i     ## demo_0到demo_5
                enable_render = i < n_train_vis     ##只给前 2 个录视频
                # init_state = f[f'data/demo_{train_idx}/states'][0]        ##没有真的把 train env reset 到 dataset demo 的初始状态！！

                # def init_fn(env, init_state=init_state, 
                ## 任务初始化函数，没有seed 
                def init_fn(env, enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicPcdWrapper)
                    # env.env.env.init_state = init_state                   ##没有真的把 train env reset 到 dataset demo 的初始状态！！

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))           ## n_train=6 个任务 分别初始化 (待执行的 rollout 任务)
        
        # test      ## test 任务:靠随机种子区分 10000到10049
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis
            ## 任务初始化函数                                             ## 指定env的seed，init_state，file_path等，用于配置env的reset需要的参
            def init_fn(env, seed=seed, enable_render=enable_render):   ## 传入列表对应seed给对应任务初始化函数
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset  ##随机 seed 来控制环境 reset
                assert isinstance(env.env.env, RobomimicPcdWrapper)
                env.env.env.init_state = None                           ## 传入环境，用于环境初始化
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))           ## n_test=50 个任务 分别初始化 (待执行的 rollout 任务)
        
        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)    ## AsyncVectorEnv 在真正启动28个 worker 之前，先在主进程里创建一个“样板环境”来看一下
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.use_subgoal = use_subgoal
        self.use_pcd = use_pcd
        self.observation_history_num = observation_history_num
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.replay_buffer = replay_buffer
        self.test_run = test_run
        self.image_keys = image_keys
        self.rollout_lowdim_keys = rollout_lowdim_keys
        self.qpos_normalize = bool(qpos_normalize)
        self.qpos_mean = None
        self.qpos_std = None
        if self.qpos_normalize and ('qpos_state' in self.replay_buffer):
            qpos_stat = self.replay_buffer['qpos_state']
            self.qpos_mean = np.mean(qpos_stat, axis=0).astype(np.float32)
            self.qpos_std = np.std(qpos_stat, axis=0).astype(np.float32)
            self.qpos_std = np.clip(self.qpos_std, 1e-2, np.inf)


    def run(self, policy: BasePcdPolicy, first=False):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        expected_image_hw = tuple(int(v) for v in getattr(policy, 'image_size', (84, 84)))
        
        # plan for rollout
        n_envs = len(self.env_fns)  # 28                        ## 有28个并行环境
        n_inits = len(self.env_init_fn_dills)   # 56            ## 共有56个初始化任务待rollout      ##test和train的环境相同？
        n_chunks = math.ceil(n_inits / n_envs)  # 向上取整 2     ## 跑两轮

        # allocate data     ##分配结果存储空间
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        branch_score_A_trace = list()
        branch_score_B_trace = list()
        selected_branch_trace = list()
        branch_select_source_0_trace = list()
        branch_select_source_1_trace = list()
        branch_select_source_2_trace = list()
        doser_trace_keys = (
            'doser_action_percentile_A',
            'doser_action_percentile_B',
            'doser_action_id_A',
            'doser_action_id_B',
            'doser_state_percentile_A',
            'doser_state_percentile_B',
            'doser_q_A',
            'doser_q_B',
            'doser_v_A',
            'doser_v_B',
        )
        doser_traces = {key: list() for key in doser_trace_keys}

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)         ## [0:27]和[28:55]
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)  ## [0:27]
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]   ## init_fn [0:27]和[28:55]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            # 运行 init_fn(env), env_fn作为参数env      ## 把任务分配给 AsyncVectorEnv 里的每个 worker env
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()                         ## 初始化28个环境 ## (B, 2+latency, obs_dim)
            # past_action = None
            policy.reset()  ## 清除policy内部状态（RNN hidden state，历史缓存，扩散采样状态。。。）
            B = n_envs  ## 28
            ########################### use_rollout_image_qpos：是否需要B分支输入 ###########################################
            if hasattr(policy, 'needs_rollout_image_qpos'):
                use_rollout_image_qpos = bool(policy.needs_rollout_image_qpos())
            else:
                use_rollout_image_qpos = hasattr(policy, 'branch_condition_encoder')
            if first: return    ## 调试用，只reset不rollout

            # **** 记录图像和轨迹 ****
            ## path = '/home/wdx/research/diffusion_robot_manipulation/trajectory_all_task/square'？？？？？？？？？？？？？？
            results_action = list()
            step = 0

            # 从replay_buffer获取 scene_pcd/object_pcd/goal
            scene_pcd = self.replay_buffer.scene_pcd    # (1024, 3)
            object_pcd = self.replay_buffer.object_pcd  # (1024, 3)
            scene_pcd = np.expand_dims(scene_pcd, axis=0).repeat(B, axis=0) # (B, 1024, 3)
            object_pcd = np.expand_dims(object_pcd, axis=0).repeat(B, axis=0) # (B, 1024, 3)

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {self.env_meta['env_name']}Pcd {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)    ## 一轮 rollout 最多推进500步env底层step，显示进度条
            
            # 在toolhang任务时，replay_buffer额外保存`tool_pcd`和`frame_frame`
            # obs的最后1位为0/1，0表示操作物体为frame, 1表示操作物体为tool;
            # 根据标志位将对应的物体点云与`replay_buffer.scene_pcd_ori`合并

            done = False
            nnn = 0
            while not done:
                ########################### A分支输入量 ##################################################################
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    # 环境实际返回的观测包含5个观测，算法利用的观测只有前5-n_latency_steps个，模拟延迟获得n_latency_steps个观测数据
                    'state': obs[:,:self.observation_history_num].astype(np.float32)  # (28, 2, 7+7+6)  ##应该是14+7+6！！！  ## policy 只取前 2 帧
                }

                state = np_obs_dict['state']

                # *** 记录图像 ***
                # img = env.render_img()[10][..., ::-1]
                # cv2.imwrite(path+"/{:03d}.png".format(step), img)

                if self.use_pcd:
                    # 计算历史物体点云 (B, 2, 1024, 3)
                    obj_pcd = list()
                    for h in range(self.observation_history_num):
                        ## 根据历史 state，把物体模板点云object_pcd变换到每个历史时刻对应的位置和姿态obj_pcd_h
                        obj_pcd_h = tf.transPts_tq_npbatch(
                            object_pcd, state[:, h, :3], state[:, h, 3:7])      ## 位置和旋转
                        obj_pcd.append(obj_pcd_h)
                    obj_pcd = np.array(obj_pcd).transpose(1, 0, 2, 3)   # (n, B, 1024, 3)->(B, n, 1024, 3)
                    np_obs_dict['pcd'] = obj_pcd

                if self.use_subgoal:
                    # 预测子目标
                    Tinput_dict = dict_apply(np_obs_dict, 
                                            lambda x: torch.from_numpy(x).to(device=device))
                    subgoal = policy.predict_subgoal(Tinput_dict).detach().to('cpu').numpy()    # (B, 8)        ## policy预测subgoal
                    np_obs_dict['subgoal'] = subgoal 

                    #! 可视化状态和子目标
                    nnn += 1
                    # if self.test_run:
                    #     b = 0
                    #     print('*'*10, 'b =', b, '*'*10)
                    #     print('subgoal =', np_obs_dict['subgoal'][b])
                    #     visual_subgoals_tilt_v44_2(
                    #         state[b, -1], np_obs_dict['subgoal'][b], scene_pcd[b], object_pcd[b])

                ########################### B分支输入量 ##################################################################
                if use_rollout_image_qpos:
                    extras = env.call('get_policy_extras')
                    front = _stack_rollout_rgb_obs(extras, 'front_image', expected_image_hw)
                    wrist = _stack_rollout_rgb_obs(extras, 'wrist_image', expected_image_hw)
                    image_curr = np.stack((front, wrist), axis=1)  # (B,2,3,H,W)
                    np_obs_dict['image'] = np.stack((image_curr, image_curr), axis=1).astype(np.float32)

                    qpos_curr = np.stack([e['qpos_state'] for e in extras], axis=0).astype(np.float32)
                    if self.qpos_normalize and (self.qpos_mean is not None):
                        qpos_curr = (qpos_curr - self.qpos_mean[None, :]) / self.qpos_std[None, :]
                    np_obs_dict['qpos'] = np.stack((qpos_curr, qpos_curr), axis=1).astype(np.float32)

                # device transfer       ## np_obs_dict搬去cuda
                Tinput_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(Tinput_dict)    # dict{'action', 'action_pred'}         ## policy预测AC

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                ## 从 policy 输出：err等 取诊断字段做日志聚合（平均28个env)########################
                if 'branch_score_A' in np_action_dict:
                    branch_score_A_trace.append(float(np_action_dict['branch_score_A'].mean()))         ## (B,1) -> 标量 -> (循环次数,1)
                if 'branch_score_B' in np_action_dict:
                    branch_score_B_trace.append(float(np_action_dict['branch_score_B'].mean()))         ## (B,1) -> 标量 -> (循环次数,1)

                if 'selected_branch_exec_ratio' in np_action_dict:
                    selected_branch_trace.append(float(np_action_dict['selected_branch_exec_ratio'].mean()))## (B,1) -> 标量 -> (循环次数,1)
                elif 'selected_branch' in np_action_dict:
                    selected_branch_trace.append(float(np_action_dict['selected_branch'].mean()))   ## (B,1) -> 标量 -> (循环次数,1)
                if 'branch_select_source' in np_action_dict:
                    branch_select_source = np_action_dict['branch_select_source']
                    branch_select_source_0_trace.append(float((branch_select_source == 0).mean()))
                    branch_select_source_1_trace.append(float((branch_select_source == 1).mean()))
                    branch_select_source_2_trace.append(float((branch_select_source == 2).mean()))
                for key in doser_trace_keys:
                    if key in np_action_dict:
                        doser_traces[key].append(float(np_action_dict[key].mean()))

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]     ## 删掉前latency步a    ## (B,AC长，dim_a)
                if not np.all(np.isfinite(action)):         ##避免策略输出 NaN/Inf，防止环境崩掉。
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:                     ## rotation 6d->3d
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)      ##28 个环境执行一次rollout（AC-latency步 env底层step）

                # **** 记录action ****
                # 修改了multistep_wrapper的输出
                results_action.append(info[10])
                
                done = np.all(done)         ## 28 个环境都结束了，才跳出 while，执行下一rollout
                # past_action = action
                # update pbar
                pbar.update(action.shape[1])

                step += self.n_action_steps     ##应该是n_action_steps - n_latency_steps！！

            # **** 保存手指位置 ****
            # results_action = np.concatenate(tuple(results_action), axis=0)
            # np.save(path+'/action.npy', results_action)
            # print('轨迹记录完成!')
            # print('results_action.shape =', results_action.shape)

            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]     ## 28个环境对应的视频文件路径（VideoRecordingWrapper）
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]       ## 整条 episode 的 reward 记录

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):  #!!!!!!!!!!!!
        # and comment out this line
        for i in range(n_inits):        ## 遍历所有 56 个任务，整理指标
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]    ## train/或test/
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward  ## 各episode有一个最大reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video    ##可能录了视频

        # log aggregate metrics     ## train/test 平均reward
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        ###### 记录此次rollout（平均整条traj）的policy两分支输出########################
        if len(branch_score_A_trace) > 0:
            log_data['branch_score_A_mean'] = float(np.mean(branch_score_A_trace))          ## (循环次数,1) -> 标量
        if len(branch_score_B_trace) > 0:
            log_data['branch_score_B_mean'] = float(np.mean(branch_score_B_trace))          ## (循环次数,1) -> 标量
        if len(selected_branch_trace) > 0:
            log_data['selected_branch_B_ratio'] = float(np.mean(selected_branch_trace)) ## (循环次数,1) -> 标量
            # rollout内各step的选B率（不平均整条traj）
            selected_branch_table = wandb.Table(
                data=[
                    [int(step_idx), float(step_ratio)]
                    for step_idx, step_ratio in enumerate(selected_branch_trace)
                ],
                columns=['rollout_step', 'selected_branch_B_ratio'],
            )
            log_data['selected_branch_B_ratio_trace'] = wandb.plot.line(
                selected_branch_table,
                'rollout_step',
                'selected_branch_B_ratio',
                title='Selected Branch B Ratio Trace',
            )
        if len(branch_select_source_0_trace) > 0:
            if getattr(policy, 'branch_selector', None) in ('doser', 'doser_gt'):
                log_data['branch_select_source_both_action_id_q_ratio'] = float(np.mean(branch_select_source_0_trace))
                log_data['branch_select_source_one_action_id_one_ood_ratio'] = float(np.mean(branch_select_source_1_trace))
                log_data['branch_select_source_both_action_ood_ratio'] = float(np.mean(branch_select_source_2_trace))
            else:
                log_data['branch_select_source_err_ratio'] = float(np.mean(branch_select_source_0_trace))
                log_data['branch_select_source_q_ratio'] = float(np.mean(branch_select_source_1_trace))
                log_data['branch_select_source_hybrid_linear_ratio'] = float(np.mean(branch_select_source_2_trace))
        doser_log_names = {
            'doser_action_percentile_A': 'doser_action_percentile_A_mean',
            'doser_action_percentile_B': 'doser_action_percentile_B_mean',
            'doser_action_id_A': 'doser_action_id_A_rate',
            'doser_action_id_B': 'doser_action_id_B_rate',
            'doser_state_percentile_A': 'doser_state_percentile_A_mean',
            'doser_state_percentile_B': 'doser_state_percentile_B_mean',
            'doser_q_A': 'doser_q_A_mean',
            'doser_q_B': 'doser_q_B_mean',
            'doser_v_A': 'doser_v_A_mean',
            'doser_v_B': 'doser_v_B_mean',
        }
        for key, log_name in doser_log_names.items():
            if len(doser_traces[key]) > 0:
                log_data[log_name] = float(np.mean(doser_traces[key]))

        return log_data
    

    def undo_transform_action(self, action):            ## rotation 6d->3d
        # raw_shape = action.shape
        # if raw_shape[-1] == 20:
        #     # dual arm
        #     action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4    # 6
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        # if raw_shape[-1] == 20:
        #     # dual arm
        #     uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
