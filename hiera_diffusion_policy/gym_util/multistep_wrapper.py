import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill

def stack_repeated(x, n):       ## 重复x n次：  原来：x.shape=(20,) -->现在：(n, 20)
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n): ## 单步空间 变成 连续n步空间：表示一次输入n步动作/输出n步观测    原 action_space 是 (7,) -->现在：(n, 7)
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):   ## 递归地把单步空间扩展成多步空间
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):          ## 取序列的最后 n 个元素
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):     ## 字典x的每个 key 都取最后 n 个 value
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

def aggregate(data, method='max'):  ##把多步reward或done聚合成一个值：某一步done=True，最终就 done=True / reward最终取最大
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps): ##从 observation 历史里取最后 n_steps（2+latency） 帧；不够 就拿最早的一帧去padding
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)       ## 改action space（重复8次）
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)## 改observation space （重复 2+latency次）
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps  ## 2+latency
        self.n_action_steps = n_action_steps    ## 8
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps
        ## 初始化：保存 rollout 过程中的历史：
        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()       ## 单步环境reset
        ## 内部缓存初始化：一开始只有一帧 observation
        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        obs = self._get_obs(self.n_obs_steps)   ## observation 历史里的最后 2+latency 步 observation
        return obs
    
    def goal(self):
        return self.env.pcd_goal()


    def step(self, action, render=False):
        """
        actions: (n_action_steps,) + action_shape       ## (8, action_dim)
        """
        #! 测试用，将info修改为历史手指位置
        # fin_poss = list()

        for act in action:
            if len(self.done) > 0 and self.done[-1]:    ##之前已经 done，就停止
                # termination
                break
            observation, reward, done, info = super().step(act)     ## 执行单步环境

            # fin_poss.append(observation['low_dim'][-6:])    #! Tilt
            # fin_poss.append(observation[-6:])    #! Robomimic

            if render:
                super().render()
            ## 保存observation, reward, done, info
            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):       ## 检查最大步数截断；len(self.reward)：当前 episode 已执行多少个底层 step
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)           ## 历史里取最后 n_steps（2+latency） 帧
        reward = aggregate(self.reward, self.reward_agg_method) ## 1个值
        done = aggregate(self.done, 'max')                      ## 1个值
        info = dict_take_last_n(self.info, self.n_obs_steps)

        return observation, reward, done, info    #! 原输出
        # return observation, reward, done, np.array(fin_poss)

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)      ## 从 observation 历史里取最后 n_steps（2+latency） 帧
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):      ## 把每一步 step 返回的 info 按字段保存成历史序列
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):   ## 在环境中 执行任务初始化函数
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
