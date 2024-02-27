from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from gym.wrappers import RecordVideo
from pathlib import Path
import base64
import os
import torch
from torch_geometric.data import Batch
# display = Display(visible=0, size=(1400, 900))
# display.start()
import time

def record_videos(env, video_folder="videos"):
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped


def show_videos(path="videos"):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))

def save_model(model,model_path="sac_gnn_models/"):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{model_path}/sac_gnn_{timestamp}.pt"
    torch.save(model,filename)
        

def save_model_expl(model,model_path="sac_gnn_models/exploration/"):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{model_path}/sac_gnn_expl{timestamp}.pt"
    torch.save(model,filename)
        
def save_model_gat_plus(model,model_path="sac_gnn_models/gat_plus/"):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{model_path}/sac_gat_plus{timestamp}.pt"
    torch.save(model,filename)
def save_model_uni(model,model_path="sac_gnn_models/unigraph/"):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{model_path}/sac_gat_uni{timestamp}.pt"
    torch.save(model,filename)
        
        

# in evaluation phase only need the leraned policy part(actor)
def load_model(model_path,actor_network):
    #inference on cpu for gpu trained model
    device = torch.device('cpu')
    model_eval = torch.load(model_path,map_location=device)
    actor_network.load_state_dict(model_eval['actor']) 
    return actor_network
    
def transfer_graph_to_gpu(graph_batch,device):
    if isinstance(graph_batch,Batch):
        for key in graph_batch.keys:
            graph_batch[key] = graph_batch[key].to(device)
    graph_batch = graph_batch.to(device)
    return graph_batch
        
def check_param_device(model):
    print(f"params: {model}")
    for name, param in model.named_parameters():
        print(f"Variable name: {name}, Device: {param.device}")



######important code to be put inside sb3########################

# # put in the type_alias.py file of sb3
# class GraphReplayBufferSamples(NamedTuple):
#     observations: Batch
#     actions: th.Tensor
#     next_observations: Batch
#     dones: th.Tensor
#     rewards: th.Tensor

# # put inside replay buffer class of sb3
# class GraphReplayBuffer(BaseBuffer):
#     """
#     Replay buffer used in off-policy algorithms like SAC/TD3.

#     :param buffer_size: Max number of element in the buffer
#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param device: PyTorch device
#     :param n_envs: Number of parallel environments
#     :param optimize_memory_usage: Enable a memory efficient variant
#         of the replay buffer which reduces by almost a factor two the memory used,
#         at a cost of more complexity.
#         See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
#         and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
#         Cannot be used in combination with handle_timeout_termination.
#     :param handle_timeout_termination: Handle timeout termination (due to timelimit)
#         separately and treat the task as infinite horizon task.
#         https://github.com/DLR-RM/stable-baselines3/issues/284
#     """

#     def __init__(
#         self,
#         buffer_size: int,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         device: Union[th.device, str] = "auto",
#         n_envs: int = 1,
#         optimize_memory_usage: bool = False,
#         handle_timeout_termination: bool = True,
#     ):
#         super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

#         # Adjust buffer size
#         self.buffer_size = max(buffer_size // n_envs, 1)

#         # Check that the replay buffer can fit into the memory
#         if psutil is not None:
#             mem_available = psutil.virtual_memory().available

#         # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
#         # see https://github.com/DLR-RM/stable-baselines3/issues/934
#         if optimize_memory_usage and handle_timeout_termination:
#             raise ValueError(
#                 "ReplayBuffer does not support optimize_memory_usage = True "
#                 "and handle_timeout_termination = True simultaneously."
#             )
#         self.optimize_memory_usage = optimize_memory_usage

#         #self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
#         self.observations = np.empty(shape=(self.buffer_size,), dtype=object)
#         if optimize_memory_usage:
#             # `observations` contains also the next observation
#             self.next_observations = None
#         else:
#             #self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
#             self.next_observations = np.empty(shape=(self.buffer_size,), dtype=object)
#         self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

#         self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         # Handle timeouts termination properly if needed
#         # see https://github.com/DLR-RM/stable-baselines3/issues/284
#         self.handle_timeout_termination = handle_timeout_termination
#         self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

#         if psutil is not None:
#             total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

#             if self.next_observations is not None:
#                 total_memory_usage += self.next_observations.nbytes

#             if total_memory_usage > mem_available:
#                 # Convert to GB
#                 total_memory_usage /= 1e9
#                 mem_available /= 1e9
#                 warnings.warn(
#                     "This system does not have apparently enough memory to store the complete "
#                     f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
#                 )

#     def add(
#         self,
#         obs: Data,
#         next_obs: Data,
#         action: np.ndarray,
#         reward: np.ndarray,
#         done: np.ndarray,
#         infos: List[Dict[str, Any]],
#     ) -> None:
#         # Reshape needed when using multiple envs with discrete observations
#         # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        
#         if isinstance(self.observation_space, spaces.Discrete):
#             obs = obs.reshape((self.n_envs, *self.obs_shape))
#             next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

#         # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
#         action = action.reshape((self.n_envs, self.action_dim))

#         # Copy to avoid modification by reference
#         #self.observations[self.pos] = np.array(obs).copy()
#         self.observations[self.pos] =  obs

#         if self.optimize_memory_usage:
#             self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
#         else:
#             #self.next_observations[self.pos] = np.array(next_obs).copy()
#             self.next_observations[self.pos] = next_obs

#         self.actions[self.pos] = np.array(action).copy()
#         self.rewards[self.pos] = np.array(reward).copy()
#         self.dones[self.pos] = np.array(done).copy()

#         if self.handle_timeout_termination:
#             self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

#         self.pos += 1
#         if self.pos == self.buffer_size:
#             self.full = True
#             self.pos = 0

#     def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
#         """
#         Sample elements from the replay buffer.
#         Custom sampling when using memory efficient variant,
#         as we should not sample the element with index `self.pos`
#         See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

#         :param batch_size: Number of element to sample
#         :param env: associated gym VecEnv
#             to normalize the observations/rewards when sampling
#         :return:
#         """
#         if not self.optimize_memory_usage:
#             return super().sample(batch_size=batch_size, env=env)
#         # Do not sample the element with index `self.pos` as the transitions is invalid
#         # (we use only one array to store `obs` and `next_obs`)
#         if self.full:
#             batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
#         else:
#             batch_inds = np.random.randint(0, self.pos, size=batch_size)
#         return self._get_samples(batch_inds, env=env)

#     def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
#         # Sample randomly the env idx
#         env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

#         # if self.optimize_memory_usage:
#         #     next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
#         # else:
#         #     next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
#         obs = [self.observations[idx] for idx in batch_inds]
#         obs = Batch.from_data_list(obs) 
        
#         next_obs =[self.next_observations[idx] for idx in batch_inds] if self.next_observations is not None else  None
#         next_obs = Batch.from_data_list(next_obs) if next_obs is not None else None
#         actions = self.to_torch(self.actions[batch_inds, env_indices, :])
#         # Only use dones that are not due to timeouts
#         # deactivated by default (timeouts is initialized as an array of False)
#         dones = self.to_torch((self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1))
#         rewards = self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env))
        
#         data = (
#             obs,
#             actions,
#             next_obs,
#             dones,
#             rewards,
#         )
#         return  GraphReplayBufferSamples(*tuple(data))