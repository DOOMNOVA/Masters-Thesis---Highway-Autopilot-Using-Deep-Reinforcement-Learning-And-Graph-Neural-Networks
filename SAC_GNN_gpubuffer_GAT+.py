import argparse
import os
import random
import time
from distutils.util import strtobool
import highway_env
import gymnasium as gym
import numpy as np
#import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import GraphReplayBuffer
from torch.utils.tensorboard import SummaryWriter
#from torch_geometric.nn import GATv2Conv
from gatpconv import GATpConv
from graph_generator import create_whole_ego_bigraph_tensor
#import traceback
from torch_geometric.data import Batch

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
from utils_gnn import save_model_gat_plus,  check_param_device
#import pygame
import os 




def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="SAC_GNN",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="highway-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=200001,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")#1e6
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default= 500,
        help="timestep to start learning")#5e3,15000
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")#1e-3
    parser.add_argument("--policy-frequency", type=int, default=15,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks") #train frequency in sb3 IT IS 1
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.5,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--dropout",type=float,default=0.5,
        help="dropout for the GNN layers")
    parser.add_argument("--vehicles-count",type=int,default=25,
        help="number of vehicles")
    
    # rewards
    # parser.add_argument("--high-speed-reward",type=float,default=0.7,
    #     help="reward for high speed")
    # parser.add_argument("--on-road-reward",type=float,default=0.2,
    #     help=" on road reward")
    # parser.add_argument("--collision-reward",type=float,default=-1,
    #     help="penalty for collisons")
    # parser.add_argument("--lane-change-reward",type=float,default=0.1,
    #     help="rewarding while changing lanes")
    # parser.add_argument("--right-lane-reward",type=float,default=0.1,
    #     help="reward for driving in the right lane. useful for exit ramp scenario")
    
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    #def thunk():
    env = gym.make(env_id)
    env.configure({
     
    #"import_module": "highway_env",
    "observation" : { "type" : "Kinematics",
                     "vehicles_count" :25,
                     "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h","heading","long_off","lat_off","ang_off"],
                    # "features_range": {
                    #     "x": [-100, 100],
                    #     "y": [-100, 100],
                    #     "vx": [-20, 25],
                    #     "vy": [-20, 25]  },
                    "absolute" : True,
                    "order" : "sorted",
                    "on_road" :  True,
                     "normalize" : True,
                       
        },
    "action": {
        "type": "ContinuousAction",
        "acceleration_range" : [-3.5,3.5],
        "steering_range" : [-np.pi/18,np.pi/18],
        'lateral' : True,
        'longitudinal': True,
        "speed_range" : [10,25],    
        
    }, #'offscreen_rendering': True,
        'offroad_terminal': True,
        "normalize_reward" : True,
        "simulation_frequency": args.policy_frequency,
        'high_speed_reward': 0.5,
        'on_road_reward' : 0.1,
        'collision_reward': -1,
        #'right_lane_reward' : 0.1,
        #"high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
        "lane_change_reward": 0.1,
        'reward_speed_range': [20, 25],
        'vehicles_density' : 1,
        
        "centering_position": [0.3, 0.5],
   
        'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
  
        'scaling': 5.5,
 
        'duration' : 200,
        "policy_frequency": args.policy_frequency,
    })

   
        #env.reset()

        #record  during training phase - comment if  not needed
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        if idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    #env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.reset(seed=seed)
    return env

    #return thunk



#the input should be the Data object generated from the obsevation 
#nodes - vehicles , node attributes-  kinematic features
#node attributes - (#nodes,node attributes) , adjacency matrix -(#nodes,#nodes) , edge attributes -(# edges, #edge attributes)
# class GAT_plus(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads):
#         super().__init__()
#         self.conv1 = GATpConv(in_channels, hidden_channels, heads, dropout=0.6, mode = 'att', self_loop =True,)
#         # On the Pubmed dataset, use `heads` output heads in `conv2`.
#         self.conv2 = GATpConv(hidden_channels * heads, out_channels, heads=1, mode= 'att', self_loop =True,
#                              concat=False, dropout=0.6)

#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x

class SoftGATv2Network(nn.Module):
    """ The SoftGATv2Network class is a neural network model that uses graph attention networks to process
     input data and output a new embedding for the ego vehicle node.
    """
    
    def __init__(self,env,graph_data,device,num_heads=5,dropout=0.6):
        super(SoftGATv2Network,self).__init__()
        self.device = device
        #graph_data = graph_data.to(self.device)
        self.num_heads = num_heads
        self.input_dim = graph_data.num_node_features
        #num_actions = np.prod(env.action_space.shape)
        edge_dim = graph_data.edge_attr.shape[1]
        #output_dim = graph_data.x.shape
        output_dim_row = graph_data.num_nodes
        #output_dim_column = graph_data.num_node_features
        self.output_dim_column = np.array(env.observation_space.shape).prod()
        self.dropout = dropout
        #self.input_dim = np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape)
        self.conv1 = GATpConv(in_channels= self.input_dim, out_channels= num_heads*self.input_dim,heads=num_heads,dropout=self.dropout,edge_dim=edge_dim,mode = 'att',add_self_loops= False)
        
        self.conv2 = GATpConv(in_channels= -1 ,out_channels=2*num_heads*self.input_dim,heads=1,dropout=self.dropout,edge_dim=edge_dim,mode = 'att',add_self_loops=False)
        
        #self.conv3 = GATv2Conv(in_channels= -1,out_channels= 4*num_heads*self.input_dim,heads=1,dropout=self.dropout,edge_dim=edge_dim,add_self_loops=False)
        
        self.dense = nn.Sequential(nn.Linear(2*num_heads*self.input_dim,output_dim_row),nn.Dropout(dropout),
                                   nn.Linear(output_dim_row,self.output_dim_column))
        
        
        
    
       #handle bothe Batch of Graphs and single graph inputs 
    def forward(self,graph_data):
        
        
        if isinstance(graph_data,Batch):
            x ,edge_index,edge_attr,batch = graph_data.x , graph_data.edge_index, graph_data.edge_attr, graph_data.batch
            #graph_batch_size = args.batch_size
        else:
            x ,edge_index,edge_attr = graph_data.x , graph_data.edge_index, graph_data.edge_attr
                                                                                                                                    
                                                                                                                                    
        
        x = F.elu(self.conv1(x,edge_index,edge_attr=edge_attr))
        x = F.dropout(x,p=self.dropout,training=self.training)
        
        x = F.elu(self.conv2(x,edge_index,edge_attr=edge_attr))
        x = F.dropout(x,p=self.dropout,training=self.training)
        
        # x = F.relu(self.conv3(x,edge_index,edge_attr=edge_attr))
        # x = F.dropout(x,p=self.dropout,training=self.training)

        #get the ego embeddings from the network
        if isinstance(graph_data,Batch):
            #new_ego_node_embeddings = torch.zeros((graph_data.num_graphs,4*self.num_heads*self.input_dim),device=self.device) # Shape: (batch_size, embedding_size)
            new_ego_node_embeddings = []
            for graph_index in torch.unique(batch):
                graph_mask = (batch == graph_index)
                new_ego_embed = x[graph_mask][0]
                new_ego_node_embeddings.append(new_ego_embed)
            new_ego_node_embeddings = torch.stack(new_ego_node_embeddings,dim=0)
        # for a single graph input while training
        else:
            new_ego_node_embeddings = x[0]

        #pass the ego embeddings to a dense layer
        new_ego_node_embeddings = torch.sigmoid(self.dense(new_ego_node_embeddings))
       
        return new_ego_node_embeddings
        
        
        
        
# ALGO LOGIC: initialize agent here: 
# the critic networks - Q1 and Q2 
class SoftQNetwork_GNN(nn.Module):
    def __init__(self, env,graph_data,device):
        super().__init__()
        self.device = device
        self.gat_gnn_critic = SoftGATv2Network(env,graph_data,self.device).to(self.device)
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        

    def forward(self, x, a):
        x = self.gat_gnn_critic(x)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor_GNN(nn.Module):
    def __init__(self, env,graph_data,device):
        super().__init__()
        self.device = device
        self.gat_gnn_actor = SoftGATv2Network(env,graph_data,self.device).to(self.device)
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling --maybe not needed, the acceleration is less, so the ego vehicle is slow.
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
        

    def forward(self, x):
        x = self.gat_gnn_actor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        if log_prob.dim() ==1:
            log_prob = log_prob.unsqueeze(1)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean



if __name__ == "__main__":
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
    #uncommnet below two line to view the vido while training
    # os.system("Xvfb :1 -screen 0 1024x768x24 &")
    # os.environ['DISPLAY'] = ':1'
    
    # pygame.display.init()
    #os.putenv('SDL_VIDEODRIVER', 'fbcon')
    
    # Set the multiprocessing start method
    #mp.set_start_method('spawn')
    
    #get the available cpu cores from the cluster
    # ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    # print("ncpus",ncpus)
    
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    #episodic_rewards = 0

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)
    # env setup
    envs = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)
    #envs.
    #envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    #assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"
    #print(envs.action_space.high[0])

    max_action = float(envs.action_space.high[0])

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.observation_space.dtype = np.float32
    rb = GraphReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs,info = envs.reset(seed=args.seed)
    print("info1: ",info)
 
    #print("obs1",obs)
    #print()
   
    
    
    ego_graph_data_cuda = create_whole_ego_bigraph_tensor(obs).to(device)
    #ego_graph_data_cuda = ego_graph_data.cuda()
    
    # initialize the actor and critics
    actor = Actor_GNN(envs,ego_graph_data_cuda,device).to(device) 
    qf1 = SoftQNetwork_GNN(envs,ego_graph_data_cuda,device).to(device) #critic 1
    check_param_device(qf1)
    qf2 = SoftQNetwork_GNN(envs,ego_graph_data_cuda,device).to(device) #critic 2
    check_param_device(qf2)
    qf1_target = SoftQNetwork_GNN(envs,ego_graph_data_cuda,device).to(device)
    check_param_device(qf1_target)
    qf2_target = SoftQNetwork_GNN(envs,ego_graph_data_cuda,device).to(device)
    check_param_device(qf2_target)
    qf1_target.load_state_dict(qf1.state_dict())
    qf1_target.to(device)
    check_param_device(qf1_target)
    qf2_target.load_state_dict(qf2.state_dict())
    qf2_target.to(device)
    check_param_device(qf2_target)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    
    
    
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        print(global_step)
        if global_step < args.learning_starts:
            #actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
            actions = np.array(envs.action_space.sample())
            #print(actions)
        else:
            #probably have to change it here to pass the new embedding to the actor 
        
            actions, _, _ = actor.get_action(ego_graph_data_cuda.to(device))
            actions = actions.detach().cpu().numpy()
            print(actions)
            print("dd")

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        #episodic_rewards += rewards
        # print("rewards: ",rewards)
        #print("next_obs: ",next_obs)
        print("infos_2: ",infos)
        next_ego_graph_data_cuda = create_whole_ego_bigraph_tensor(next_obs).to(device)
        
       

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        
        real_next_obs = next_obs.copy()
        
        
        #handling truncated observations -- have to double check
        if truncated and "terminal_observation" in infos.keys():
            real_next_obs = infos["terminal_observation"]
            next_ego_graph_data_cuda = create_whole_ego_bigraph_tensor(real_next_obs).to(device)
            print("terminal_obs",infos)
            
        #add data to the replay buffer  
        rb.add(ego_graph_data_cuda,next_ego_graph_data_cuda,actions,rewards,terminated,infos)
        
       
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook -convert the next observation to the graph
        #obs = next_obs
        ego_graph_batch_cuda = next_ego_graph_data_cuda
        # ego_graph_data = create_whole_ego_bigraph_tensor(vehicle_node_attribute_matrix)
        # ego_graph_data_cuda = ego_graph_data.cuda()
        
        
        #reset the episode if done
        if truncated or terminated:
            # TRY NOT TO MODIFY: record rewards for plotting purposes -#have to add the code for this
            if "episode" in infos.keys():
                print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
                
            
            #reset the environment
            obs, info_3 = envs.reset(seed=args.seed)
          
            ego_graph_data_cuda = create_whole_ego_bigraph_tensor(obs).to(device)
          
            print("reset_env")
    
      
   

   
       
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            next_ego_graph_batch = []
            ego_graph_batch = []
    

            with torch.no_grad():
                
                #get the batch of actions 
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
            
        
        
        
            
            #critic networks q1, q2 get current q values 
            qf1_a_values = qf1( data.observations, data.actions).view(-1)
            qf2_a_values = qf2( data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    #https://github.com/vwxyzjn/cleanrl/issues/379 min_qf_pi = torch.min(qf1_pi, qf2_pi) is the correct way!
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    # try:
                    #     actor_loss.backward()
                    # except Exception as e:
                    #     traceback.print_exc()
        #    
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            
            if global_step % 50000 == 0:
                #save the  trained model every 2e5 global_step/checkpoint
                model = dict()
                model['actor'] = actor.state_dict()
                model['actor_optimizer'] = actor_optimizer.state_dict()
                model['qf1'] = qf1.state_dict()
                model['qf2'] = qf2.state_dict()
                model['qf1_target'] = qf1_target.state_dict()
                model['qf2_target'] = qf2_target.state_dict()
                model['q_optimizer'] = q_optimizer.state_dict()
                #model = {k: v.cpu() for k, v in model.items()}
                save_model_gat_plus(model)
        if  global_step == 200000:
            model_final = dict()
            model_final['actor'] = actor.state_dict()
            model_final['actor_optimizer'] = actor_optimizer.state_dict()
            model_final['qf1'] = qf1.state_dict()
            model_final['qf2'] = qf2.state_dict()
            model_final['qf1_target'] = qf1_target.state_dict()
            model_final['qf2_target'] = qf2_target.state_dict()
            model_final['q_optimizer'] = q_optimizer.state_dict()
            save_model_gat_plus(model_final)
      

    envs.close()
    writer.close()