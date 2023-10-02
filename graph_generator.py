import torch

from torch_geometric.data import Data
#from torch_geometric.utils import visualize
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx




#maybe make this whole file as a python class?

def create_edge_attributes_tensor(vehicle_1, vehicle_2):
    """
    The function creates edge attributes based on the differences between two vehicles' coordinates and velocities.
    
    :param vehicle_1: The first vehicle in the edge
    :param vehicle_2: The second vehicle in the edge, which is being compared to the first vehicle (vehicle_1) to calculate the edge attributes.
    :return: A tensor containing the edge attributes between two vehicles, which are the differences in their x and y positions, velocities, and orientations.
    """
    #handle null nodes 
    if (vehicle_1[1]!=0 and vehicle_1[2]!=0) and (vehicle_2[1]!=0 and vehicle_2[2]!=0):
        edge_attributes = torch.tensor([
        vehicle_1[1] - vehicle_2[1],
        vehicle_1[2] - vehicle_2[2],
        vehicle_1[3] - vehicle_2[3],
        vehicle_1[4] - vehicle_2[4],
        vehicle_1[7] - vehicle_2[7]
         ], dtype=torch.float)
    else:
        edge_attributes = torch.zeros(5,dtype=torch.float)
    
    return edge_attributes




def create_whole_ego_bigraph_tensor(node_attribute_matrix):
    """
    The `create_whole_ego_graph` function takes in a node attribute matrix and 
    creates a graph object using PyTorch Geometric library. 
    
    "vehicle_node_features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h","heading","long_off","lat_off","ang_off"]
    
    The graph is created with respect to the ego vehicle, 
    which is assumed to be the first vehicle in the node attribute matrix. 
    
    The edge attributes/features are calculated based on the difference between the ego vehicle and other vehicles' 
    attributes. The function returns the PyG Data object, 
    the position feature matrix, and the filtered vehicle indices. 
    
    The difference between `create_whole_ego_graph` and `create_ego_graph` 
    is that the former creates a graph with respect to all the vehicles(in the env.PERCEPTION_DISTANCE range) in the node attribute matrix, 
    while the latter creates a graph with respect to only the ego vehicle.
    """
    
    
    
    #edge_index = []
    #edge_attrs = []
   # num_node_features = node_attribute_matrix.shape
    max_obs_vehicles = 14 # no of nodes
    vehicle_node_features = torch.as_tensor(node_attribute_matrix[:max_obs_vehicles],dtype=torch.float).clone().detach()
    num_vehicles  = vehicle_node_features.shape[0]
    #ego_vehicle_idx = 0
    #edge_dim = 5 
    vehicle_indices = torch.arange(num_vehicles)
        
        #create edge indices for valid vehciles pairs using broadcasting and tensor operations, indexing follows matrix indexing convention
    vehicle_i, vehicle_j = torch.meshgrid(vehicle_indices,vehicle_indices,indexing='ij')
    valid_pair_masks = (vehicle_i!= vehicle_j)
    #print(valid_pair_masks)
    valid_pair_indices = torch.nonzero(valid_pair_masks,as_tuple=True)
    #print(valid_pair_indices)
    edge_index = torch.stack(valid_pair_indices,dim=1).t().contiguous()
    edge_attrs = torch.stack([create_edge_attributes_tensor(vehicle_node_features[i],vehicle_node_features[j]) for i,j in zip(*valid_pair_indices) ] )
    #edge_attrs = torch.stack([torch.tensor(list(edge_attr.values()), dtype=torch.float) for edge_attr in edge_attrs])
    
    #
   
    graph_data = Data(x=vehicle_node_features,edge_index = edge_index,edge_attr=edge_attrs)
    
    
    return graph_data


   

def create_whole_ego_bigrapher_multi(idx,vehicle_node_attribute_matrix):
    return create_whole_ego_bigraph_tensor(vehicle_node_attribute_matrix)
















def create_edge_attributes(vehicle_1,vehicle_2):
    """
    The function creates edge attributes based on the differences between two vehicles' coordinates and
    velocities.
    
    :param vehicle_1: The first vehicle in the edge
    :param vehicle_2: The second vehicle in the edge, which is being compared to the first vehicle
    (vehicle_1) to calculate the edge attributes. The vehicles are represented as a list of attributes,
    where the index of each attribute corresponds to a specific property of the vehicle (e.g. index 1
    represents the x
    :return: a dictionary containing the edge attributes between two vehicles, which are the differences
    in their x and y positions, velocities, and orientations.
    """
    #relative to the ego vehicle's viewpoint - fixing the coordinates system origin at the ego vehicle.
    edge_attributes = {"delta_x" : vehicle_1[1]- vehicle_2[1],
                      "delta_y" : vehicle_1[2]- vehicle_2[2],
                      "delta_vx" : vehicle_1[3]- vehicle_2[3],
                      "delta_vy" : vehicle_1[4]- vehicle_2[4],
                      "delta_theta" : vehicle_1[7] - vehicle_2[7]}
    
    return edge_attributes
    

def create_whole_ego_grapher_multi(idx,vehicle_node_attribute_matrix):
    return create_whole_ego_graph(vehicle_node_attribute_matrix)


def create_ego_graph(node_attribute_matrix):
    """
    This function creates a PyG Data object representing a graph with edge attributes based on a node
    attribute matrix, with a focus on the ego vehicle.
    
    "vehicle_node_features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h","heading","long_off","lat_off","ang_off"]
     
    param node_attribute_matrix: Shape (#vehicles, #features) where the first index is the truck
    creates the the graph wrt to the ego vehicle
    
    :param node_attribute_matrix: The node_attribute_matrix is a 2D numpy array with shape (#vehicles,
    #features), where each row represents a vehicle and each column represents a feature of that
    vehicle. The first row represents the ego vehicle. The features include attributes such as position,
    velocity, and heading
    
    :return: a PyG Data object, a matrix of vehicle positions, and a tensor of filtered vehicle indices.
    """
    

    
    edge_index = []
    edge_attrs = []
    vehicle_node_features = torch.tensor(node_attribute_matrix,dtype=torch.float).clone().detach()
    ego_vehicle_idx = 0
    
    #get  only x and y pos
    pos_feature_matrix = node_attribute_matrix[:,1:3]
    
    #filter out vehicles with zero entries in x and y direction
    filter_vehicle_node_features = vehicle_node_features[vehicle_node_features[:,1]!=0]
    filter_vehicle_node_features = filter_vehicle_node_features[filter_vehicle_node_features[:,2]!=0]
    #filter_vehicle_idx = torch.where(filter_vehicle_node_features)[0]
    filter_vehicles = torch.nonzero(vehicle_node_features[:,1:]).flatten()
    
    #adding the ego vehicle index
    filter_vehicle_idx = torch.cat([torch.tensor([ego_vehicle_idx]),filter_vehicles+1])

   
    # Create edge indices from the non-zero entries of the feature matrix
    for i in range(1, len(filter_vehicle_node_features)):
        #if vehicle_node_features[i][1]!=0 or vehicle_node_features[i][2]!=0:
        if any(filter_vehicle_node_features[i]):
            edge_index.append([ego_vehicle_idx,i])
            edge_index.append([i,ego_vehicle_idx])
            edge_attributes = create_edge_attributes(vehicle_node_features[0],vehicle_node_features[i])
            edge_attrs.append(torch.tensor(list(edge_attributes.values()),dtype=torch.float))
            #add edge_attributes in the other direction  - for  bidirectional graph
            edge_attrs.append(torch.tensor(list(edge_attributes.values()),dtype=torch.float))
    edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous()
    
     # for handling terminal/crash condtions( torch.stack() expects a non-empty TensorList)
    if edge_attrs:
        edge_attrs = torch.stack(edge_attrs)
        graph_data = Data(x=filter_vehicle_node_features,edge_index = edge_index,edge_attr=edge_attrs)
    else :
        graph_data = Data(x=filter_vehicle_node_features,edge_index = edge_index)
    
    
    #graph_data = Data(x=filter_vehicle_node_features,edge_index = edge_index,edge_attr=torch.stack(edge_attrs))
    
    return graph_data , pos_feature_matrix, filter_vehicle_idx



def create_whole_ego_bidir_graph(node_attribute_matrix):
    """
    The `create_whole_ego_graph` function takes in a node attribute matrix and 
    creates a graph object using PyTorch Geometric library. 
    
    "vehicle_node_features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h","heading","long_off","lat_off","ang_off"]
    
    The graph is created with respect to the ego vehicle, 
    which is assumed to be the first vehicle in the node attribute matrix. 
    
    The edge attributes/features are calculated based on the difference between the ego vehicle and other vehicles' 
    attributes. The function returns the PyG Data object, 
    the position feature matrix, and the filtered vehicle indices. 
    
    The difference between `create_whole_ego_graph` and `create_ego_graph` 
    is that the former creates a graph with respect to all the vehicles(in the env.PERCEPTION_DISTANCE range) in the node attribute matrix, 
    while the latter creates a graph with respect to only the ego vehicle.
    """
    
    
    
    edge_index = []
    edge_attrs = []
    vehicle_node_features = torch.tensor(node_attribute_matrix,dtype=torch.float).clone().detach()
    ego_vehicle_idx = 0
    #get  only x and y pos
    pos_feature_matrix = node_attribute_matrix[:,1:3]
    
    #filter out vehicles with zero entries in x and y direction
    filter_vehicle_node_features = vehicle_node_features[vehicle_node_features[:,1]!=0]
    filter_vehicle_node_features = filter_vehicle_node_features[filter_vehicle_node_features[:,2]!=0]
    #filter_vehicle_idx = torch.where(filter_vehicle_node_features)[0]
    filter_vehicles = torch.nonzero(vehicle_node_features[:,1:]).flatten()
    
    #adding the ego vehicle index
    filter_vehicle_idx = torch.cat([torch.tensor([ego_vehicle_idx]),filter_vehicles+1])

   
    # Create edge indices from the non-zero entries of the feature matrix between all the observed vehicles.
    for vehicle_i in range(len(filter_vehicle_node_features)):
        for vehicle_j in range(len(filter_vehicle_node_features)):
            if vehicle_i == vehicle_j:
                continue
        #if vehicle_node_features[i][1]!=0 or vehicle_node_features[i][2]!=0:
            if any(filter_vehicle_node_features[vehicle_i]) and any(filter_vehicle_node_features[vehicle_j]):
                edge_index.append([vehicle_i,vehicle_j])
                edge_index.append([vehicle_j,vehicle_i])
                edge_attributes = create_edge_attributes(vehicle_node_features[vehicle_i],vehicle_node_features[vehicle_j])
                
                edge_attrs.append(torch.tensor(list(edge_attributes.values()),dtype=torch.float))
                #add edge_attributes in the other direction  - for  bidirectional graph
                edge_attrs.append(torch.tensor(list(edge_attributes.values()),dtype=torch.float))
    edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous()
    
     # for handling terminal/crash condtions( torch.stack() expects a non-empty TensorList)
    if edge_attrs:
        edge_attrs = torch.stack(edge_attrs)
        graph_data = Data(x=filter_vehicle_node_features,edge_index = edge_index,edge_attr=edge_attrs)
    else :
        graph_data = Data(x=filter_vehicle_node_features,edge_index = edge_index)
    
    #graph_data = Data(x=filter_vehicle_node_features,edge_index = edge_index,edge_attr=torch.stack(edge_attrs))
    
    return graph_data , pos_feature_matrix, filter_vehicle_idx

def create_whole_ego_graph(node_attribute_matrix):
    """
    The `create_whole_ego_graph` function takes in a node attribute matrix and 
    creates a graph object using PyTorch Geometric library. 
    
    "vehicle_node_features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h","heading","long_off","lat_off","ang_off"]
    
    The graph is created with respect to the ego vehicle, 
    which is assumed to be the first vehicle in the node attribute matrix. 
    
    The edge attributes/features are calculated based on the difference between the ego vehicle and other vehicles' 
    attributes. The function returns the PyG Data object, 
    the position feature matrix, and the filtered vehicle indices. 
    
    The difference between `create_whole_ego_graph` and `create_ego_graph` 
    is that the former creates a graph with respect to all the vehicles(in the env.PERCEPTION_DISTANCE range) in the node attribute matrix, 
    while the latter creates a graph with respect to only the ego vehicle.
    """
    
    
    
    edge_index = []
    edge_attrs = []
    vehicle_node_features = torch.as_tensor(node_attribute_matrix,dtype=torch.float).clone().detach()
    ego_vehicle_idx = 0
    edge_dim = 5 #edge attributes 
    #get  only x and y pos
    #pos_feature_matrix = node_attribute_matrix[:,1:3]
    
    #filter out vehicles with zero entries in x and y direction
    filter_vehicle_node_features = vehicle_node_features[vehicle_node_features[:,1]!=0]
    filter_vehicle_node_features = filter_vehicle_node_features[filter_vehicle_node_features[:,2]!=0]
    filter_vehicle_node_features = filter_vehicle_node_features
    
    #check to see if there is only one node(ego vehicle)
    if len(filter_vehicle_node_features) ==1:
        #create an empty edge_index and edge_attribute tensor
        edge_index = torch.empty((2,0),dtype=torch.long)
        edge_attrs = torch.empty((0,edge_dim),dtype=torch.float)
        graph_data = Data(x=filter_vehicle_node_features,edge_index = edge_index,edge_attr=edge_attrs)
    
    else:
        # Create edge indices from the non-zero entries of the feature matrix between all the observed vehicles.
        for vehicle_i in range(len(filter_vehicle_node_features)):
            for vehicle_j in range(len(filter_vehicle_node_features)):
                if vehicle_i == vehicle_j:
                    continue
            #if vehicle_node_features[i][1]!=0 or vehicle_node_features[i][2]!=0:
                if any(filter_vehicle_node_features[vehicle_i]) and any(filter_vehicle_node_features[vehicle_j]):
                    edge_index.append([vehicle_i,vehicle_j])
                    edge_index.append([vehicle_j,vehicle_i])
                    edge_attributes = create_edge_attributes(vehicle_node_features[vehicle_i],vehicle_node_features[vehicle_j])
                    
                    edge_attrs.append(torch.tensor(list(edge_attributes.values()),dtype=torch.float))
                    #add edge_attributes in the other direction  - for  bidirectional graph
                    edge_attrs.append(torch.tensor(list(edge_attributes.values()),dtype=torch.float))
        edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous()
   
    # # for handling terminal/crash condtions( torch.stack() expects a non-empty TensorList)
        if edge_attrs:
            edge_attrs = torch.stack(edge_attrs)
            graph_data = Data(x=filter_vehicle_node_features,edge_index = edge_index,edge_attr=edge_attrs)
        else :
            graph_data = Data(x=filter_vehicle_node_features,edge_index = edge_index)
                                                                                    
    # if len(edge_attrs)==0:
    #graph_data = graph_data
        
    #     print(node_attribute_matrix)
    #     print(filter_vehicle_node_features)
    #     print(edge_attrs)
     
    
    
    return graph_data




#add the main plot function 