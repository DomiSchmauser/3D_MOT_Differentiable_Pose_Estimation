import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_max, scatter_add

from Tracking.networks.mlp import MLP


class MetaLayer(nn.Module): # Single Layer of Message Passing Network

    def __init__(self, edge_model=None, node_model=None):
        super(MetaLayer, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix with shape [num_nodes, num_node_features]
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)
        Returns: Updated Node and Edge Feature matrices
        """
        row, col = edge_index # row = start node, col = destination node

        # Edge Update
        edge_attr = self.edge_model(x[row], x[col], edge_attr) # edge update with nodes i,j and edge ij

        # Node Update
        x = self.node_model(x, edge_index, edge_attr) # node i update with node i at t-1 and edge ij at t

        return x, edge_attr

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)

class EdgeModel(nn.Module):

    def __init__(self, edge_mlp):
        super(EdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, source, target, edge_attr):
        out = torch.cat([source, target, edge_attr], dim=1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    '''
    try, except to avoid cuda error
    '''

    def __init__(self, node_mlp, node_agg_fn):
        super(NodeModel, self).__init__()

        self.node_mlp = node_mlp
        self.node_agg_fn = node_agg_fn

    def forward(self, x, edge_index, edge_attr):

        row, col = edge_index

        message = self.node_agg_fn(edge_attr, row, x.size(0)) # node_i x edge_dim

        node_message = torch.cat([x, message], dim=1)
        return self.node_mlp(node_message)

class TimeAwareNodeModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """
    def __init__(self, flow_in_mlp, flow_out_mlp, node_mlp, node_agg_fn):
        super(TimeAwareNodeModel, self).__init__()

        self.flow_in_mlp = flow_in_mlp
        self.flow_out_mlp = flow_out_mlp
        self.node_mlp = node_mlp
        self.node_agg_fn = node_agg_fn

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        flow_out_mask = row < col
        flow_out_row, flow_out_col = row[flow_out_mask], col[flow_out_mask]
        flow_out_input = torch.cat([x[flow_out_col], edge_attr[flow_out_mask]], dim=1)
        flow_out = self.flow_out_mlp(flow_out_input)
        flow_out = self.node_agg_fn(flow_out, flow_out_row, x.size(0))

        flow_in_mask = row > col
        flow_in_row, flow_in_col = row[flow_in_mask], col[flow_in_mask]
        flow_in_input = torch.cat([x[flow_in_col], edge_attr[flow_in_mask]], dim=1)
        flow_in = self.flow_in_mlp(flow_in_input)

        flow_in = self.node_agg_fn(flow_in, flow_in_row, x.size(0))
        flow = torch.cat((flow_in, flow_out), dim=1)

        return self.node_mlp(flow)

class MLPGraphIndependent(nn.Module):

    def __init__(self, edge_in_dim = None, edge_out_dim = None, edge_fc_dims = None,
                 dropout_p = None, use_batchnorm = None, use_leaky_relu=False):
        super(MLPGraphIndependent, self).__init__()

        self.edge_mlp = MLP(input_dim=edge_in_dim, fc_dims=list(edge_fc_dims) + [edge_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm, use_leaky_relu=use_leaky_relu)

    def forward(self, edge_feats = None):

        out_edge_feats = self.edge_mlp(edge_feats)

        return out_edge_feats

class MPGraph(nn.Module):
    """
    Main Model Class. Contains all the components of the model. It consists of of several networks:
    Edge Encoder: MLP encodes initial edge embedding
    Edge MLP: Updates edge embedding with Nodes i, j and Edge ij
    Node MLP: Updates node embedding with Node i and Edge ij
    """

    def __init__(self, model_params, time_aware_mp=False, use_leaky_relu=True):
        super(MPGraph, self).__init__()

        self.model_params = model_params
        if use_leaky_relu:
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()

        # Define Encoder Network
        encoder_feats_dict = model_params['encoder_feats_dict']
        self.encoder = MLPGraphIndependent(edge_in_dim=encoder_feats_dict['edge_in_dim'],
                                           edge_fc_dims=encoder_feats_dict['edge_fc_dims'],
                                           edge_out_dim=encoder_feats_dict['edge_out_dim'],
                                           use_leaky_relu=use_leaky_relu)

        # Define the 'Core' message passing network (i.e. node and edge update models)
        self.MPNet = self._build_core_MPNet(model_params=model_params, encoder_feats_dict=encoder_feats_dict, time_aware_mp=time_aware_mp, use_leaky_relu=use_leaky_relu)
        self.num_mp_steps = model_params['num_mp_steps']

    def _build_core_MPNet(self, model_params, encoder_feats_dict, time_aware_mp, use_leaky_relu=None):
        # Define an aggregation operator for nodes to 'gather' messages from incident edges
        node_agg_fn = model_params['node_agg_fn']
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'."

        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size) # out=source tensor, row=index to scatter, dim_size=same size as num nodes = x.0

        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]

        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        # Define all MLPs involved in the graph network
        self.reattach_initial_nodes = model_params['reattach_initial_nodes']
        self.reattach_initial_edges = model_params['reattach_initial_edges']

        edge_factor = 2 if self.reattach_initial_edges else 1
        node_factor = 2 if self.reattach_initial_nodes else 1

        edge_model_in_dim = node_factor * 2 * encoder_feats_dict['node_out_dim'] + edge_factor * encoder_feats_dict[
            'edge_out_dim'] # h_i, h_j, h_ij
        node_model_in_dim = node_factor * encoder_feats_dict['node_out_dim'] + encoder_feats_dict['edge_out_dim']

        # Define all MLPs used within the MPN
        edge_model_feats_dict = model_params['edge_model_feats_dict']
        node_model_feats_dict = model_params['node_model_feats_dict']

        edge_mlp = MLP(input_dim=edge_model_in_dim,
                       fc_dims=edge_model_feats_dict['fc_dims'],
                       dropout_p=edge_model_feats_dict['dropout_p'],
                       use_batchnorm=edge_model_feats_dict['use_batchnorm'],
                       use_leaky_relu=use_leaky_relu)

        if time_aware_mp:

            node_mlp = MLP(input_dim=2 * encoder_feats_dict['node_out_dim'],
                           fc_dims=node_model_feats_dict['fc_dims'],
                           dropout_p=node_model_feats_dict['dropout_p'],
                           use_batchnorm=node_model_feats_dict['use_batchnorm'],
                           use_leaky_relu=use_leaky_relu)

            flow_in_mlp = MLP(input_dim=node_model_in_dim,
                              fc_dims=node_model_feats_dict['fc_dims'],
                              dropout_p=None,
                              use_batchnorm=False,
                              use_leaky_relu=use_leaky_relu)

            flow_out_mlp = MLP(input_dim=node_model_in_dim,
                               fc_dims=node_model_feats_dict['fc_dims'],
                               dropout_p=None,
                               use_batchnorm=False,
                               use_leaky_relu=use_leaky_relu)

            # Define all MLPs used within the MPN
            return MetaLayer(edge_model=EdgeModel(edge_mlp=edge_mlp),
                             node_model=TimeAwareNodeModel(flow_in_mlp=flow_in_mlp, flow_out_mlp=flow_out_mlp,
                                                           node_mlp=node_mlp, node_agg_fn=node_agg_fn))

        else:

            node_mlp = MLP(input_dim=node_model_in_dim,
                           fc_dims=node_model_feats_dict['fc_dims'],
                           dropout_p=node_model_feats_dict['dropout_p'],
                           use_batchnorm=node_model_feats_dict['use_batchnorm'])

            # Define all MLPs used within the MPN
            return MetaLayer(edge_model=EdgeModel(edge_mlp=edge_mlp),
                             node_model=NodeModel(node_mlp=node_mlp, node_agg_fn=node_agg_fn))



    def forward(self, data):
        """
        Provides a fractional solution to the data association problem.
        First, node and edge features are independently encoded by the encoder network. Then, they are iteratively
        'combined' for a fixed number of steps via the Message Passing Network (self.MPNet).
        """

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encoding features step
        latent_edge_feats = self.encoder(edge_feats=edge_attr)
        latent_node_feats = self.relu(x)
        #latent_node_feats = x
        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats

        outputs = []

        # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
        # passing steps are classified in order to compute the loss.
        for step in range(1, self.num_mp_steps + 1):

            # Reattach the initially encoded embeddings before the update
            if self.reattach_initial_edges:
                latent_edge_feats = torch.cat((initial_edge_feats, latent_edge_feats), dim=1)
            if self.reattach_initial_nodes:
                latent_node_feats = torch.cat((initial_node_feats, latent_node_feats), dim=1)

            # Message Passing Step
            latent_node_feats, latent_edge_feats = self.MPNet(latent_node_feats, edge_index, latent_edge_feats)

            if step > 1: # For classifying edges at multiple message passing step times
                outputs.append(latent_edge_feats)

        return outputs#[latent_edge_feats]
