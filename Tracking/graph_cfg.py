# Graph Setup

def init_graph_cfg(node_in_size=16):
    '''
    Graph Neural Network setup
    '''
    graph_cfg = {
        'undirected_graph': True,
        'use_time_aware_mp': False,
        'use_leaky_relu': True,
        'max_frame_dist': 5,
        'num_mp_steps': 4,
        'node_agg_fn': 'mean',
        'reattach_initial_nodes': False,
        'reattach_initial_edges': True,
        'encoder_feats_dict': {
            'edge_in_dim': 8,
            'edge_fc_dims': [12],
            'edge_out_dim': 12,
            'node_out_dim': node_in_size,
            'dropout_p': None,
            'use_batchnorm': False,
        },
        'edge_model_feats_dict': {
            'fc_dims': [32, 12],
            'dropout_p': None,
            'use_batchnorm': False,
        },
        'node_model_feats_dict': {
            'fc_dims': [20, node_in_size],
            'dropout_p': None,
            'use_batchnorm': False,
        },
    }
    return graph_cfg
