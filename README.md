# AMI Protocol Package
This package implements the AMI authentication and key generation protocol on a real system. 

# Directory structure (centralized setting)
    agent_info_files/
        agent0_info.txt
        agent1_info.txt
        agent2_info.txt
        
    agent_secrets_files/
        agent0_secrets.txt
        agent1_secrets.txt
        agent2_secrets.txt
        
    agent_neuralnet_models/
        agent0_neuralnet_models/
            agent1_statedict.pth
            agent2_statedict.pth
        agent1_neuralnet_models/
            agent1_statedict.pth
        agent2_neuralnet_models/
            agent2_statedict.pth
            
    secret_neuralnet_models/
        agent0_secret_models/
            agent1.pth
            agent2.pth
        agent1_secret_models/
            agent0.pth
        agent2_secret_models/
            agent0.pth

Note that `agent_neuralnet_models/`'s subdirectories may not be of group size, as not all agents use a neural network as their behavioural model.

# JSON File Format: Central server with same behavioral model for all clients
`agent0_info.txt`
    {
        "group_id": 0
        "unique_seed": 0
        "reuse\_agent\_model": True
        "models_list": [ -1,
                        {"group_id: 0", "unique_seed": 0, ....},
                        {"group_id: 0", "unique_seed": 0, ....}
                        ]
    }

# JSON File Format: Central server with multiple behavioral models
If the central server has multiple behavioural models for users, then it places a -1 at its own group index and lists the rest of the models at the index of the other agent. 
`agent0_info.txt`
    {
        "group_id": 0
        "unique_seed": 0
        "reuse\_agent\_model": False
        "models_list": [-1,
                        {"group_id: 1", "unique_seed": 1, ....},
                        {"group_id: 2", "unique_seed": 2, ....}
                        ]
    }

# JSON File Format: System agent in centralized setting
This agent may only use a single model to interact with the central server.

`agent1_info.txt`
    {
        "group_id": 1
        "unique_seed": 1
        "models_list": [-1,
                        {"group_id: 1", "unique_seed": 1, ....},
                        -1
                        ]
    }
    
# JSON File Format: System agent in decentralized setting
This agent uses one model for all others. The model is capable of conditioning on the actions of all other agents. Model type is either multi-tree or neural network.

`agent0_info.txt`
    {
        "group_id": 0
        "models_list": [-1,
                        {"group_id: 1", "unique_seed": 1, ....},
                        {"group_id: 2", "unique_seed": 2, ....},
                        ]
    }

# JSON File Format: Shared secrets 
`agent0_secrets.txt`
    {
        "unique_seed": 0
        "secrets_list": [-1,
                        {"group_id: 1", "unique_seed": 1, ....},
                        {"group_id: 2", "unique_seed": 2, ....}
                        ]
    }
    
# Running (centralized system)
    python run.py --group_size 3 --path_agent_files c_agent_info_files --path_shared_secrets c_shared_secret_files --path_neuralnet_models c_agent_neuralnet_models --path_secret_neuralnets c_secret_neuralnet_models --system_type centralized
# Running (decentralized system)
    python run.py --group_size 3 --path_agent_files d_agent_info_files --path_shared_secrets d_shared_secret_files --path_neuralnet_models d_agent_neuralnet_models --path_secret_neuralnets d_secret_neuralnet_models --system_type decentralized
