# PyAMI: System Prototype
This repository is the official implementation of PyAMI for the paper: 
**Quantum-Secure Authentication and Key Agreement via Abstract Multi-Agent Interaction**

# Description
The PyAMI project implements the AMI authentication and key generation protocol in a multi-agent system consisting of multiple virtual machines. 

# Requirements
### Local Requirements

* [Python 3](https://www.python.org/downloads/) 

### Remote Requirements
If multi-agent system specified to run on remote machines, `run.py` will use [pip](https://pypi.org/project/pip/) to install the latest version of the following packages on provided machines:
* [Scipy](https://www.scipy.org/install.html)
* [PyTorch](https://pytorch.org/get-started/locally/)

# Start-up Guide
The following sections describe how to get up and running with the supported multi-agent system deployments:

1. Local directory structure.
2. JSON file format for agents' info.
3. Run commands for supported deployments.

### 1. Local directory structure 
The `run.py` requires a specific directory/subdirectory structure on the local machine for agent information:

    c_pdt_agent_info_files/
        agent0_info.txt
        agent1_info.txt
        agent2_info.txt
        
    c_pdt_agent_secrets_files/
        agent0_secrets.txt
        agent1_secrets.txt
        agent2_secrets.txt
        
    c_pdt_agent_neuralnet_models/
        agent0_neuralnet_models/
            agent1.pth
            agent2.pth
        agent1_neuralnet_models/
            agent1.pth
        agent2_neuralnet_models/
            agent2.pth
            
    c_pdt_secret_neuralnet_models/
        agent0_secret_models/
            agent1.pth
            agent2.pth
        agent1_secret_models/
            agent0.pth
        agent2_secret_models/
            agent0.pth

**Note:** `c_pdt_agent_neuralnet_models/`'s subdirectories may contain less than *group size* neural net models, as agents may or may not use a neural net as their behavioral model.

The same is true for `c_pdt_secret_neuralnet_models/`.

# JSON File Format: Central server with same behavioral model for all clients
`agent0_info.txt`
```json
{
    "group_id": 0
    "unique_seed": 0
    "reuse_agent_model": True
    "models_list": [ -1,
                    {"group_id: 0", "unique_seed": 0, ....},
                    {"group_id: 0", "unique_seed": 0, ....}
                    ]
}
```
# JSON File Format: Central server with multiple behavioral models
If the central server has multiple behavioural models for users, then it places a -1 at its own group index and lists the rest of the models at the index of the other agent. 
`agent0_info.txt`
```json
{
    "group_id": 0
    "unique_seed": 0
    "reuse_agent_model": False
    "models_list": [-1,
                    {"group_id: 1", "unique_seed": 1, ....},
                    {"group_id: 2", "unique_seed": 2, ....}
                    ]
}
```

# JSON File Format: System agent in centralized setting
This agent may only use a single model to interact with the central server.
`agent1_info.txt`
```json
{
    "group_id": 1
    "unique_seed": 1
    "models_list": [-1,
                    {"group_id: 1", "unique_seed": 1, ....},
                    -1
                    ]
}
```
# JSON File Format: System agent in decentralized setting
This agent uses one model for all others. The model is capable of conditioning on the actions of all other agents. Model type is either multi-tree or neural network.

`agent0_info.txt`
```json
{
    "group_id": 0
    "models_list": [-1,
                    {"group_id: 1", "unique_seed": 1, ....},
                    {"group_id: 2", "unique_seed": 2, ....},
                    ]
}
```
# JSON File Format: Shared secrets 
`agent0_secrets.txt`
```json
{
    "unique_seed": 0
    "secrets_list": [-1,
                    {"group_id: 1", "unique_seed": 1, ....},
                    {"group_id: 2", "unique_seed": 2, ....}
                    ]
}
```
    
# Running (centralized system)
Agent json info files (`agent0_info.txt`, etc.) must contain IP address of live remote machines agents will run on, in "ipaddr" field as a string. 

Example:
```json
"ipaddr": "34.82.122.236"
```

Then run with:
```python
python run.py --group_size 3 --path_agent_files c_pdt_agent_info_files --path_shared_secrets c_pdt_shared_secret_files --path_neuralnet_models c_pdt_agent_neuralnet_models --path_secret_neuralnets c_pdt_secret_neuralnet_models --system_type centralized
```

# Running (decentralized system)
Provide IP addresses as in centralized case, then run with:

```python
python run.py --group_size 3 --path_agent_files d_pdt_agent_info_files --path_shared_secrets d_pdt_shared_secret_files --path_neuralnet_models d_pdt_agent_neuralnet_models --path_secret_neuralnets d_pdt_secret_neuralnet_models --system_type decentralized
```
    
# Results

**Agent output files:**
Results for authentication and symmetric session keys are written to individual output files for each agent, and stored on the local machine. 
```
file_out_0.txt
file_err_0.txt
```

**Sample authentication results (centralized setting):**
A central server successfully authenticates agents a1 and a2:
```
Agent auth results:  [-1, True, True]
```

**Sample session key results (centralized setting):**
A central server creates group session key for two agents:
```
group_key: 0.302,0.390,0.328,0.329,0.329*0.355,0.330,0.433,0.318,0.420*0.964,0.406,0.786,0.881,0.900*0.533,0.676,0.844,0.844,0.972
```