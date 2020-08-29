"""Functions to help with file collection."""
import os, json, torch


def collect_json_from_dir(dir_path, num_files):
    """Return json objects from provided directory."""
    assert os.path.isdir(dir_path), "Directory does not exist!"

    # Get sorted filenames from directory
    file_names_list = os.listdir(dir_path)
    file_names_list.sort()
    print("file_names_list: ", file_names_list)


    # Ensure expected number of files were collected
    assert len(file_names_list) == num_files, "Incorrect number of files in dir!"

    # Load json object from each file in dir
    json_object_list = []
    for filename in file_names_list:
        print("\nOn filename: ", filename)

        with open(str(dir_path+'/'+filename), 'r') as f:
            data = json.load(f)
            json_object_list.append(data)
            print("json object: ", data.items())

    # Return json object list
    return json_object_list

def collect_neuralnets_from_subdir(subdir_path, group_size):
    """
    Return list of  available pytorch neural nets (.pth object) in subdir.
    List element is -1 if statedict file not listed in directory.
    Note that number of neural net models may not match group size.

    Args:
    subdir_path: full path from cwd to subdirectory
                i.e. 'agent_neuralnet_models/agent0_neuralnet_models'
    """

    assert os.path.isdir(subdir_path), "Subdirectory does not exist!"

    # Define variables
    agent_statedict_list = [-1 for _ in range(group_size)]

    # Get sorted filenames from directory
    file_names_list = os.listdir(subdir_path)
    file_names_list.sort()
    #print("file_names_list: ", file_names_list)

    # If agent neural net found, load state dict and place in list at group id
    for k in range(group_size):
        # Example of .pth file name: 'agent2.pth'
        model_name = 'agent' + str(k) + '.pth' 
        
        if model_name in file_names_list:
            agent_statedict_list[k] = torch.load(subdir_path + '/' + model_name)

    # Return the list of statedicts under this agent's subdirectory
    return agent_statedict_list

def collect_neuralnets_from_dir(dir_path, subdirs, group_size):
    """
    Return list of available pytorch neural nets (.pth object) using specified dir.
    List element is -1 if statedict file not listed in directory.
    Note that number of neural net models may not match group size.

    Args:
    dir_path: directory which holds subdirectories for each agent
    subdirs: list of agent-specific subdirectory names

    Example directory structure:

        ./agent_neuralnet_models/

            agent0_neuralnet_models/
                - # skip myself
                - agent1.pth
                - agent2.pth

            agent1_neuralnet_models/
                - agent0.pth

            agent2_neuralnet_models/
                - agent0.pth
    """
    assert os.path.isdir(dir_path), "Directory does not exist!"

    # Define variables
    neuralnet_statedict_list = [-1 for _ in range(group_size)]

    # Go through subdirectories for each agent
    for l in range(group_size):

        # Define variables
        agent_statedict_list = [-1 for _ in range(group_size)]

        # Example: agent_neuralnet_models/agent0_neuralnet_models
        subdir_path = dir_path + '/' + subdirs[l]
        #print("\nsubdir_path: ", subdir_path)

        assert os.path.isdir(subdir_path), "Subdirectory does not exist!"

        # Get sorted filenames from directory
        file_names_list = os.listdir(subdir_path)
        file_names_list.sort()
        #print("file_names_list: ", file_names_list)

        # If agent neural net found, load state dict and place in list at group id
        for k in range(group_size):
            # Example of .pth file name: 'agent2.pth'
            model_name = 'agent' + str(k) + '.pth' 
            
            if model_name in file_names_list:
                agent_statedict_list[k] = torch.load(subdir_path + '/' + model_name)

        #print("agent_statedict_list: ", agent_statedict_list)

        # Update list for all agent
        neuralnet_statedict_list[l] = agent_statedict_list

    #print("neuralnet_statedict_list: ", neuralnet_statedict_list)

    return (neuralnet_statedict_list, file_names_list)

def collect_local_dir_names(agent_id, path_agent_files, path_shared_secrets, 
                                    path_neuralnet_models, path_secret_neuralnets):
    """Build list of all local dir/subdir names for specified agent id."""

    dirs_list = []

    # Append all required local dir/subdir names
    dirs_list.append(path_agent_files)
    dirs_list.append(path_shared_secrets)
    
    dirs_list.append(path_neuralnet_models)
    # Example: 'agent_neuralnet_models/agent0_neuralnet_models'
    dirs_list.append(path_neuralnet_models+'/agent'+str(agent_id)+'_neuralnet_models')


    dirs_list.append(path_secret_neuralnets)
    # Example: 'secret_neuralnet_models/agent0_secret_models'
    dirs_list.append(path_secret_neuralnets+'/agent'+str(agent_id)+'_secret_models')

    return dirs_list

def update_json_file(filepath, key, value):
    """
    Update json file with dict key and value.

    Args:
    filepath: full json file path (i.e. 'agent_info_files/agent0_info.txt')
    """

    assert os.path.exists(filepath), "Filepath %s doesn't exist!" % filepath
    
    # Load json dict from json file
    with open(str(filepath), "r") as infile:
        data = json.load(infile)

    infile.close()

    # Update json dictionary if (key,value) not already there
    data[str(key)] = value

    # Upload updated dict to json file
    with open(str(filepath), "w") as outfile:
        json.dump(data, outfile, indent=4)
