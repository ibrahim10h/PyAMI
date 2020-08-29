"""Class for initializing remote virtual machines for agent lifecycle."""
from file_handling import collect_local_dir_names, collect_neuralnets_from_dir
from subprocess import Popen, PIPE  
import os, time, json

# Define global variable
stricthostkeycheck_flag = "-o StrictHostKeyChecking=no"

class RemoteEnvsSetup(object):
    """Methods to set up remote machine envs."""

    def __init__(self, local_system, 
                        path_agent_files, path_shared_secrets,
                        path_neuralnet_models, path_secret_neuralnets):
        """Store local multi-agent system."""

        # Set variables
        self.local_system = local_system
        self.num_remote_envs = local_system.group_size # corresponds to num agents

        self.path_agent_files = path_agent_files
        self.path_shared_secrets = path_shared_secrets
        self.path_neuralnet_models = path_neuralnet_models
        self.path_secret_neuralnets = path_secret_neuralnets


    def run_ssh_subprocess(self, ipaddr, cmd):
        """Subprocess to ssh into specific machine and execute command."""
        
        ssh_subproc = Popen(["ssh", "-q", stricthostkeycheck_flag, 
                            str(ipaddr), cmd], stdout=PIPE, stderr=PIPE)

        # Wait for subprocess to complete
        ssh_subproc.wait()

        print("\nssh_subproc stdout: ", ssh_subproc.stdout.readlines())
        print("\nssh_subproc stderr: ", ssh_subproc.stderr.readlines())

    def begin_ssh_subprocess(self, ipaddr, cmd, file_obj_tuple):
        """Begin and return ssh subprocess without waiting for completion."""

        # Get variables
        (fout,ferr) = file_obj_tuple
        ssh_subproc = Popen(["ssh", "-q", stricthostkeycheck_flag, 
                            str(ipaddr), cmd], stdout=fout, stderr=ferr)#stdout=PIPE, stderr=PIPE)

        # Return subproc
        return ssh_subproc

    def run_scp_subprocess(self, files_list, dest):
        """
        Subprocess to scp transfer provided file(s) to remote machine.

        Args:
        file_list: list of file names for transfer
        dest: destination on remote machine, i.e. '12.34.56:dir1/'
        """

        args_list = ["scp", "-q", stricthostkeycheck_flag] + files_list[:] + [str(dest)]
        print("\nargs_list: ", args_list)

        scp_subproc = Popen(args_list, stdout=PIPE, stderr=PIPE)

        # Wait for subprocess to complete and return info
        (scp_stdout, scp_stderr) = scp_subproc.communicate()

        print("\nscp_stdout: ", scp_stdout)
        print("\nscp_stderr: ", scp_stderr)


    def send_required_files(self, root_dir_scripts, 
                                path_agent_files, path_shared_secrets, 
                                path_neuralnet_models, path_secret_neuralnets):
        """
        Send agent-specific scripts, info files, and agent models to all remote machines.

        Args:
        root_dir_scripts: list of local scripts in root dir
        """
        print("\nInside send_required_scripts!")

        for agent in self.local_system.agents_list:
            print("\nOn agent id: ", agent.group_id)

            # 1. Send local scripts in root dir to remote root dir

            dest = str(agent.ipaddr) + ':~'
            files_list = list(root_dir_scripts)
            print("1. files_list: ", files_list)

            # example dest: '12.34.56:~'

            self.run_scp_subprocess(files_list, dest)


            # 2. Send agent info file
            dest = str(agent.ipaddr) + ':' + path_agent_files + '/'
            files_list = [path_agent_files+'/agent'+str(agent.group_id)+'_info.txt']
            print("2. files_list: ", files_list)

            # example dest: '12.34.56:agent_info_files/'
            # example files_list: ['agent_info_files/agent0_info.txt']

            self.run_scp_subprocess(files_list, dest)

            
            # 3. Send agent secrets file
            dest = str(agent.ipaddr) + ':' + path_shared_secrets + '/'
            files_list = [path_shared_secrets+'/agent'+str(agent.group_id)+'_secrets.txt']
            print("3. files_list: ", files_list)

            # example dest: '12.34.56:shared_secret_files/'
            # example files_list: ['shared_secret_files/agent0_secrets.txt']

            self.run_scp_subprocess(files_list, dest)


            # 4. Send agent neural net behavioral models
            dest = str(agent.ipaddr) + ':' + path_neuralnet_models + '/' 

            # get behavioral model neural net .pth file names
            agent_model_subdir = 'agent'+str(agent.group_id)+'_neuralnet_models'

            # example path: 'agent_neuralnet_models/agent0_neuralnet_models'
            neuralnet_filenames = os.listdir(path_neuralnet_models+'/'+agent_model_subdir)
            neuralnet_filenames.sort() # i.e. ['agent1.pth', 'agent2.pth']

            files_list = []
            for neural_net in neuralnet_filenames:
                # example: 'agent_neuralnet_models/agent0_neuralnet_models/agent1.pth'
                files_list.append(path_neuralnet_models+'/'+agent_model_subdir+'/'+str(neural_net))
            print("4. files_list: ", files_list)

            # if files collected, send them
            if files_list: self.run_scp_subprocess(files_list, dest)


            # 5. Send agent neural net shared secret models
            dest = str(agent.ipaddr) + ':' + path_secret_neuralnets + '/' 

            # get shared secret neural net model .pth file names
            agent_secret_subdir = 'agent'+str(agent.group_id)+'_secret_models'

            # example path: 'secret_neuralnet_models/agent0_secret_models'
            neuralnet_filenames = os.listdir(path_secret_neuralnets+'/'+agent_secret_subdir)
            neuralnet_filenames.sort() # i.e. ['agent1.pth', 'agent2.pth']

            files_list = []
            for neural_net in neuralnet_filenames:
                # example: 'secret_neuralnet_models/agent0_secret_models/agent1.pth'
                files_list.append(path_secret_neuralnets+'/'+agent_secret_subdir+'/'+str(neural_net))
            print("5. files_list: ", files_list)

            # if files collected, send them
            if files_list: self.run_scp_subprocess(files_list, dest)            


    def remove_remote_directories(self):
        """Clear all dir structure on all remote machines."""
        print("\nInside remove_remote_directories!")

        for agent in self.local_system.agents_list:
            print("\nOn agent id: ", agent.group_id)

            # Build command to clear home directory
            cmd = 'rm -rf ./*'

            # Now run subprocess on remote env
            self.run_ssh_subprocess(agent.ipaddr, cmd)

    def build_remote_directories(self, path_agent_files, path_shared_secrets, 
                                    path_neuralnet_models, path_secret_neuralnets):
        """Replicate local dir structure on all remote machines."""
        print("\nInside build_remote_directories!")

        for agent in self.local_system.agents_list:
            print("\nOn agent id: ", agent.group_id)

            # Get list of dirs/subdirs names for this agent
            dirs_list = collect_local_dir_names(agent.group_id, 
                                                path_agent_files, 
                                                path_shared_secrets, 
                                                path_neuralnet_models, 
                                                path_secret_neuralnets)
            print("dirs_list: ", dirs_list)

            # Build list of commands to make all dirs
            cmd = ''
            for dir in dirs_list:
                cmd += 'mkdir ' + str(dir) + ';'

            cmd = cmd[:-1]  # remove trailing semicolon
            # Example: cmd='mkdir agent_info_files; mkdir shared_secret_files; ...'

            # Now run the subprocess on remote env
            self.run_ssh_subprocess(agent.ipaddr, cmd)

    def install_required_pylibs(self, install_script):
        """
        Run subprocess to load required py libraries on remote machines.

        Args:
        install_script: name of bash script for loading py libs
        """
        print("\nInside install_required_pylibs!")

        # Build cmd to run install script
        cmd = "chmod 700 " + str(install_script) + ";"
        cmd += " sudo ./" + str(install_script)

        # example cmd: "chmod 700 install.sh; sudo ./install.sh"
        
        # Run subprocess for all agent remote machines
        for agent in self.local_system.agents_list:
            print("\nOn agent id: ", agent.group_id)

            self.run_ssh_subprocess(agent.ipaddr, cmd)


    def run_agents_remotely(self, run_script):
        """
        Run agents simultaneously until all have completed their lifecycle.
        
        Return list of (fout,ferr) files for each agent subprocess.

        Args:
        run_script: Name of script to run agent lifecycle.
        """
        print("\nInside run_agents_remotely!")

        # Build cmd to run script, i.e. "python3 run_virtual_agent.py"
        cmd = "python3 " + str(run_script)

        # Store subprocess for each agent
        ssh_subproc_list = []
        output_obj_list = []

        for i in range(self.local_system.group_size):
            fout = open('file_out_' + str(i) + '.txt','w')
            ferr = open('file_err_' + str(i) + '.txt','w')
            output_obj_list.append((fout, ferr))

        # Begin each agent subprocess and allow them to run simultaneously
        for (agent, id) in zip(self.local_system.agents_list, range(self.local_system.group_size)):
            print("\nOn agent id: ", agent.group_id)

            # Start each subprocess
            (fout, ferr) = output_obj_list[id]
            ssh_subproc = self.begin_ssh_subprocess(agent.ipaddr, cmd, (fout,ferr))

            # Store subprocess for later
            ssh_subproc_list.append(ssh_subproc)

            # Buffer each subproc by 1 second to allow for setup
            time.sleep(1)

        # Print poll codes right when subprocs released
        print("\nPoll code 0: complete with no error.")
        print("Poll code 1: complete with error.")
        print("Poll code None: incomplete, still running.")

        print("\nCurrent poll codes: ", [p.poll() for p in ssh_subproc_list])
        
        # Wait until all simultaneous subprocesses have finished
        counter = 0
        while(1):
            all_subproc_finished = True

            for p in ssh_subproc_list:

                # poll() == 0 means: complete with no errors
                # poll() == 1 means: complete with errors
                # poll() == None means: still running

                if p.poll() == None:  
                    all_subproc_finished = False

            if all_subproc_finished == True:
                break

            counter += 1

        print("\nAll subprocesses finished: ", [p.poll() for p in ssh_subproc_list])

        # Close subprocess files
        for file_tuple in output_obj_list:
            (fout,ferr) = file_tuple
            fout.close()
            ferr.close()

        # Print subprocess return
        '''num_subproc = len(ssh_subproc_list)
        for (subproc, index) in zip(ssh_subproc_list, range(num_subproc)):

            print("\n\n\n\n------------------------subproc stdout: %d" % (index))
            for line in subproc.stdout.readlines():
                print(line)

            print("\nsubproc stderr:")
            for line in subproc.stderr.readlines():
                print(line)'''

        # Return (fout,ferr) files list
        return output_obj_list

    def create_settings_file(self, filename):
        """Create system settings file for remote system use."""

        # Create json dict with settings of local system
        sys_settings = {}

        sys_settings['system_type'] = self.local_system.system_type
        sys_settings['group_size'] = self.local_system.group_size
        sys_settings['max_interaction_length'] = self.local_system.max_interaction_length

        if self.local_system.system_type == 'centralized':
            sys_settings['central_server_group_id'] = self.local_system.central_server_group_id

        sys_settings['path_agent_files'] = self.path_agent_files
        sys_settings['path_shared_secrets'] = self.path_shared_secrets
        sys_settings['path_neuralnet_models'] = self.path_neuralnet_models
        sys_settings['path_secret_neuralnets'] = self.path_secret_neuralnets

        # Open file with specified file name and dump json dict
        with open(str(filename), "w") as outfile:
            json.dump(sys_settings, outfile, indent=4)

