#! /bin/bash

echo "About to install pylibs if required." 

python3 -c "from scipy.stats import skewnorm"	# command success stores 0 in '?'' variable, and failure stores 1 in '?' variable
skewnorm_installed=$?

python3 -c "import torch"
torch_installed=$?

# If either was failed command, perform update
if [ $skewnorm_installed -eq 1 ] || [ $torch_installed -eq 1 ]	 # bash comparison of numbers, using the equality syntax
then
	sudo apt-get -qq update							# -qq flag for 'no output except errors'
	sudo apt-get -qq install --assume-yes python3-pip
	sudo pip -q install numpy scipy
	sudo python3 -m pip -q install -U pip 				# update pip to for matplotlib as it requires later version
	sudo python3 -m pip -q install -U torch
	#sudo apt-get -qq install --assume-yes python-tk
else
	echo "All Python libs already installed"
fi