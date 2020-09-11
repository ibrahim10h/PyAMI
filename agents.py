"""General agent classes."""

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import random
from itertools import product # cartesian product

#from graphviz import Digraph

#from generate_histories_vGAN_and_recGAN import generateIncrementalHistory_randomPDT_randomPDT
from device_settings import device


class Agent(object):
	"""Abstract agent class."""

	def __init__(self, type):  # noqa

		# Type can be 'PDT', 'RNN', or 'MLP'
		self._type = type

	def act(self):
		"""
		Select action.

		All agents must implement an act() function.
		"""
		raise NotImplementedError("Agent must implement act method.")

	def reset(self):
		"""
		Initialize agent for interaction.

		All agents must implement a reset() function.
		"""
		raise NotImplementedError("Agent must implement reset method.")

	def dist(self, inputs, **kwargs):
		"""
		Get agent's distribution over actions given inputs.

		All agents must implement this function.
		"""
		raise NotImplementedError("Agent must implement dist method.")


class PDTNode(object):
	"""Node for recursive PDT structure."""
	
	def __init__(self, n_actions, subtree, index=0):

		self._visits = 0   # track number of times node is traversed over (and sampled from)
		self._uses = 0     # track number of times node is used to sample from
		self._label = None # label for visualizing PDT as graph

		self._children = []
		self._index = index
		if len(subtree) == n_actions + 1:
			self._distribution = subtree[0]
			for i in range(1, n_actions + 1):
				self._children.append(PDTNode(n_actions, subtree[i], index=0))
		else:
			self._distribution = subtree

	def __str__(self):
		"""
		String representation of PDTNode.

		Returns string of distribution followed by string rep of children.
		"""
		strrep = str(self._distribution) + "\n"
		for child in self._children:
			strrep += str(child)
		return strrep

	def sample(self):
		"""Sample node distribution for [1,...,n_actions]."""

		return Categorical(self._distribution).sample().data.item() + 1

	def has_children(self):
		"""Test if non-empty list of children."""
		return len(self._children) > 0


	def diff(self, other):
		"""KL between other pdt node and self.
		
		In MLE Attack, the estimated tree often has probabilities of 0.0, which mess up KL-div calculation.
		Replace all probabilities of 0.0 with a small value near zero.
		"""

		def replaceZerosInTensor(input, num_actions, zero_approx=1e-12):
			"""
			Convert all 0.0's in tensor to a zero-approximate value, while preserving any gradient. 

			Args:
			input: tensor of dimension (num_actions)
			num_actions: size of actions space and length of t
			zero_approx: float value such as: 1e-12
			"""
			if input.shape[0] != num_actions:
				exit("Exiting in replaceZeroInTensor(): input.shape must be same as num_actions.")

			# Create buffer of same length as input
			zero_buffer = torch.zeros(num_actions)
			#print("initial zero_buffer: ", zero_buffer)

			for x in range(num_actions):
				if input[x].data.item() == 0:
					zero_buffer[x] = zero_approx

			#print("modified zero_buffer: ", zero_buffer)

			# Add input to zero_buffer to replace all zeros with zero-approximate value
			input = input + zero_buffer

			#print("modified input: ", input)
			return input

		# Replace any probabilities of value 0.0 with small value of 1e-12
		if 0 in other._distribution:
			small_value = 1e-12
			n_actions = other._distribution.shape[0]
			q = replaceZerosInTensor(other._distribution,n_actions,small_value)
		else:
			q = other._distribution

		p = self._distribution
		kl_div = (p * (p / q).log()).sum()

		if torch.isinf(kl_div):
			exit("Exiting: kl divergence was inf in PDTNode.diff().")
		else:
			return kl_div


class PDTAgent(Agent):
	"""Agent that selects actions with pdt."""

	def __init__(self, n_actions, depth, action_probs, window=1):  # noqa
		self._type = "pdt"
		self._max_node_visits = 0  # track highest number of traversals to/over any node
		self._max_node_uses = 0    # track highest number of uses (sampling) for any node

		# print('Creating pdt with depth %d and window %d' % (depth, window))
		self._num_actions = n_actions
		self._depth = depth
		self._window = window
		self._root = PDTNode(n_actions, action_probs, index=0)
		self._current_node = self._root
		self._num_nodes = sum([n_actions ** d for d in range(depth + 1)])

		# State records the history used to traverse the tree.
		# Only remembers fixed history of up to a given size.
		self._state = deque(maxlen=window)

		# Record each history of actions and nodal dists
		self._interaction_history = []


	def __str__(self):  # noqa
		return str(self._root)

	def record_node_visit(self, node):
		#print("\nInside record_node_visit with node: ", node)
		"""
		Increment visit count if node is traversed to or traversed over.
		
		Also update self _max_node_visits if need be.
		"""
		#print("node: ", node)

		node._visits += 1 
		self.update_max_node_visits(node._visits)

	def record_node_use(self, node):
		"""
		Increment use count if node is sampled from.
		Also update self _max_node_uses if need be. """

		node._uses += 1

		# Check if _max_node_uses needs to be updated
		self.update_max_node_uses(node._uses)

	def update_max_node_visits(self,updated_node_visits):
		"""Update record of max node visits if need be."""

		if updated_node_visits > self._max_node_visits:
			self._max_node_visits = updated_node_visits

	def update_max_node_uses(self, updated_node_uses):
		"""Update record of max node uses if need be."""

		if updated_node_uses > self._max_node_uses:
			self._max_node_uses = updated_node_uses

	def reset(self):
		"""Reset before interaction."""
		self._state.clear()

	def act(self, t, inputs=[]):
		#print("\nInside act() with t: ", t, " and inputs: ", inputs)
		"""
		Select action given time and inputs.

		Args:
		t: int, current time-step
		inputs: list of ints, past decisions to condtion action selection on.

		Returns:
		A tuple as (sampled action, nodal distribution)
		"""
		assert t >= 0, "act() requires t >= 0"

		# Update state history
		if len(inputs) == 0:
			# If no input then clear state of model.
			self._state.clear()

			# Clear existing interaction history
			self._interaction_history.clear()
		else:
			self._state.extend(inputs)

		# Traverse tree based on state
		final_node = self.traverse_tree_nodes(self._state)

		sampled_action = final_node.sample()

		# Update record of uses for this node
		self.record_node_use(final_node)

		# Update current interaction history with tuple
		self._interaction_history.append((sampled_action, final_node._distribution))

		return (sampled_action, final_node._distribution)

	def traverse_tree_nodes(self, inputs):
		#print("\nInside traverse_tree_nodes! with inputs: ", inputs)
		"""
		Traverse pdt to a leaf node using given history.
		Record traversals over and to nodes. 

		Args:
		inputs: list of ints, past decisions to condition action selection on.

		Returns:
		A torch.Tensor
		"""
		# Start with the main tree
		current_subtree = self._root
		self.record_node_visit(self._root)

		if len(inputs) == 0:
			return self._root
		# print(inputs, self._root._children)
		for current_decision in inputs:

			# in case the decision is still in the form [2] instead of just '2'
			if type(current_decision) == list:
				current_decision = current_decision[0]
			# If history exceeds depth of tree then we wrap back around to the root.
			if current_subtree.has_children():
				current_subtree = current_subtree._children[current_decision - 1]
			else:
				self.record_node_visit(self._root)
				current_subtree = self._root._children[current_decision-1]  # if at a leaf, traverse from root

			# Record the traversal to or over this node
			self.record_node_visit(current_subtree)

		self._current_node = current_subtree
		return current_subtree  # we have traversed down, so return the subtree we've come to

	def dist(self, t, inputs, **kwargs):
		"""Get distribution over actions given inputs."""

		# Update state history
		if len(inputs) == 0:
			# If no input then clear state of model.
			self._state.clear()
		else:
			self._state.extend(inputs)
			
		# Traverse tree based on state
		leaf_node = self.traverse_tree_nodes(self._state)
		return leaf_node._distribution

	def diff(self, other_pdt):
		"""Compute KL between other pdt and self."""
		nodes = [self._root]
		other_nodes = [other_pdt._root]
		total = 0.0
		ct = 0
		while len(nodes) > 0:
			node = nodes.pop()
			other = other_nodes.pop()
			ct += 1
			total += node.diff(other)
			if node.has_children():
				nodes.extend(node._children)
				other_nodes.extend(other._children)

		return total / ct

	def current_as_vector(self):
		"""
		Return last node to be traversed to as 1-hot vector.

		Example: current node has index 1 and tree has 4 nodes
		>>> self.current_as_vector()
		np.array([0, 1, 0, 0])

		Returns:
		np.array with shape (num_nodes,)
		"""
		index = self._current_node._index
		vec = np.zeros(self._num_nodes)
		vec[index] = 1.0
		return vec

	def build_pdt_graph(self,node,dot,labels,by_visits=True):
		"""
		Construct Digraph of PDT with nodes colored for illumination (by uses or visits).
		
		Returns filled Digraph dot object

		Args: 
		node:   root node of self
		dot:    Digraph object from graphviz library
		labels: list of characters, equal in length to total number of nodes in PDT
		by_visits: boolean is True to illuminate nodes by visits, or False for uses
		"""

		if node.has_children():
			my_label = str(labels.pop(0)) if node._label is None else node._label

			# Create node and label for self
			if node._label is None:

				if by_visits:
					node_text = str(tensorToPylist(node._distribution)) + "\n" + str(node._visits) + " visits"
					node_color = self.illumination_color(node._visits,self._max_node_visits)
				else:
					node_text = str(tensorToPylist(node._distribution)) + "\n" + str(node._uses) + " uses"
					node_color = self.illumination_color(node._uses,self._max_node_uses)
				
				dot.node(my_label, node_text, color=node_color)
				node._label = my_label

			# Create node and label for all children
			for child in node._children:
				child_label = str(labels.pop(0)) if child._label is None else child._label

				if child._label is None:

					if by_visits:
						child_text = str(tensorToPylist(child._distribution)) + "\n" + str(child._visits) + " visits"
						child_color = self.illumination_color(child._visits,self._max_node_visits)
					else:
						child_text = str(tensorToPylist(child._distribution)) + "\n" + str(child._uses) + " uses"
						child_color = self.illumination_color(child._uses,self._max_node_uses)
				   
					dot.node(child_label, child_text, color=child_color)
					child._label = child_label
				
				# Create edge between self and child
				dot.edge(my_label, child_label)

				# Recursive call to check for grandchildren
				dot = self.build_pdt_graph(child,dot,labels,by_visits)

				# If child has depth of only 1 and on last child then return dot
				if node._children[-1] == child: 
					return dot

		else:
			return dot

	def illumination_color(self, value, max_value):
		"""
		Provide a color in yellow-orange-red spectrum from ratio: value/max_value.

		Args
		value: either a node's visits count or uses count
		max_value: either _max_node_visits or _max_node_uses of self
		"""

		# Get each Red, Green, Blue value as 2-digit hex string

		# Red color is fixed at highest intensity of 255
		red_RGB = hex(255).replace('0x', '') # get 'ff' from '0xff'

		# Blue color fixed at zero intensity
		blue_RGB = '00' # from '0x00'


		# Green color determines spectrum of yellow(green=255) - orange - red(green=0)

		# Example of RGB spectrum for Red
		#
		# Pure Red      | Pure Yellow
		#   
		# Red = 255     | Red = 255
		# Blue = 0      | Blue = 0
		# Green = 0     | Green = 255

		# Determine how much green we need
		# For example, (1-95/100) is 0.05 so its close to red for high illumination
		ratio_of_green = 1-float(value/max_value) 

		# Compute value for Green we need
		# For example, (0.05*255) is Green=12, which is close to red
		green_RGB = int(ratio_of_green*255)

		# Convert to hex and strip leading '0x'
		green_RGB = hex(green_RGB).replace('0x', '')

		# Pad Green hex value if less than 2 digits
		if len(green_RGB) < 2:
			green_RGB += '0'  

		# Combine into three 2-digit hex characters for graphviz
		RGB_hex_str = '#' + red_RGB + green_RGB + blue_RGB   # ie '#ff0000'

		assert len(RGB_hex_str) == 7, "Exiting in node_color(): len(RGB_hex_str) != 7."

		return RGB_hex_str


	@property
	def temperature(self):
		"""Access temperature property."""
		return self._temperature

	@temperature.setter
	def temperature(self, t):
		assert t >= 0, "Temperature must be non-negative."
		self._temperature = t

	def get_properties(self):
		"""Return properties needed to create random pdt from same distribution."""
		return self._num_actions, self._depth, self._window, self._temperature

class PDTMultiAgent(Agent):
	"""Agent that selects actions with multi-tree, a 
	pdt which interacts with multiple agents at once. """

	def __init__(self, n_actions, n_other_agents, depth, action_probs, window=1):  # noqa
		self._type = "multitree"
		self._max_node_visits = 0  # track highest number of traversals to/over any node
		self._max_node_uses = 0    # track highest number of uses (sampling) for any node

		# print('Creating pdt with depth %d and window %d' % (depth, window))
		self._num_actions = n_actions # size of action space
		self._num_other_agents = n_other_agents # number of agents which interact with self
		self._n_branches = (n_actions ** n_other_agents) # branches per node

		self._depth = depth
		self._window = window
		self._root = PDTNode(n_actions, action_probs, index=0)
		self._current_node = self._root
		self._num_nodes = sum([n_actions ** d for d in range(depth + 1)])

		# Set ordered branch codes used for multi-tree traversals
		self._ordered_branch_codes = self.create_branch_codes(n_actions, n_other_agents)

		# State records the history used to traverse the tree.
		# Only remembers fixed history of up to a given size.
		self._state = deque(maxlen=window)

		# Record each history of actions and nodal dists
		self._interaction_history = []

	def __str__(self):  # noqa
		return str(self._root)

	def record_node_visit(self, node):
		#print("\nInside record_node_visit with node: ", node)
		"""
		Increment visit count if node is traversed to or traversed over.
		
		Also update self _max_node_visits if need be.
		"""
		#print("node: ", node)

		node._visits += 1 
		self.update_max_node_visits(node._visits)

	def record_node_use(self, node):
		"""
		Increment use count if node is sampled from.
		Also update self _max_node_uses if need be. """

		node._uses += 1

		# Check if _max_node_uses needs to be updated
		self.update_max_node_uses(node._uses)

	def update_max_node_visits(self,updated_node_visits):
		"""Update record of max node visits if need be."""

		if updated_node_visits > self._max_node_visits:
			self._max_node_visits = updated_node_visits

	def update_max_node_uses(self, updated_node_uses):
		"""Update record of max node uses if need be."""

		if updated_node_uses > self._max_node_uses:
			self._max_node_uses = updated_node_uses


	def create_branch_codes(self, n_actions, n_other_agents):
		"""Set ordered list of input actions which map to multi-tree branches.

		Args:
		n_actions: size of the action space, i.e. 3
		n_other_agents: number of other agents which interact with self, i.e. 2

		Return: list of permutations 
			i.e. [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
		"""
		
		# Create action space
		action_space_list = range(1,n_actions+1) # i.e. [1,2,3]

		# Get ordered list of cartesian product of action space with itself
		action_space_sets = [action_space_list for _ in range(n_other_agents)]

		# If 3 other agents, action_space_sets = [[1,2,3],[1,2,3],[1,2,3]]
		ordered_branch_codes = list(product(*action_space_sets))

		""" 
		Example with 3 other agents, action space list = [1,2,3]:

		[(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3), 
		(1, 3, 1), (1, 3, 2), (1, 3, 3), (2, 1, 1), (2, 1, 2), (2, 1, 3), 
		(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 3, 1), (2, 3, 2), (2, 3, 3), 
		(3, 1, 1), (3, 1, 2), (3, 1, 3), (3, 2, 1), (3, 2, 2), (3, 2, 3), 
		(3, 3, 1), (3, 3, 2), (3, 3, 3)]
		"""

		# Return ordered list
		return ordered_branch_codes

	def convert_to_branch(self, traversal_actions=[]):
		"""Convert provided ordered actions to a branching decision.

		Args:
		traversal_actions: list of ordered ints, i.e. [3,2]

		Return integer action indicating which branch to take.
		"""
		
		# Convert traversal_actions to tuple
		perm_tuple = tuple(traversal_actions[:]) # i.e. (3,2)
		#print("perm_tuple: ", perm_tuple)

		# Map to a permutation in multi-tree's branch codes
		branch_index = self._ordered_branch_codes.index(perm_tuple)

		# Example branch codes: [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

		# Adjust as multi-tree inputs begin with 1, not 0
		branch_decision = branch_index + 1

		# Return branching decision
		return branch_decision


	def reset(self):
		"""Reset before interaction."""
		self._state.clear()

	def act(self, t, traversal_actions=[]):
		#print("\nInside act() with t: ", t, " and traversal_actions: ", traversal_actions)
		"""
		Convert provided actions to branch code input.
		Select action given time and  branch code input.

		Args:
		t: int, current time-step
		traversal_actions: list of ints, actions from other agents.
						Empty list at t == 0.
						Must be converted to branch code.

		Returns:
		A tuple as (sampled action, nodal distribution)
		"""
		assert t >= 0, "act() requires t >= 0"

		
		# Update state history
		if len(traversal_actions) == 0: # empty at t == 0
			# If no input then clear state of model.
			self._state.clear()

			# Clear existing interaction history
			self._interaction_history.clear()
		else:
			# Convert inputs to branch code
			inputs = [self.convert_to_branch(traversal_actions)] # i.e. [5]

			self._state.extend(inputs)

		# Traverse tree based on state
		final_node = self.traverse_tree_nodes(self._state)

		sampled_action = final_node.sample()

		# Update record of uses for this node
		self.record_node_use(final_node)

		# Update current interaction history with tuple
		self._interaction_history.append((sampled_action, final_node._distribution))

		return (sampled_action, final_node._distribution)

	def traverse_tree_nodes(self, inputs):
		#print("\nInside traverse_tree_nodes! with inputs: ", inputs)
		"""
		Traverse pdt to a leaf node using given history.
		Record traversals over and to nodes. 

		Args:
		inputs: list of ints, past decisions to condition action selection on.

		Returns:
		A torch.Tensor
		"""
		# Start with the main tree
		current_subtree = self._root
		self.record_node_visit(self._root)

		if len(inputs) == 0:
			return self._root
		# print(inputs, self._root._children)
		for current_decision in inputs:

			# in case the decision is still in the form [2] instead of just '2'
			if type(current_decision) == list:
				current_decision = current_decision[0]
			# If history exceeds depth of tree then we wrap back around to the root.
			if current_subtree.has_children():
				current_subtree = current_subtree._children[current_decision - 1]
			else:
				self.record_node_visit(self._root)
				current_subtree = self._root._children[current_decision-1]  # if at a leaf, traverse from root

			# Record the traversal to or over this node
			self.record_node_visit(current_subtree)

		self._current_node = current_subtree
		return current_subtree  # we have traversed down, so return the subtree we've come to

	@property
	def temperature(self):
		"""Access temperature property."""
		return self._temperature

	@temperature.setter
	def temperature(self, t):
		assert t >= 0, "Temperature must be non-negative."
		self._temperature = t

	def get_properties(self):
		"""Return properties needed to create random pdt from same distribution."""
		return self._num_actions, self._depth, self._window, self._temperature


def generate_nodal_dist(n_children, temperature, small_value):
	"""
	Create a nodal probability distribution affected by temperature. 
	Ensures no all values in dist > small_value.

	Args:
	small_value: python float, ie 1e-20
	"""

	while(1):

		# Generate a distribution affected by temperature
		nodal_dist = F.softmax(torch.rand(n_children) / temperature, dim=0).to(device)
		#print("nodal_dist: ", nodal_dist, " : ", (nodal_dist <= small_value))
	
		# If all items in distribution are > small_value, return the distribution
		if (nodal_dist > small_value).all().data.item():
			break

	return nodal_dist

'''# Uncomment to test generate_nodal_dist
generate_nodal_dist(n_children=3,temperature=0.01, small_value = 1e-20)'''

def create_random_multitree(depth, n_children, n_actions, index=0, temperature=1.0):
	"""
	Generate random multi-agent PDT with specified depth and branching factor.
	Note that size of nodal dists may not match number of children per node.

	Args:
	depth: int, depth of tree.
	n_children: int, number of children at each node.
	n_actions:, int, size of the action space, i.e. 3
	index: int, id for root node.
	temperature: float, controls entropy of distribution
		> 1.0 is more random, < 1.0 is more deterministic.

	Returns:
	root: PDTNode, root of PDT of specified depth and width.
	"""
	nodal_dist = generate_nodal_dist(n_actions, temperature, small_value = 1e-20)	
	root = PDTNode(n_children, nodal_dist)

	# Recursively create each of children if required.
	if depth > 0:
		n_in_subtree = sum([n_children ** i for i in range(depth)])
		for j in range(n_children):
			ind = index + 1 + (n_in_subtree) * j
			root._children.append(
				create_random_multitree(depth - 1, n_children, n_actions, 
					index=ind, temperature=temperature)
			)
	return root


def create_random_pdt(depth, n_children, index=0, temperature=1.0):
	"""
	Generate random PDT with specified depth and branching factor.

	Args:
	depth: int, depth of tree.
	n_children: int, number of children at each node.
	index: int, id for root node.
	temperature: float, controls entropy of distribution
		> 1.0 is more random, < 1.0 is more deterministic.

	Returns:
	root: PDTNode, root of PDT of specified depth and width.
	"""

	# Version 1: Create PDTNode with no children as root
	#root = PDTNode(n_children, F.softmax(torch.rand(n_children) / temperature, dim=0).to(device))

	# Version 2: Creat PDTNode with no children as root (ensure no very small values)
	nodal_dist = generate_nodal_dist(n_children, temperature, small_value = 1e-20)	
	root = PDTNode(n_children, nodal_dist)

	# Recursively create each of children if required.
	if depth > 0:
		n_in_subtree = sum([n_children ** i for i in range(depth)])
		for j in range(n_children):
			ind = index + 1 + (n_in_subtree) * j
			root._children.append(
				create_random_pdt(
					depth - 1, n_children, index=ind, temperature=temperature
				)
			)
	return root


def create_random_pdt_agent(depth, n_actions, window=1, temperature=1.0):
	"""
	Generate random PDT agent with specified depth and branching factor.

	Args:
	depth: int, depth of tree.
	n_actions: int, number of children at each node.
	temperature: float, controls entropy of distribution
		> 1.0 is more random, < 1.0 is more deterministic.

	Returns:
	agent: PDTAgent, PDTAgent using PDT with specified properties.
	"""
	agent = PDTAgent(n_actions, depth, [], window)
	root = create_random_pdt(depth, n_actions, temperature=temperature)
	agent._root = root

	# Set temperature to be able to create similar models if loaded from file.
	agent.temperature = temperature
	return agent

def create_random_pdt_multiagent(depth, n_actions, n_other_agents,
								window=1, temperature=1.0):
	"""
	Generate random multi-tree PDT agent with specific properties.
	Capable of interacting with multiple other agents.

	Args:
	depth: int, depth of tree.
	n_actions: int, size of action space
	temperature: float, controls entropy of distribution
		> 1.0 is more random, < 1.0 is more deterministic.
	n_other_agents: int, number of other agents which interact
					with this agent

	Returns:
	agent: PDTMultiAgent using multi-tree with specified properties.
	n_other_agents: number of agents which provided action input to this multitree 
	"""


	# Create class object 
	agent = PDTMultiAgent(n_actions, n_other_agents, depth, [], window)
	
	# Assign pdt to root node
	n_branches = n_actions ** n_other_agents

	root = create_random_multitree(depth, n_children = n_branches,
								n_actions = n_actions, 
								temperature = temperature)
	agent._root = root
	agent.temperature = temperature

	return agent


def print_pdt(node):
	"""
	Print tree of up to depth 2.
	
	Args:
	node: the root PDTnode of a PDTAgent
	"""

	if node.has_children():
		print("\na root: ", node._distribution, " visits:%d uses:%d" % (node._visits,node._uses))

		for child in node._children:
			print_pdt(child)

	else:
		print("\tleaf: ", node._distribution, " visits:%d uses:%d" % (node._visits,node._uses))

def tensorToPylist(tensor_list):
	"""Convert tensor list to python list, rounded to 0.2f"""

	np_list = np.around(np.array(tensor_list.tolist()),2)
	py_list = list(np_list)

	return py_list

def test_PDTMultiAgent():
	"""Test PDTMultiAgent methods."""

	# Define variables
	n_actions = 3
	n_other_agents = 2
	depth = 1
	window = 1
	
	# Create object
	agent = create_random_pdt_multiagent(depth, n_actions, n_other_agents,
								window, temperature=1.0)
	print("\npdt multi-agent: ", agent)
	print("\n ordered_branch_codes: ", agent._ordered_branch_codes)

	# Traverse
	a0 = agent.act(t=0, traversal_actions=[])
	print("a0: ", a0)

	a1 = agent.act(t=1, traversal_actions=[3,2]) # branch 8
	print("a1: ", a1)

	a2 = agent.act(t=2, traversal_actions=[1,3]) # branch 3
	print("a2: ", a2)
	
#test_PDTMultiAgent()


'''# Uncomment to test create_random_pdt_agent()

"""Seed all RNG."""
seed = 1
random.seed(seed)
np.random.seed(random.randint(0, 1e6))
torch.manual_seed(random.randint(0, 1e6))

depth = 2
n_actions = 3
window = 5
temp = 0.5

serverPDT = create_random_pdt_agent(depth, n_actions, window, temp)
userPDT = create_random_pdt_agent(depth, n_actions, window, temp)


# Generate incremental history
totalTime = 3
from generate_histories_vGAN_and_recGAN import generateIncrementalHistory_randomPDT_randomPDT
(modelA_actionsList, modelB_actionsList), _, finalIncremHistory =  generateIncrementalHistory_randomPDT_randomPDT(totalTime, serverPDT, userPDT, separate_history=False)

print("\nPrint serverPDT after traversal")
print_pdt(serverPDT._root)
print("_max_node_visits: ",serverPDT._max_node_visits, " _max_node_uses: ", serverPDT._max_node_uses)

print("\nPrint userPDT after traversal")
print_pdt(userPDT._root)
print("_max_node_visits: ",userPDT._max_node_visits, " _max_node_uses: ", userPDT._max_node_uses)'''
'''
# Test build_pdt_graph()
#labels_d3 = ['a','b','c','d',  'e','f','g','h','i','j','k','l','m',  'n','o','p','q','r','s','t','u','v','w','x','y','z',   'a2','b2','c2','d2',  'e2','f2','g2','h2','i2','j2','k2','l2','m2',  'n2']
labels_d2 = ['a','b','c','d',  'e','f','g','h','i','j','k','l','m']

server_label = "\nServer PDT Agent\n Yellow-Orange-Red Spectrum (Red is Maximum)"
serverdot = Digraph(graph_attr= {'label': server_label}, node_attr={'fontsize': '7'})
serverdot = serverPDT.build_pdt_graph(serverPDT._root, serverdot, labels_d2[:], by_visits=False)
serverdot.render('./server_digraph.gv', view=True)

user_label = "\nUser PDT Agent\n Yellow-Orange-Red Spectrum (Red is Maximum)"
userdot = Digraph(graph_attr= {'label': user_label}, node_attr={'fontsize': '7'})
userdot = userPDT.build_pdt_graph(userPDT._root, userdot, labels_d2[:], by_visits=False)
userdot.render('./user_digraph.gv', view=True)'''


