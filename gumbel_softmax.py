"""Functions for gumbel-softmax transform."""

import torch
import torch.nn.functional as F
from device_settings import device 

def gumbel_softmax_transform(raw_output_batch, numActions, gumbelTemp, gumbelReturnHard):
	"""
	Return gumbel samples based on raw output of network.
	Return format concatenates each row of samples as joined vector,
	 ...ready to be input into neural network.
	
	Args: 'raw_output_batch' shape: (batch_size, network output dim).
		  'gumbelReturnHard': return a hard one-hot-vector if True.
	"""

	batch_size = raw_output_batch.shape[0]

	# Place batch outputs into list of outputs
	all_samples_outputs = []
	# print("printing samples in output batch")

	for i in range(batch_size):
		# print("raw_output_batch[i]: ", raw_output_batch[i]")
		all_samples_outputs.append(raw_output_batch[i])

	# print("all_samples_outputs: ", all_samples_outputs)
	
	# Split raw output values from output layer into sets for each action vector
	all_samples_splits = []
	for sample_output in all_samples_outputs:
		raw_output_split = torch.split(sample_output, numActions, dim=0)  # returns a tuple
		all_samples_splits.append(raw_output_split)


	# print("all_samples_splits: ", all_samples_splits) # length is batch_size, as a list of tuples
	# print("len(all_samples_splits): ", len(all_samples_splits))

	# Perform gumbel softmax transform
	all_samples_gumbel_softmax_vectors = []
	for this_sample_splits in all_samples_splits:
		# print "\nthis_sample_splits: ", this_sample_splits

		gumbel_softmax_vectors = []
		for item in this_sample_splits:
			item_logsoftmax = F.log_softmax(item,dim=0).to(device)
			#print "item_logsoftmax: ", item_logsoftmax

			item_gumbel_softmax = F.gumbel_softmax(logits=item_logsoftmax.view(1,-1).contiguous(), tau=gumbelTemp, hard=gumbelReturnHard).to(device)
			#print "item_gumbel_softmax: ", item_gumbel_softmax

			gumbel_softmax_vectors.append(item_gumbel_softmax)
		
		#print "gumbel_softmax_vectors: ", gumbel_softmax_vectors

		all_samples_gumbel_softmax_vectors.append(gumbel_softmax_vectors)


	# all_samples_gumbel_softmax_vectors is tuple of length batch_size
	#print("\nall_samples_gumbel_softmax_vectors: ", all_samples_gumbel_softmax_vectors)
	#print("len(all_samples_gumbel_softmax_vectors): ", len(all_samples_gumbel_softmax_vectors))

	# Example of all_samples_gumbel_softmax_vectors with batch size of 2:
	# 
	# [[tensor([[0., 1., 0.]]), tensor([[0., 1., 0.]]), tensor([[0., 1., 0.]])], 
	#  [tensor([[0., 1., 0.]]), tensor([[0., 0., 1.]]), tensor([[0., 1., 0.]])]
	# ]


	# For each sample, combine it into a single concatenated array of softmax vectors
	#print("\nConcatenating each sample tensors into a single array")
	
	all_samples_concatenated = []
	for sample in all_samples_gumbel_softmax_vectors:
		#print "sample: ", sample
		sample_unwrapped = tuple(x[0] for x in sample)
		#print "sample_unwrapped: ", sample_unwrapped
		
		this_sample_concat = torch.cat(tuple(sample_unwrapped), dim=0).to(device)
		#print "this_sample_concat: ", this_sample_concat
		all_samples_concatenated.append(this_sample_concat)

	#print "all_samples_concatenated: ", all_samples_concatenated
	#print "len(all_samples_concatenated): ", len(all_samples_concatenated)

	# Wrap batch into outer tensor containing tensor for each sample
	batch_samples_stacked = torch.stack(all_samples_concatenated,dim=0).to(device)
	#print("batch_samples_stacked: ", batch_samples_stacked)
	#print("batch_samples_stacked[0]: ", batch_samples_stacked[0])

	# Example of batch_samples_stacked with batch size of 2:
	#
	# tensor([[0., 1., 0., 0., 1., 0., 0., 1., 0.], [0., 1., 0., 0., 0., 1., 0., 1., 0.]])

	'''# Uncomment to view gradients flowing back into Generator 
	gen_batch_output_flowchart = make_dot(batch_samples_stacked, params=None)	 
	gen_batch_output_flowchart.render('gen_batch_output_flowchart.gv', view=True)'''

	return batch_samples_stacked, all_samples_gumbel_softmax_vectors
	
def test_gumbel_softmax_transform():
	"""Test the gumbel-softmax transform function."""

	# Define variables
	numActions = 3
	batch_size = 3
	gumbelTemp = -1
	gumbelReturnHard = True

	x = torch.randn(batch_size,9)

	# Perform transform
	(batch_samples_stacked, all_samples_gumbel_softmax_vectors) = gumbel_softmax_transform(
																	x, 
																	numActions, 
																	gumbelTemp, 
																	gumbelReturnHard)
	print("all_samples_gumbel_softmax_vectors: ")
	for h in all_samples_gumbel_softmax_vectors:
		print("h: ", h)
