# Functions for computing session keys.
import torch
from torch.distributions.categorical import Categorical

import base64
import cryptography
from cryptography.fernet import Fernet # AES 128-based cipher suite 
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def pbkd(password, salt):
	"""
	Password-based key derivation function to produce key.
	This password-based key used to build ciphersuite for encryption/decryption.

	Args:
	password: of type string
	salt: of type int or float
	"""

	# Convert password/salt to type bytes if required
	if type(password) != bytes:
		password = password.encode()
	if type(salt) != bytes:
		salt = str(salt).encode()

	# Build one-time key derivation function
	kdf = PBKDF2HMAC(
	    algorithm=hashes.SHA256(),
	    length=32,
	    salt=salt,
	    iterations=100000,
	    backend=default_backend()
	)

	# Produce key from one-time function
	cipher_key = base64.urlsafe_b64encode(kdf.derive(password)) 
	print("\ncipher_key: ", cipher_key) 

	# Return key for building ciphersuite
	return cipher_key

def encrypt(cipher_key, plaintext):
	"""
	Use ciphersuite from provided cipher key to encrypt plaintext.

	Args:
	cipher_key: URL-safe base-64 encoded key of length 32 bytes.
				Provided from key derivation function.
	plaintext: of type string.
	"""

	# Build ciphersuite from key
	ciphersuite = Fernet(cipher_key)

	# Convert plaintext to type bytes
	if type(plaintext) != bytes:
		plaintext_encoded = plaintext.encode()

	# Encrypt plaintext with ciphersuite
	ciphertext = ciphersuite.encrypt(plaintext_encoded)
	print("\nciphertext: ", ciphertext)

	# Return cipher text
	return ciphertext

def decrypt(cipher_key, ciphertext):
	"""
	Use ciphersuite from provided cipher key to decrypt ciphertext.

	Args:
	cipher_key: URL-safe base-64 encoded key of length 32 bytes.
				Provided from key derivation function.
	ciphertext: of type bytes.
	"""

	# Build ciphersuite from key
	ciphersuite = Fernet(cipher_key)

	# Decrypt ciphertext with ciphersuite
	plaintext = ciphersuite.decrypt(ciphertext)

	# Convert plaintext from type bytes to type string
	plaintext_decoded = plaintext.decode()
	print("\nplaintext_decoded: ", plaintext_decoded)

	# Return string plaintext
	return plaintext_decoded


def session_key_v1(histories_list):
	"""
	Compute session key from agent interaction histories.
	
	Note: history dists are rounded to 3 decimal places.

	Args:
	histories_list: [agent0_history, agent1_history,...]
		with agent0_history = [(a0,dist0),(a1,dist1),(a2,dist2)...]
	"""

	# Define variables
	session_key = ''

	# Build session key as string of concatenated histories
	for (history,l) in zip(histories_list, range(len(histories_list))):

		history_str = ''

		# Build history string as: 'dist0[a0-1],dist1[a1-1],...'
		for (tup,t) in zip(history, range(len(history))):
			(action, dist) = tup

			# Adjust for 1-indexed action space
			prob = dist[action-1] 

			# Convert if tensor
			if type(prob) == torch.Tensor:
				prob = prob.data.item()

			# Round to 3 decimal places (as string)
			prob_str = '%.3f' % (prob) # i.e. '0.935'

			# Format i.e. '0.935,'
			if t != len(history)-1:
				prob_str += ','

			# Append to history string
			history_str += prob_str

		#print("history_str: ", history_str)

		# Format i.e. '0.935,0.675,0.872*'
		if l != len(histories_list)-1:
			history_str += '*'

		# Build session key as: history0 * history1 * ...
		session_key += history_str

	# Example session key: '0.2,0.4,0.1*0.3,0.9,0.7'
	return session_key

def test_session_key_v1():
	"""Test the compute_key_v1() function."""

	# Define variables
	num_agents = 3
	num_timesteps = 5
	action_space_size = 3

	# Build agent histories
	histories_list = []
	for l in range(num_agents):

		# Build history for number of timestepss
		history = []
		for t in range(num_timesteps):

			logits = torch.rand(action_space_size)
			Cat = Categorical(logits=logits)
			
			dist = Cat.probs
			action = Cat.sample().data.item() + 1 # adjust as 1-indexed

			history.append((action,dist))

		# Append to histories list
		histories_list.append(history)

	print("histories_list: ", histories_list)

	# Compute session key v1
	key = session_key_v1(histories_list)

	print("key: ", key)