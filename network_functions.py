"""Networking functions."""
import random
from random import randint
import socket, threading, queue, select, time


def get_random_port_number(lowest_legal_port, highest_port_range):
	"""
	Generates a port number between lowest_legal_port (~1025) ...
	and lowest_legal_port + highest_port_range.
	"""
	return lowest_legal_port + randint(1,highest_port_range)


def setup_server_style_socket(ip_addr, port_number):
	"""Set up a server-style socket, and return when it has accepted a connecting client."""
	print("\nBeginning setup_server_style_socket()")

	f = open("Beginning_setup_server_style_socket()"+str((ip_addr, port_number))+str(threading.current_thread().ident), "w+")
	f.close()

	# Create server-style socket
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 

	# Bind socket to provided port
	try: 
		s.bind(('', port_number)) # bound to empty IP string so it can listen and accept
	except:
		print("Couldn't bind to port: ", port_number)
		f = open("COULDNT_BIND_TO_PORT_"+str(port_number), "w+")
		f.close()

	# Listening mode - upto 5 connections can wait if server is busy
	try:
		s.listen(5) 
	except:
		print("Couldn't listen")

	# Blocking call waits here until a client style socket connects
	server_sock_for_client, addr = s.accept()	
	
	# Record if success
	print_msg = "Accepted as server from addr: " + str(addr) + " at port: " + str(port_number) + " with server-style socket: " + str(server_sock_for_client)
	print(print_msg)

	f = open("Accepted_as_server_from_addr"+str(addr)+"_with_socket:" + str(server_sock_for_client), "w+")
	f.close()

	return server_sock_for_client

def setup_client_style_socket(ip_addr, port_number):
	"""Return a client-style socket once it connects to an already expecting server-style socket."""
	print("\nBeginning setup_client_style_socket()")

	f = open("Beginning_setup_client_style_socket()"+str((ip_addr, port_number))+str(threading.current_thread().ident), "w+")
	f.close()

	# Create client-style socket
	client_sock_for_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)	
	client_sock_for_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


	# Connect to already listening server-style socket of other agent on another machine

	conn_result = client_sock_for_server.connect_ex((str(ip_addr), port_number))

	if conn_result == 0:		# this call requires server socket to be already listening to accept
		print_msg = "Connected successfully as client to: " + str((ip_addr, port_number)) +" with socket: " + str(client_sock_for_server)
		print(print_msg)

		f = open("Connected_successfully_as_client_to_"+str(ip_addr)+':'+str(port_number), "w+")
		f.close()
	else:
		print("Failed to connect as the client to: ", (ip_addr, port_number))
		f = open("Failed_to_connect_as_client_to_"+str(ip_addr)+':'+str(port_number)+"_conn_result_"+str(conn_result), "w+")
		f.close()

	return client_sock_for_server

def connect_socket_pairs(setup_function, socket1_info_tuple, socket2_info_tuple):
    """
    (Multi-threaded) Set up and connect a pair of sockets to remote socket pair.

    Args:
    setup_function: either setup_server_style_socket() or 
                           setup_client_style_socket()
    socket1_info_tuple: (ipaddr1, port1, thread_type) 
                        thread_type = 'send' or 'receive'                       
    """

    # Create queue to hold returned threads from threaded functions
    threads_queue = queue.Queue()
    threads_list = []
    num_threads = 2

    # Create lambda function for threads to use with provided params
    lambda_func = lambda q, ipaddr, port, thread_type: q.put(
                        (setup_function(ipaddr, port), str(thread_type)) )

    # Get connection info for each socket
    (ipaddr1, port1, thread1_type) = socket1_info_tuple
    (ipaddr2, port2, thread2_type) = socket2_info_tuple

    # Create threads with agent's connection info
    thread1 = threading.Thread(target = lambda_func, args = (
                                    threads_queue, 
                                    ipaddr1, port1, 
                                    thread1_type))

    thread2 = threading.Thread(target = lambda_func, args = (
                                    threads_queue,
                                    ipaddr2, port2,
                                    thread2_type))
    # Add threads to threads list
    threads_list.append(thread1)
    threads_list.append(thread2)

    # Now release all threads so they run simultaneously
    for t in threads_list:
        t.start()

    # Wait for all threads to finish
    while(1):
        all_threads_finished = True 

        for t in threads_list:
            if t.isAlive():
                all_threads_finished = False

        if all_threads_finished == True:
            break

    # Now ensure threads queue has received return values from all threads
    assert threads_queue.qsize() == num_threads, "Thread queue not full!"

    # Collect return values from threads queue
    threads_queue_items = []
    for item in range(num_threads):
        threads_queue_items.append(threads_queue.get())

    # Print returned values
    print("\nthreads_queue_items: ", threads_queue_items)

    # Order sockets by socket function (by thread label)
    sockets_for_agent = [-1,-1]
    for item in threads_queue_items:
        (socket, thread_label) = (item[0], item[1]) # i.e. (socket, 'receive')

        # By convention, socket info format is: [send_sock, receive_sock]
        if 'send' in thread_label:
            sockets_for_agent[0] = socket
        else:
            sockets_for_agent[1] = socket

    # Return prepared socket pair
    return sockets_for_agent

def receive_from_socket(recv_sock):
	"""
	Perform blocking call in script until msg arrives at receive socket.
	Message convention format (with delimiter): 'msg_COMPLT'.

	Return: message of type string (delimiter removed).
	"""

	# Define variables
	end_delim = '_COMPLT'
	end_delim_size = len(end_delim)

	# Wait until receive socket contains at least one message
	while(1):

		# Blocking call until socket is read-ready
		socket_ready = select.select([recv_sock], [], [])

		if socket_ready[0]:
			peek_data = recv_sock.recv(1024,socket.MSG_PEEK).decode()

			# i.e. if '_COMPLT' found in peek data which is string
			if end_delim in peek_data: 
				#print("\nend delimiter: ", end_delim, " found in peek_data: ", peek_data)
				break

	# Compute size of first message
	first_msg_end_index = peek_data.find(end_delim) # i.e. find '_COMPLT'
	first_msg = peek_data[:first_msg_end_index+end_delim_size] # i.e. 'msg_COMPLT'
	first_msg_size = len(first_msg.encode('utf-8'))
	#print("\nfirst_msg: ", first_msg)

	# Remove only first available msg from socket buffer
	received_msg = recv_sock.recv(first_msg_size).decode()
	#print("\nreceived_msg: ", received_msg, " type: ", type(received_msg))

	# Remove delimiter from message
	msg = received_msg.replace(end_delim, '') # i.e. remove '_COMPLT'
	#print("\nmsg without delimiter: ", msg)
	
	# Return message
	return msg

def receive_action_from_socket(recv_sock):
	"""
	Perform blocking call in script until msg arrives at receive socket.
	Return all received messages in socket buffer, ordered by msg timestep.
	Message convention example: '3F_t0_C'

	Return: list of time-ordered tuples: [(3,0), (2,1)]
			list of unordered time-formatted msgs ['3F_t0_C', '2F_t1_C']
	"""
	#print("\nInside receive_from_socket()!")

	# Wait until receive socket contains at least one message
	while(1):

		# Blocking call until socket is read-ready
		socket_ready = select.select([recv_sock], [], [])

		if socket_ready[0]:
			peek_data = recv_sock.recv(1024,socket.MSG_PEEK).decode()

			if 'C' in peek_data:
				break

	msg_list = -1

	# Compute size of first message
	first_msg_end_index = peek_data.find('C')
	first_msg = peek_data[:first_msg_end_index+1] # i.e. '3F_t0_C'
	first_msg_size = len(first_msg.encode('utf-8'))
	#print("first_msg: ", first_msg)

	# Remove only first available msg from socket buffer
	received_msg = recv_sock.recv(first_msg_size).decode()
	#print("received_msg: ", received_msg)
	
	# Wrap single msg in list
	msg_list = [received_msg]

	# Convert each string msg to tuple: (msg_action, msg_time)
	msg_list_tuples = [strip_msg(x) for x in msg_list]

	# Example: [(3,0), (2,1)] for 3F,t0 then 2F,t1

	# Order received messages by timestep, the second value
	msg_list_tuples.sort(key=lambda x:x[1])

	#print("\nmsg_list: ", msg_list)
	#print("time-ordered msg_list_tuples: ", msg_list_tuples)

	return (msg_list, msg_list_tuples)

def strip_msg(msg):
	"""
	Strip integer action and timestep from received message.

	Args:
	msg: convention is i.e. '3F_t7_C' (action=3, timestep=7)
	"""
	#print("\nInside strip_msg with: ", msg)

	# Collect timestep
	time_begin_index = msg.index('t') + 1
	time_end_index = msg.index('_C')

	msg_timestep = int(msg[time_begin_index:time_end_index])

	# Collect integer action
	action_end_index = msg.index('F')
	msg_action = int(msg[:action_end_index])

	#print("\nmsg_action: %d, msg_timestep: %d" % (msg_action,msg_timestep))
	return (msg_action, msg_timestep)

def send_over_sockets(send_sock_list, msg_list, network_protocol):
	"""
	(Multi-threaded) Simultaneously send multiple messages by given sockets.
	Messages are formatted simply with '_COMPLT' delimiter: 'msg_COMPLT'.

	Args:
	send_sock_list: list of sockets by which msgs should be sent.
	msg_list: list of strings, i.e. ['msg1', 'msg2', 'msg3']
	"""

	# Define variables
	threads_list = []
	end_delim = '_COMPLT'

	# Build list of threads for all msgs
	for (send_sock, msg) in zip(send_sock_list, msg_list):

		# Format message with action and time, i.e. 'msg_COMPLT'
		msg_formatted = str(msg) + end_delim
		
		# Build thread
		send_thread = threading.Thread(target = send_over_sock, 
						args = (send_sock, msg_formatted, network_protocol))

		threads_list.append(send_thread)

	# Start all threads so all sockets send simultaneously
	for t in threads_list:
		t.start()

	# Wait for the threads to finish
	for t in threads_list:
		t.join()


def send_actions_over_sockets(send_sock_list, msg_action_list, t, network_protocol):
	"""
	(Multi-threaded) Simultaneously send multiple messages by given sockets.
	Messages are formatted as string with int action and time marker.
	Formatted message example: '3F_t0_C'

	Args:
	send_sock_list: list of sockets by which msgs should be sent.
	msg_action_list: list of ints, i.e. [1,3,2,1]
	t: int, current timestep or time marker
	network_protocol: 'tcp' or 'udp'
	"""

	threads_list = []

	# Build list of threads for all msgs
	for (send_sock, msg_action) in zip(send_sock_list, msg_action_list):

		# Format message with action and time, i.e. '3F_t0_C'
		msg_formatted = str(msg_action) + 'F' + '_t' + str(t) + '_C'
		
		# Build thread
		send_thread = threading.Thread(target = send_over_sock, 
						args = (send_sock, msg_formatted, network_protocol))

		threads_list.append(send_thread)

	# Start all threads so all sockets send simultaneously
	for t in threads_list:
		t.start()

	# Wait for the threads to finish
	for t in threads_list:
		t.join()

def send_over_sock(send_sock, msg, network_protocol):
	"""
	Transmit string-formatted msg over send socket.

	Args:
	msg: formatted msg string, i.e. '3F_t0_C'
	"""
	
	#print("\nInside send_over_sock")
	#print("with msg.encode():", msg.encode(), "of type: ", type(msg.encode()))
	
	# Send result should be None if successful
	send_result = send_sock.sendall(msg.encode()) # encode as bytes
	#print("\nSend result is None if successful: ", send_result)

	# Protect receive socket from being overloaded in case we send repeatedly
	#time.sleep(0.25)
