# Functions for performing hypothesis test by Albrecht & Ramamoorthy.
from scipy import optimize
from scipy.stats import skewnorm, norm
from torch.distributions.categorical import Categorical
import numpy as np

def snorm_mean(p_abc):  # input is vector [location, scale, shape]
    """Compute skew-norm mean."""
    # [e, w, B] are [location, scale, shape] parameters, so below is...
    # ... mean = e + w * (B / sqrt(1+B)) * 2/sqrt(pi)

    # Use floats as divisors for float division
    mean = p_abc[0] + (p_abc[1] * (p_abc[2] / np.sqrt(1.0 + np.power(p_abc[2], 2))) * np.sqrt(2.0 / np.pi))

    return mean

def snorm_pdf(X, p_abc):    
    """
    Compute densities/probabilities for each element in X.
    Find the y-value in the skew-normal pdf graph for each x-value.

    Verified against Matlab HBA, provided parameters are correct

    Args:
    X: vector of data points
    p_abc: [location, scale, shape]
    """
    X = np.array(X)

    # Numpy's true float division using a float divisor
    X_ = np.divide((X - p_abc[0]) * 1.0, p_abc[1])  # Transformation: (X - e) / w

    # Eq 10 from HBA paper: skew-normal pdf = 2/w * pdf(X_) * cdf(B * X_)
    Y = 2.00/p_abc[1] * norm.pdf(X_) * norm.cdf(p_abc[2] * X_)

    return Y


def snorm_mode(x, p_abc):
    """
    Compute mode of skew normal.
    Verified against Matlab with both scalar and array inputs for X.
    The mode of a pdf is at the local maxima, that's why we use optimize().

    Returns: tuple (xopt, fopt, iter, funcalls, warnflags, allvecs)
                fopt is the value of function at minimum (the mode)
    """

    def negative_snorm_pdf(x):

        x = np.array(x)   

        # Numpy's true float division using a float divisor
        X_ = np.divide((x - p_abc[0]) * 1.0, p_abc[1])   # Transformation: (X - e) / w

        # Eq 10 from HBA paper: skew-normal pdf = 2/w * pdf(X_) * cdf(B * X_)
        Y = 2.00 / p_abc[1] * norm.pdf(X_) * norm.cdf(p_abc[2] * X_)

        return -Y

    # output: tuple sized 6, (xopt, fopt, iter, funcalls, warnflags, allvecs)
    output = optimize.fmin(negative_snorm_pdf, snorm_mean(p_abc), full_output=True, disp=False)  

    # Take 'fopt', the value of function at minimum
    # Dmode is value of graph at local minima, so we need the actual y-value
    mode = -output[1]  

    xval_mode = output[0]
    # print("non-returned x-value at mode: ", output[0])  # check the x-value

    return [mode, xval_mode]

def z1_score_function(list_a_t, list_x_t, num_time_steps):
    """z1 performs the E_time (prob of action chosen / max prob of all available actions)

    Args:
    list_a_t: the observed actions by client, for example [1,3,2,3,1,2,3,1,1]
    list_x_t: the list of distributions at each node that client went to
            i.e. [ [.2,.5,.3], [.5,.2.,.3], ...]
    """

    list_of_items = []  # holds each 0.5 / max([0.5,0.2,0.3]) for the two lists

    for i in range(num_time_steps):

        try:
            # action chosen by client from {1,2,3} for example
            action_chosen = list_a_t[i]   
        except:
            exit("Exiting: error in z1_score_function!")

        if type(action_chosen) == list:
            # if they came in form [2], or [1], for example
            action_chosen = action_chosen[0]    

        # Probability distribution at this node, like [0.5,0.2,0.3]
        probs_available_actions = list_x_t[i]   

        # Change from {1,2,3} to 0-indexed index
        prob_action_chosen = probs_available_actions[action_chosen - 1]   

        max_prob = max(probs_available_actions)  

        # Example of max_prob: 0.5 from [0.5,0.2,0.3]

        list_of_items.append(prob_action_chosen / (max_prob * 1.0))

    z1_score = sum(list_of_items) * 1.0 / num_time_steps
    return z1_score

def z2_score_function(list_a_t, list_x_t, num_time_steps):

    # format of CDT gives each node as a single-value list, so convert them to ints
    # i.e. convert from [[2],[3],[1]] to [2,3,1] 
    if all([type(x) == list for x in list_a_t]) == True:    
        list_a_t = [y[0] for y in list_a_t]

    list_of_items = []
    for i in range(num_time_steps):

        try:
            # action chosen by client from {1,2,3} for example
            action_chosen = list_a_t[i]            
        except:
            exit("Exiting: error in z2_score_function()!")

        # probability distribution at this node, like [0.5,0.2,0.3]
        probs_available_actions = list_x_t[i]   

        # 'action_chosen-1' is to change from {1,2,3} to 0-indexed index
        prob_action_chosen = probs_available_actions[action_chosen-1]   

        # the expectation E (the E is SUM: |selected prob-prob of a_j| * prob of a_j)
        this_expectation = 0 

        for j in range(len(probs_available_actions)):
            this_expectation += (probs_available_actions[j] * (abs(prob_action_chosen - probs_available_actions[j])) )
        #print "this_expectation: ", this_expectation

        list_of_items.append(1 - this_expectation)

    z2_score = sum(list_of_items)*1.0 / num_time_steps

    return z2_score

def z3_score_function(list_a_t, list_x_t, num_time_steps):

    # holds the min between two lists, for all actions a_i from A = {}
    bigList = [] 

    numActions = len(list_x_t[0])   # the size of A = {}
    for i in range(numActions):

        currentAction = i + 1   # this variable will be 1-3, for example

        firstList = []
        secondList = []
        for j in range(num_time_steps):

            # Calculate item for firstList
            try:
                # action chosen by client from {1,2,3} for example
                action_chosen = list_a_t[j]             
            except:
                exit("Exiting: error in z3_score_function!")

            if action_chosen == currentAction:   # if a_j^t == a_j
                firstList.append(1)
            else:
                firstList.append(0)

            # Calculate item for secondList

            # probability distribution at this node, like [0.5,0.2,0.3]
            probs_available_actions = list_x_t[j]   

            # probability of getting a1, for example
            prob_currentAction = probs_available_actions[currentAction-1]   

            secondList.append(prob_currentAction)

        # Find min between averages of two lists, then append to outer list
        firstList_avg = sum(firstList)*1.0/num_time_steps
        secondList_avg = sum(secondList)*1.0/num_time_steps

        # for this a_i from A={}, append the minimum
        bigList.append(min(firstList_avg, secondList_avg))  

    # Now with minimum values for each a_i in A={}, sum up for the z3 score
    z3_score = sum(bigList)

    return z3_score

def calculate_test_statistic(score_funcs_list, weight_value, 
                            actions_a, dists_a, actions_b, dists_b, 
                            num_time_steps):
    """Compute test statistic using provided score functions."""

    outer_sum = 0   # to hold the sum over all time steps
    for i in range(num_time_steps):

        inner_sum = 0  # to hold the sum over all score functions
        for score_fn in score_funcs_list:

            z_i_a = score_fn(actions_a, dists_a, i + 1)  # num_time_steps is 1 at t=0
            z_i_b = score_fn(actions_b, dists_b, i + 1)

            #print("z_i_a: ", z_i_a)
            #print("z_i_b: ", z_i_b)
            #print("z_i_a - z_i_b: ", z_i_a - z_i_b)

            inner_sum += (z_i_a - z_i_b)

        outer_sum += inner_sum * weight_value

    test_statistic = outer_sum * 1.0 / num_time_steps

    return test_statistic


def hypothesis_test_v1(unknown_agent_actions, known_agent_dists, 
                update_times, num_actions, num_samples=50, 
                score_funcs_list = [z1_score_function, z2_score_function, z3_score_function], 
                weight_value=(1.0/3)):
    """
    Calculate p-values for various timesteps of an interaction process.

    Args:
    unknown_agent_actions: python list of integers
    known_agent_dists: python list of sublists, each sublist a probability distribution

    Returns: tuple(pvalue_regular, pvalue_ratio) as the final p-value for interaction process
    """

    # Check for correct parameter size
    max_time = len(unknown_agent_actions)
    assert (
        len(unknown_agent_actions) == max_time
    ), "unknown_agent_actions is incorrect length."
    assert len(known_agent_dists) == max_time, "known_agent_dists is incorrect length."

    # Declare variables
    a = []  # observed actions from unknown agent
    a_hat = []  # sampled actions from dists of known agent
    a_tilda_list = [
        [] for _ in range(num_samples)
    ]  # sampled actions dists of known agent

    pvalue_ratio_list = []
    pvalue_regular_list = []

    # Run hypothesis test
    for t in range(max_time):

        # Part 1: Collect observed action from unknown agent
        a.append(unknown_agent_actions[t])

        # Part 2: Sample action from known agent dist, adjusted for action space
        a_hat.append(Categorical(probs=known_agent_dists[t]).sample().item() + 1)

        # Part 3: Sample set of actions from known agent dists
        for n in range(num_samples):
            a_tilda_list[n].append(
                Categorical(probs=known_agent_dists[t]).sample().item() + 1
            )

        # Part 4: If specified, compute p-value at current time
        if t in update_times:
            # Part 4A: Fill D array with test statistics calculated between known agent and known agent

            D = np.array([])  # holds test statistics for known agent with known agent

            # Convert known_agent_dists dists from tensors to python lists
            known_agent_dists_pylists = [
                x.tolist() for x in known_agent_dists[0 : t + 1]
            ]

            # Compute test statistic between known agent (a_hat) and known agent (a_tilda_list)
            for n in range(num_samples):

                test_statistic = calculate_test_statistic(
                    score_funcs_list,
                    weight_value,
                    a_tilda_list[n],
                    known_agent_dists_pylists,
                    a_hat,
                    known_agent_dists_pylists,
                    num_time_steps=t + 1
                )
                D = np.append(D, test_statistic)

            # Part 4B: Fit parameters to skew-normal distribution of D
            [shape, loc, scale] = skewnorm.fit(D)
            skew_params = [loc, scale, shape]

            # Part 4C: compute test statistic between unknown agent (a) and known agent (a_hat)
            q = calculate_test_statistic(
                score_funcs_list,
                weight_value,
                a,
                known_agent_dists_pylists,
                a_hat,
                known_agent_dists_pylists,
                num_time_steps=t + 1
            )

            # Part 4D: Calculate p-value from 'q' by two calculation methods

            # Calculate p-value by 'ratio' method
            pvalue_ratio = (
                snorm_pdf(q, skew_params) * 1.00 / snorm_mode(D, skew_params)[0]
            )  # does f(q| params) / f(mode| params)

            pvalue_ratio_list.append(pvalue_ratio)

            # Calculate p-value by 'regular' method (fitting the normal distribution to D)

            m, s = norm.fit(D)  # fit normal dist to D, and get its mean/std dev

            pvalue_regular = -1  # intialize variable

            # If tt is to the left of mean, then graphically, we take cdf(tt)
            if q <= m:
                pvalue_regular = norm.cdf(q, m, s) * 2.0  # double the tail area

            # Otherwise, if tt to the right of mean, take 1-cdf(tt)
            else:
                pvalue_regular = (1 - norm.cdf(q, m, s)) * 2.0

            pvalue_regular_list.append(pvalue_regular)

    # Return final pvalue_regular and pvalue_ratio as tuple
    return (pvalue_regular_list[-1], pvalue_ratio_list[-1])
