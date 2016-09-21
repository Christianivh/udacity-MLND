import random
import numpy as np
from math import log
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha=0.1, gamma=0.1, epsilon=1.0):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # nine possible states of smartcab based on decision tree
        #
        self.possible_states = [
            'Red Light - Next waypoint NOT right',
            'Red Light - Next waypoint right - Clear on the left',
            'Red Light - Next waypoint right - NOT clear on the left - left car NOT going forward',
            'Red Light - Next waypoint right - NOT clear on the left - left car is going forward',
            'Green Light - Next waypoint right',
            'Green Light - Next waypoint forward',
            'Green Light - Next waypoint left - Clear oncoming',
            'Green Light - Next waypoint left - NOT clear oncoming - oncoming car turning left',
            'Green Light - Next waypoint left - NOT clear oncoming - oncoming car NOT turning left'
        ]

        # four possible actions
        self.possible_actions = [None, 'left', 'right', 'forward']

        # initialize Q-Table
        self.q_table = {state:{action:0 for action in self.possible_actions} for state in self.possible_states}

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # list of whether car reaches destination, 1 for success, 0 for failure
        self.success = []
        self.total_reward = 0
        self.sim_time = 0
        # the deadline at start time
        self.deadline_start = 0
        # used time in percentage of deadline, 1.0 on failure
        self.percentile_time = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

        # reset
        self.previous_state = self.possible_states[0]
        self.previous_action = None
        self.previous_reward = 0
        # reset deadline and success state for new trail
        self.deadline_start = 0
        self.success.append(0)
        self.percentile_time.append(1.0)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        if deadline > self.deadline_start:
            self.deadline_start = deadline

        # TODO: Update state
        if inputs['light'] == 'red':
            if self.next_waypoint != 'right':
                self.state = self.possible_states[0]
            else:
                if inputs['left'] == None:
                    self.state = self.possible_states[1]
                else:
                    if inputs['left'] != 'forward':
                        self.state = self.possible_states[2]
                    else:
                        self.state = self.possible_states[3]
        else:
            if self.next_waypoint == 'right':
                self.state = self.possible_states[4]
            elif self.next_waypoint == 'forward':
                self.state = self.possible_states[5]
            else:
                if inputs['oncoming'] == None:
                    self.state = self.possible_states[6]
                else:
                    if inputs['oncoming'] == 'left':
                        self.state = self.possible_states[7]
                    else:
                        self.state = self.possible_states[8]

        # TODO: Select action according to your policy
        # Epsilon greedy learning
        if self.sim_time < 1500:
            self.epsilon = 0.02
        else:
            self.epsilon = 0.0
#        self.epsilon = 0.02 / log(self.sim_time + 2)
#        self.epsilon = 0.9 / (1 + self.sim_time / 10)
#        self.epsilon = 0.5

        # choose random move is occasionally
        random_move = np.random.choice([1,0],p=[self.epsilon,1-self.epsilon])
        if random_move == 1:
            action_choose = random.choice(self.q_table[self.state].keys())
            action = action_choose
        # if an epsilon random move is not chosen
        else:
            # choose random move if all Q values are identical for each action
            if self.q_table[self.state][self.possible_actions[0]] \
                == self.q_table[self.state][self.possible_actions[1]] \
                == self.q_table[self.state][self.possible_actions[2]] \
                == self.q_table[self.state][self.possible_actions[3]]:
                action_choose = random.choice(self.q_table[self.state].keys())
                action = action_choose
            # else choose the action with highest Q value
            else:
                action_choose =  max(self.q_table[self.state].iterkeys(), key=(lambda key: self.q_table[self.state][key]))
                action = action_choose

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward

        # Reach destination, get success counter and used time in percentage
        if self.env.agent_states[self]['location'] == self.env.agent_states[self]['destination']:
            self.success[-1] = 1
            used_time = 1.0 * (self.deadline_start-deadline) / self.deadline_start
            self.percentile_time[-1] = used_time

        # TODO: Learn policy based on state, action, reward
        # Update the Q-table values, using the formula provided in the Reinforcement Learning lecture series
        self.q_table[self.previous_state][self.previous_action] = (1-self.alpha) * self.q_table[self.previous_state][self.previous_action] + self.alpha * (self.previous_reward + self.gamma * self.q_table[self.state][action_choose])

        # Update the 'old' state, action and reward before looping back to the next move
        self.previous_state = self.state
        self.previous_action = action_choose
        self.previous_reward = reward
        self.sim_time += 1

        # print the Q-table, for visualization and troubleshooting purposes
        #print "Q-TABLE:",self.q_table
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run(alpha=0.1, gamma=0.1, epsilon=1.0, n_trials=100):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, alpha, gamma, epsilon)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=n_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    ratio = np.mean(a.success)
    time = np.mean(a.percentile_time)
    #print ratio, time
    return (ratio, time, a.epsilon)

def search_parameters():
    '''
    Improve the Q-Learning Driving Agent
    search for optimal learning parameters
    '''
    alphas = np.arange(0, 1.1, 0.1)
    gammas = np.arange(0, 1.1, 0.1)
    #epsilons = np.arange(0, 1.1, 0.1)
    epsilons = [1.0]
    results = []
    best_result = (0, 0)
    best_parameters= (0, 0, 0)
    result = (0, 0)
    parameters = (0, 0, 0)
    learning_routes = []
    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                result = run(alpha=alpha, gamma=gamma, epsilon=epsilon, n_trials=100)
                parameters = (alpha, gamma, epsilon)
                results.append([result[0], result[1], parameters[0], parameters[1], parameters[2], result[2]])
                if (result[0] > best_result[0] or
                    (result[0] == best_result[0] and result[1] < best_result[1])):
                    best_result = result
                    best_parameters = parameters
                    learning_routes.append(results[-1])
    print "\n**Searching Path**"
    print "success ratio | percentile time | alpha | gamma | epsilon | final_epsilon"
    print "---|---|---|---|---"
    for r in learning_routes:
        print "{:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3}".format(*r)
    print "\n**Best Result**"
    print "success ratio | percentile time | alpha | gamma | epsilon | final_epsilon"
    print "---|---|---|---|---"
    print "{:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3}".format(best_result[0], best_result[1],
                                                                 best_parameters[0], best_parameters[1], best_parameters[2],
                                                                 best_result[2])
    return best_result, best_parameters

if __name__ == '__main__':
    # run()
    search_parameters()
