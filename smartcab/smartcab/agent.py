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
        '''
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
        '''

        # four possible actions
        self.possible_actions = [None, 'left', 'right', 'forward']

        # initialize Q-Table
        self.q_table = {}
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

    def get_state(self):
        '''
        get current states
        '''
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        return (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])

    def get_max_utility_action(self, qtable, s):
        '''
        choose the action with highest score in q table
        '''
        tmp = [qtable[(s, x)] for x in self.possible_actions]
        maxQ = max(tmp)
        count = tmp.count(maxQ)
        if count > 1:
            best = [i for i in range(len(self.possible_actions)) if tmp[i] == maxQ]
            i = random.choice(best)
        else:
            i = tmp.index(maxQ)
        return self.possible_actions[i]

    def get_decay_rate(self, t):
        '''
        Decay rate for alpha and epsilon
        '''
        if t == 0:
            return 1
        return 1 / float(t)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # reset deadline and success state for new trail
        self.deadline_start = 0
        self.success.append(0)
        self.percentile_time.append(1.0)

    def update(self, t):
        #
        self.alpha *= self.get_decay_rate(t)
        self.epsilon *= self.get_decay_rate(t)

        # TODO: Update state
        self.current_state = self.get_state()

        # initialize deadline_start for new trial
        deadline = self.env.get_deadline(self)
        if deadline > self.deadline_start:
            self.deadline_start = deadline
        # initialize Q Table
        for action in self.possible_actions:
            if (self.current_state, action) not in self.q_table:
                self.q_table[(self.current_state, action)] = 0.0

        # TODO: Select action according to your policy
        if random.random() < self.epsilon:    # explore
            action = random.choice(self.possible_actions)
        else: # exploitation
            action = self.get_max_utility_action(self.q_table, self.current_state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        # new state after taking action
        self.next_state = self.get_state()

        # Reach destination, get success counter and used time in percentage
        if self.env.agent_states[self]['location'] == self.env.agent_states[self]['destination']:
            self.success[-1] = 1
            used_time = 1.0 * (self.deadline_start-deadline) / self.deadline_start
            self.percentile_time[-1] = used_time
        # otherwise learn and update q table
        else:
            # update q table entry
            for a in self.possible_actions:
                if (self.next_state, a) not in self.q_table:
                    self.q_table[(self.next_state,a)] = 0.0
            # TODO: Learn policy based on state, action, reward
            next_action = self.get_max_utility_action(self.q_table, self.next_state)
            change = self.alpha * (reward + self.gamma * self.q_table[(self.next_state, next_action)])
            self.q_table[(self.current_state, action)] = self.q_table[(self.current_state, action)] * (1 - self.alpha) + change

        self.total_reward += reward
        self.sim_time += 1

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
    epsilons = np.arange(0, 1.1, 0.1)
    #epsilons = [1.0]
    results = []
    best_result = (0, 0)
    best_parameters= (0, 0, 0)
    result = (0, 0)
    parameters = (0, 0, 0)
    learning_routes = []
    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                result = run(alpha=alpha, gamma=gamma, epsilon=epsilon, n_trials=1)
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
