import gym
import numpy as np
from gym.envs.registration import register
import datetime
np.set_printoptions(suppress=True)
# calculating policies for each state (U,D,R,L)
def policy_table_comp (v,i,j):
    action=np.zeros (4)
    v[4][4]=1
    if (i-1)<0:
        action[0] = v[i][j]
    else:
        action[0] = v[i-1][j]
    if (i+1)>4:
        action[1] = v[i][j]
    else:
        action[1] = v[i+1][j]
    if (j+1)>4:
        action[2] = v[i][j]
    else:
        action[2] = v[i][j+1]
    if (j-1)<0:
        action[3] = v[i][j]
    else:
        action[3] = v[i][j-1]
    a=np.argmax(action)
    if a==0:
        return "U"
    elif a==1:
        return "D"
    elif a==2:
        return "R"
    elif a==3:
        return "L"
# frozen lake map initialization
my_desc = [
              "SFFFH",
             "FFHHF",
              "FFFFF",
              "HHFHF",
              "FFFFG"
            ]
register(
    id='Stochastic-5x5-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'desc': my_desc, 'is_slippery': True})
env = gym.make('Stochastic-5x5-FrozenLake-v0')
env.reset()
env.render()
V_fin = np.zeros ((5,5))
policy_table = np.chararray((5, 5),unicode=True)
def policy_evaluation(policy, environment, discount_factor=0.85, theta=1e-9, max_iterations=1e9):
    # Number of evaluation iterations
    evaluation_iterations = 1
    V = np.zeros(environment.nS)
    for i in range(int(max_iterations)):
        delta = 0
        for state in range(environment.nS):
            v = 0
            for action, action_probability in enumerate(policy[state]):
                for state_probability, next_state, reward, terminated in environment.P[state][action]:
                    v += action_probability * state_probability * (reward + discount_factor * V[next_state])

            delta = max(delta, np.abs(V[state] - v))
            V[state] = v
        evaluation_iterations += 1

        if delta < theta:
            print(f'Policy evaluated in {evaluation_iterations} iterations.')
            return V

def one_step_lookahead(environment, state, V, discount_factor):
    action_values = np.zeros(environment.nA)
    for action in range(environment.nA):
        for probability, next_state, reward, terminated in environment.P[state][action]:
            action_values[action] += probability * (reward + discount_factor * V[next_state])
    return action_values
def policy_iteration(environment, discount_factor=0.9, max_iterations=1e9):
    # num states x num actions / num actions
    policy = np.ones([environment.nS, environment.nA]) / environment.nA
    # Initialize counter of evaluated policies
    evaluated_policies = 1
    for i in range(int(max_iterations)):
        stable_policy = True
        # Evaluate current policy
        V = policy_evaluation(policy, environment, discount_factor=discount_factor)
        for state in range(environment.nS):
            current_action = np.argmax(policy[state])
            action_value = one_step_lookahead(environment, state, V, discount_factor)
            best_action = np.argmax(action_value)
            # If action didn't change
            if current_action != best_action:
                stable_policy = True
                policy[state] = np.eye(environment.nA)[best_action]
        evaluated_policies += 1
        if stable_policy:
            for i in range(0, 5):
                for j in range(0, 5):
                    V_fin[i][j] = V[(i * 5) + j]
            print(V_fin)
            for i in range(0, 5):
                for j in range(0, 5):
                    policy_table[i][j] = policy_table_comp(V_fin, i, j)
            policy_table[4][4] = 'G'
            policy_table[0][4] = 'H'
            policy_table[1][2] = 'H'
            policy_table[1][3] = 'H'
            policy_table[3][0] = 'H'
            policy_table[3][1] = 'H'
            policy_table[3][3] = 'H'
            print(policy_table)
            return  V

n_episodes = 10000
# Load a Frozen Lake environment
environment = gym.make('Stochastic-5x5-FrozenLake-v0')
a = datetime.datetime.now()
V = policy_iteration(environment)
b = datetime.datetime.now()
c = b - a
print( float(c.total_seconds()) , " S")