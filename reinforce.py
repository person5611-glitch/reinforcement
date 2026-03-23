# import numpy as np
# import random
# # States: 0,1,2,3(2x2 grid)
# #Actions: 0=up, 1=down, 2=left, 3=right
# Q = np.zeros((4,4))
# alpha = 0.1
# gamma = 0.9
# epsilon = 0.2

# def get_reward(state):
#     if state == 3:
#         return 1
#     return 0


# def get_next_state(state, action):
#     if action == 3 and state < 3:
#         return state + 1
#     elif action == 1 and state < 2:
#         return state + 2
#     return state


# #Training
# for episode in range(1000):
#     state = 0

#     while state != 3:
#         if random.uniform(0,1) < epsilon:
#             action = random.randint(0,3)
#         else:
#             action = np.argmax(Q[state])

#         next_state = get_next_state(state, action)
#         reward = get_reward(next_state)
    
#         Q[state, action] = Q[state, action] + alpha *(reward + gamma*np.max(Q[next_state]) - Q[state, action])

#         state = next_state
        
# print("Q-table:")
# print(Q)





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
maze = np.array([
[0,0,0,0],
[0,1,1,0],
[0,0,0,0],
[1,0,1,0]
])

start = (0 , 0)
goal = (4,4)

num_episode = 5000
alpha  = 0.1
gamma = 0.9
epsilon = 0.5

reward_fire = -10
reward_goal = 50
reward_step = -1
actions = [(0, -1),(0, 1),(-1, 0),(1, 0)]
Q = np.zeros(maze.shape+(len(actions),))

def is_valid(pos):
    r, c = pos
    if r<0 or r>= maze.shape[0]:
        return False
    if c < 0 or c>=maze.shape[1]:
        return False
    if maze[r, c] == 1:
        return False
    return True

def choose_actions(state):
    if np.random.randint() < epsilon:
        return np.random.randint(len(actions))
    else:
        return np.argmax(Q[state])
    
reward_all_episode = []
for episode in range(num_episode):
    state = start 
    total_reward = 0
    done = False

    while not done:
        action_index = choose_actions(state)
        action = actions[action_index]

        next_state = (state[0]+action[0], state[1]+action[1])

        if not is_valid(next_state):
            reward = reward_fire
            done = True
        elif next_state == goal:
            reward = reward_goal
            done = True
        else:
            reward = reward_step
        old_value = Q[state][action_index]
        next_max = np.max(Q[next_state]) if is_valid(next_state) else 0

        Q[state][action_index] = old_value + alpha * \
            (reward + gamma * next_max - old_value)
        
        state = next_state
        total_reward += reward
    
    global epsilon
    epsilon = max(0.01, epsilon * 0.995)
    reward_all_episode.append(total_reward)

    def get_optimal_path(Q, start, goal, actions, maze, max_steps = 200):
        path = [start]
        state = start
        visited = set()

        best_actions = None
        best_value = -float('inf')

        for idx, move in enumerate(actions):
            next_state = (state[0] + move[0], state[1] + move[1])

            if (0 <= next_state[0] < maze.shape[0] and 
                0 <= next_state[1] < maze.shape[1] and
                maze[next_state] == 0 and
                    next_state not in visited):
                
                if Q[state][idx] > best_value:
                    best_value = Q[state][idx]
                    best_actions = idx

        if best_actions is None:
            break

        move = actions[best_actions]
        state = (state[0] + move[0], state[1] + move[1])
        path.append(state)

        return path 
    optimal_path = get_optimal_path(Q, start, goal, action, maze)

def plot_maze_with_path(path):
    cmap = ListedColormap(['#eef8ea', '#a8c79c'])

    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap=cmap)
    plt.scatter(start[1], start[0], marker='o', color = '#81c784', edgecolors='black',
                s=200, label = 'Start(Robot)', zorder = 5)
    
    plt.scatter(goal[1], goal[0], marker='*', color = '388e3c', edgecolors='black', s = 300, label= 'Goal(Diamond)', zorder = 5)
    
    rows, cols = zip(*path)
    plt.plot(cols, rows, color = '#60b37a',linewidth = 4, label = 'Learned Path', zorder = 4)

    plt.title('Reinforcement Learning: Robot maze navigation')
    plt.gca().invert_yaxis()
    plt.xticks(range(maze.shape[1]))
    plt.yticks(range(maze.shape[0]))
    plt.grid(True, alpha = 0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_maze_with_path(optimal_path)














# import numpy as np
# import random 
# alpha = 0.1
# gamma = 0.9
# epsilon = 0.2
# Q = np.zeros((4, 4))

# print(Q)
# # Reward Function
# def get_reward(state):
#     if state == 3:
#         return 1
#     return 0

# def get_next_state(state, action):
#     if action == 3 and state < 3:
#         return state+1
#     elif action == 1 and state <2:
#         return state+2
#     return state

# for episode in range(1000):
#     state = 0

#     while state != 3:
#         if random.uniform(0, 1) < epsilon:
#             action = random.randint(0,3)
#         else:
#             action = np.argmax(Q[state])

#     Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max)


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

# # Maze definition
# maze = np.array([
#     [0,0,0,0,0],
#     [0,1,1,0,0],
#     [0,0,0,0,0],
#     [1,0,1,0,0],
#     [0,0,0,0,0]
# ])

# start = (0,0)
# goal = (4,4)

# # Parameters
# num_episode = 5000
# alpha  = 0.1
# gamma = 0.9
# epsilon = 0.5

# reward_fire = -10
# reward_goal = 50
# reward_step = -1

# actions = [(0,-1),(0,1),(-1,0),(1,0)]

# Q = np.zeros(maze.shape + (len(actions),))

# # Check valid position
# def is_valid(pos):
#     r, c = pos
#     if r < 0 or r >= maze.shape[0]:
#         return False
#     if c < 0 or c >= maze.shape[1]:
#         return False
#     if maze[r,c] == 1:
#         return False
#     return True


# # Epsilon-greedy action selection
# def choose_action(state):
#     if np.random.rand() < epsilon:
#         return np.random.randint(len(actions))
#     else:
#         return np.argmax(Q[state])


# reward_all_episode = []

# # Training loop
# for episode in range(num_episode):

#     state = start
#     total_reward = 0
#     done = False

#     while not done:

#         action_index = choose_action(state)
#         action = actions[action_index]

#         next_state = (state[0] + action[0], state[1] + action[1])

#         if not is_valid(next_state):
#             reward = reward_fire
#             done = True

#         elif next_state == goal:
#             reward = reward_goal
#             done = True

#         else:
#             reward = reward_step

#         old_value = Q[state][action_index]

#         next_max = np.max(Q[next_state]) if is_valid(next_state) else 0

#         Q[state][action_index] = old_value + alpha * (
#             reward + gamma * next_max - old_value
#         )

#         if is_valid(next_state):
#             state = next_state

#         total_reward += reward

#     epsilon = max(0.01, epsilon * 0.995)

#     reward_all_episode.append(total_reward)


# # Find optimal path
# def get_optimal_path(Q, start, goal, actions, maze, max_steps=200):

#     path = [start]
#     state = start
#     visited = {state}

#     for _ in range(max_steps):

#         if state == goal:
#             break

#         best_action = np.argmax(Q[state])
#         move = actions[best_action]

#         next_state = (state[0] + move[0], state[1] + move[1])

#         if not is_valid(next_state) or next_state in visited:
#             break

#         path.append(next_state)
#         visited.add(next_state)
#         state = next_state

#     return path


# optimal_path = get_optimal_path(Q, start, goal, actions, maze)


# # Plot maze and path
# def plot_maze_with_path(path):

#     cmap = ListedColormap(['#eef8ea', '#a8c79c'])

#     plt.figure(figsize=(8,8))
#     plt.imshow(maze, cmap=cmap)

#     plt.scatter(start[1], start[0],
#                 marker='o',
#                 color='#81c784',
#                 edgecolors='black',
#                 s=200,
#                 label='Start (Robot)',
#                 zorder=5)

#     plt.scatter(goal[1], goal[0],
#                 marker='*',
#                 color='#388e3c',
#                 edgecolors='black',
#                 s=300,
#                 label='Goal (Diamond)',
#                 zorder=5)

#     rows, cols = zip(*path)
#     plt.plot(cols, rows,
#              color='#60b37a',
#              linewidth=4,
#              label='Learned Path',
#              zorder=4)

#     plt.title('Reinforcement Learning: Robot Maze Navigation')

#     plt.gca().invert_yaxis()
#     plt.xticks(range(maze.shape[1]))
#     plt.yticks(range(maze.shape[0]))

#     plt.grid(True, alpha=0.2)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# plot_maze_with_path(optimal_path)