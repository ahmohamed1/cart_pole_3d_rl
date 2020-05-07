
## Train the agent
episodes = 5000
batch_size = 32
done = False
scores = []
sync_freq = 500
# initialize the enviroment and the agent

state_size = 64
action_size = 4
agent = DQNAgent(state_size,action_size)
# agent.load("./save/cartpole-ddqn.h5"

# iterate the game
for e in range(episodes):
    env = Gridworld(mode='random')
    # reset and get the atate every game
    state = env.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
    # loop throught the 500 frame of the game
    for time_t in range(40):
        # In case we want to show the render
        #env.render()

        action = agent.act(state)
        # play this action and get the new_stat reward and done
        # Reward is 1 for every frame the pole survived
        env.makeMove(action_set[action])
        # reshape the next_state
        next_state = env.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        reward = env.reward()
        done = True if reward > 0 else False
        agent.memorize(state, action, reward, next_state, done)
        state = next_state

        # traing the agent with the experience
        if len(agent.memory) > batch_size:
            #agent.replay(batch_size)
            loss = agent.replay_improved(batch_size)
            print(e, loss)
            clear_output(wait=True)
            if e % sync_freq == 0:
                # Copies the main model parameters to the target network
                agent.update_target_model()
        # If the game is over
        if reward != -1:
            break
    if e % 10 == 0:
        agent.save("./save/gridworld-ddqn.h5")
