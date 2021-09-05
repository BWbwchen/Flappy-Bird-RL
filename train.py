import time
import flappy_bird_gym
import pygame
from agent import Agent
from replay import ReplayRecord

EPISODE_LEN = 100
STEP_REWARD_FACTOR = 0.01
EPOCH_REWARD_FACTOR = 100
INPUT_SIZE = 2
OUTPUT_SIZE = 2

def train(epochs: int, path: str) :
    agent = Agent(INPUT_SIZE, OUTPUT_SIZE)
    env = flappy_bird_gym.make("FlappyBird-v0")
    
    for ep in range(epochs):
        obs = env.reset()

        record = ReplayRecord()

        for episode in range(EPISODE_LEN):
            # collect trajectory
            agent_action = agent.Act(obs)
            actions_distribution = agent.Action_distribution(obs)
            obs, step_reward, done, info = env.step(agent_action)

            # record trajectory
            step_reward *= STEP_REWARD_FACTOR
            record.Add(obs=obs, acts=actions_distribution)

            if done:
                break

        # add to replay buffer
        
        agent.Add_to_replay_buffer(record)

        # train policy
        agent.Train(ep+1)


        epoch_rewards = int(info['score']) * EPOCH_REWARD_FACTOR
        agent.Log("Epoch {}/{} | Finished ! the reward this epoch is : {}.".format(ep+1, epochs, epoch_rewards))

        if (ep+1) % 10 == 0 :
            agent.Save(path)

    agent.Save(path)
    exit()


    while True:
        # Next action:
        # (feed the observation to your agent here)

        action = human_action()

        # Processing:
        obs, reward, done, info = env.step(action)
        
        # Rendering the game:
        # (remove this two lines during training)
        env.render()
        time.sleep(1 / 30)  # FPS
        
        # Checking if the player is still alive
        if done:
            print("You died!")
            break

    env.close()


if __name__ == '__main__' :
    train(2, './model.h5')
