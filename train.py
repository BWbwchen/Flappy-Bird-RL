import time
import flappy_bird_gym
from numpy.lib.npyio import load
import pygame
from agent import Agent
from replay import ReplayRecord

EPISODE_LEN = 1000
STEP_REWARD_FACTOR = 0.5
EPOCH_REWARD_FACTOR = 100
INPUT_SIZE = 2
OUTPUT_SIZE = 2

EPOCH_INTERVAL=50

def train(epochs: int, path=None, load_model=False, model_path=None) :
    agent = Agent(INPUT_SIZE, OUTPUT_SIZE, load_model, model_path)
    env = flappy_bird_gym.make("FlappyBird-v0")
    
    for ep in range(epochs):
        obs = env.reset()

        record = ReplayRecord()

        trajectory_reward = 0

        for episode in range(EPISODE_LEN):
            # collect trajectory
            agent_action = agent.Act(obs)
            actions_distribution = agent.Action_distribution(obs)
            obs, step_reward, done, info = env.step(agent_action)

            # record trajectory
            step_reward *= STEP_REWARD_FACTOR
            record.Add(obs=obs, acts=actions_distribution)

            trajectory_reward += step_reward

            if done:
                break

        # add to replay buffer
        
        agent.Add_to_replay_buffer(record)

        # train policy
        agent.Train(ep+1)


        epoch_rewards = int(info['score']) * EPOCH_REWARD_FACTOR
        trajectory_reward += epoch_rewards

        if (ep+1) % EPOCH_INTERVAL == 0 :
            agent.Log("Epoch {}/{} | Finished ! the reward this epoch is : {}.".format(ep+1, epochs, trajectory_reward))

        #if (ep+1) % 10 == 0 :
        #    agent.Save(path)

    if epochs % 10 != 0 :
        agent.Save(path)
    env.close()


if __name__ == '__main__' :
    train(200001, 'test.h5', load_model=False, model_path='./model/model.h5')
