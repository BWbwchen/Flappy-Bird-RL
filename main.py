import time
import flappy_bird_gym
import random
import pygame

def human_action() -> int :
    action = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if (event.type == pygame.KEYDOWN and
                (event.key == pygame.K_SPACE or event.key == pygame.K_UP)):
            action = 1

    return action

def play() :
    env = flappy_bird_gym.make("FlappyBird-v0")

    obs = env.reset()
    pygame.init()

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
    play()
