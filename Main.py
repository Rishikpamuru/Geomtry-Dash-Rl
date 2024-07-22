# Imports

import cv2
import pyautogui
import numpy as np

import gym
from gym import spaces
import time

import pygetwindow as gw

from stable_baselines3 import PPO
class GeomtryDashEnviorment(gym.Env):
    def __init__(self):
        super(GeomtryDashEnviorment, self).__innit__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low = 0, high=255,shape=(600,800,3), dtype=np.uint8)
        self.game_region= None
    
    # Resets the game when bot dies
    def reset(self):
        pyautogui.click(x=850,y=1580)
        time.sleep(0.1)
        return capture_screen(region=self.game_region)
    
    # Plays the game
    def step(self, action):
        if action == 1:
            pyautogui.press('space')
        time.sleep(0.05)
        frame = capture_screen(region=self.game_region)
        done = self.detect_death(frame) or self.detect_completion(frame)
        reward = -1 if self.detect_death(frame) else 0.1
        if self.detect_completion(frame):
            reward += 10  # Higher reward for completing the level
        return frame, reward, done, {}

    
    def detect_completion(self,frame):
        vict_region = frame[607:645,439:520]
        return np.mean(vict_region) < 47

    # Detects for the reset button in game
    def detect_death(self, frame):
        death_region = frame[758:954, 1475:1646]
        return np.mean(death_region) < 115
    
    
    def render(self, mode = 'human'):
        frame = capture_screen(region=self.game_region)
        cv2.imshow("Geometry Dash", frame)
        cv2.waitKey(1)

    # Closes the screenshot window
    def close(self):
        cv2.destroyAllWindows()



# Function to capture the oppend screen 
def capture_screen(region=None):
    screenshot = pyautogui.screenshot(region=region) # Takes the screenshot
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Moves to Top
gameWindow = gw.getWindowsWithTitle('Geometry Dash')[0]

gameWindow.moveTo(0, 0)
gameWindow.activate()

# Takes screenshot
time.sleep(1)
frame = capture_screen(region=None)

# Shows the immage
cv2.imshow("Game Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Makes the enviornment for the model
enviorment = GeomtryDashEnviorment()

# Trains the model
model = PPO('CnnPolicy',enviorment,verbose = 1)
model.learn(total_timesteps=2000)

# Test the trained model
obs = enviorment.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = enviorment.step(action)
    enviorment.render()  # Call render to display the game frame
    if done:
        if enviorment.detect_completion(obs):
            print("Level completed")
        else:
            print("Bot died")
        break  # Exit the loop when the game is done


def capture_sample(region):
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# Define regions of interest (adjust coordinates based on your game)
death_region_coords = (1475, 758, 171, 196)  # Example coordinates for death region
completion_region_coords = (50, 50, 150, 50)  # Example coordinates for completion region

# Capture sample screenshots
death_region_coords = (758, 1475, 196, 171)
completion_region_coords = (607,439,38,81)


death_sample = capture_sample(death_region_coords)
completion_sample = capture_sample(completion_region_coords)

# Calculate mean pixel values
death_mean_intensity = np.mean(death_sample)
completion_mean_intensity = np.mean(completion_sample)

print(f"Mean intensity for death region: {death_mean_intensity}")
print(f"Mean intensity for completion region: {completion_mean_intensity}")
