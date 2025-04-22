import numpy as np
import gym
from gym import spaces
import pygame

class Maze(gym.Env):
    metadata = {"render_modes": ["human","rgb_array"],"render_fps":4}

    def __init__(self,render_mode = None,size=10):
        self.size = size # The size of the square grid
        self.window_size = 512 # The size of the pygame window
        pix_square_size = (
            self.window_size/self.size
        ) # The size of a single grid square in pixels
        # Add walls to maze 
        self.walls = [
            pygame.Rect((0,pix_square_size),(pix_square_size,pix_square_size*8)),
            pygame.Rect((2*pix_square_size,pix_square_size),(pix_square_size*8,pix_square_size)),
            pygame.Rect((2*pix_square_size,3*pix_square_size),(pix_square_size,pix_square_size*4)),
            pygame.Rect((4*pix_square_size,3*pix_square_size),(pix_square_size*3,pix_square_size)),
            pygame.Rect((6*pix_square_size,4*pix_square_size),(pix_square_size,pix_square_size*2)),
            pygame.Rect((8*pix_square_size,3*pix_square_size),(pix_square_size,pix_square_size*5)),
            pygame.Rect((3*pix_square_size,6*pix_square_size),(pix_square_size*5,pix_square_size)),
            pygame.Rect((2*pix_square_size,8*pix_square_size),(pix_square_size*7,pix_square_size)),
        ]

        # We want the agent to be aware of where it is, but not where the end is
        self.observation_space = spaces.Discrete(self.size*self.size)

        # As this is a square, the agent can move "left", "down","right", "up"
        self.action_space = spaces.Discrete(4)
        
        """
        The following maps the abstract actions from action_space to 
        the directions the agent will walk when the action is chosen.
        """
        self._action_to_direction = {
            0: np.array([-1,0]), # "left"
            1: np.array([0,-1]), # "down"
            2: np.array([1,0]),  # "right"
            3: np.array([0,1])   # "up"
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        Below are necessary variables for human rendering.
        self.window: The window used to display the actions taken
        self.clock: The clock used to verify the fps
        """
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        return self._agent_location[0] * self.size + self._agent_location[1] 

    def _get_info(self):
        return  np.array([np.linalg.norm(self._agent_location - self._target_location, ord=1)])
    
    """
    Reset environment after each attempt and put agent in the starting spot. 
    No seed is needed as there is no randomness in this environment.
    """
    def reset(self):
        self._agent_location = np.array([7,9])
        self._target_location = np.array([5,5])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render_frame()
        return observation,info

    def step(self,action):
        # Map the action to the direction for the agent to walk in
        direction = self._action_to_direction[action]
        pix_square_size = (
            self.window_size/self.size
        )
        wall_collision = False
        # np.clip keeps agent within the bounds of the maze
        # colliderect checks that agent adheres to walls
        for wall in self.walls:
            next_step = self._agent_location + direction
            next_step_rect = pygame.Rect(
                pix_square_size * next_step,
                (pix_square_size,pix_square_size)
            )
            if next_step_rect.colliderect(wall):
                wall_collision = True
                break
        if not wall_collision:
            self._agent_location = np.clip(
                self._agent_location + direction,0,self.size-1
            )

        terminated = np.array_equal(self._agent_location,self._target_location)
        """
        We want to give a penalty for every time step until the agent reaches the reward.
        This will simulate a sense of urgency in the agent and lead to an optimized route.
        """
        reward = 5 if terminated else -1 
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render_frame()
        return observation,reward,terminated,False,info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size,self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size,self.window_size))
        canvas.fill((255,255,255))
        pix_square_size = (
            self.window_size/self.size
        ) # The size of a single grid square in pixels

        # Draw the target
        pygame.draw.rect(
            canvas,
            (0,255,0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size,pix_square_size)
            )
        )
        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0,0,255),
            (self._agent_location+0.5) * pix_square_size,
            pix_square_size/3
        )

        for wall in self.walls:
            pygame.draw.rect(canvas,(0,0,0),wall)
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()