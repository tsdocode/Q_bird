"""
Author: Nguyen Thanh Sang
SID : 19133048 
N.O : 42
"""


import numpy as np
import gym
from gym.wrappers import Monitor
import gym_ple
from gym_ple import PLEEnv
from utils.utils import *

import warnings
warnings.filterwarnings('ignore')


class FlappyBirdNormal(gym.Wrapper):
    ''' 
        Môi trường đơn giản đơn giản
        State = [birdVel , xDistance, yDistance]
            + bird_vel : Tốc độ của chim 
            + xDistance: Khoảng cách của chim đến cột gần nhất theo phương x
            + yDistance: Khoảng cách của chim đến mặt trên cột nằm dưới theo phương y
    '''
    
    def __init__(self, env, rounding = None):
        '''
      
        '''
        super().__init__(env)
        self.rounding = rounding
    
    def save_output(self, outdir = None):
        '''
        Lưu kết quả chơi dưới dạng video 
        '''
        if outdir:
            self.env = Monitor(self.env, directory = outdir, force = True)
        
    def step(self, action):
        '''
        Agent nhận state mới , reward và tính hiệu kết thúc game 
        
        Args:
            action (int): 0 or 1.
        
        Returns:
            tuple: state, reward, terminal.
        '''
        _, reward, terminal, _ = self.env.step(action)
        state = self.getGameState()
        if not terminal: reward += 0.5
        else: reward = -1000
        if reward >= 1: reward = 5
        return state, reward, terminal, {}

    def getGameState(self):
        '''
        Xử lý trả về cuả state 
        
        Returns:
            str: Chuỗi mô tả state
        '''
        gameState = self.env.game_state.getGameState()
        hor_dist_to_next_pipe = gameState['next_pipe_dist_to_player']
        ver_dist_to_next_pipe = gameState['next_pipe_bottom_y'] - gameState['player_y']
        if self.rounding:
            hor_dist_to_next_pipe = discretize(hor_dist_to_next_pipe, self.rounding)
            ver_dist_to_next_pipe = discretize(ver_dist_to_next_pipe, self.rounding * 2)
            
        state = []
        state.append('birdVel' + ' ' + str(gameState['player_vel']))
        state.append('xDistance' + ' ' + str(hor_dist_to_next_pipe))
        state.append('yDistance' + ' ' + str(ver_dist_to_next_pipe))
        return ' '.join(state)
        