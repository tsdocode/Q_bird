"""
Author: Nguyen Thanh Sang
SID : 19133048 
N.O : 42
"""

import os, sys
sys.path.append('../Environment')
sys.path.append('../utils')

from collections import defaultdict
import json
import random
import numpy as np
import gym
from Environment.FlappyGame import FlappyBirdNormal
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

class Qlearning():
    def __init__(self, actions , epsilon, flaprobs = 0.5, rounding = None) -> None:
        """Khởi tạo các tham số

        Args:
            actions (list): [description]
            epsilon (double): [description]
            flaprobs (float, optional): Tỉ lệ ưu cho action bay. Defaults to 0.5.
            rounding ([type], optional): Hệ số làm tròn. Defaults to None.
        """
        self.actions = actions
        self.epsilon = epsilon
        self.flaprobs = flaprobs
        self.env = FlappyBirdNormal(gym.make('FlappyBird-v0'), rounding = rounding)
        self.qTable = defaultdict(lambda : np.zeros(len(actions)))
        

    def randomFlap(self):
        """Chọn action ngẫu nhiên

        Returns:
            [int]: action bay hoặc không bay
        """
        if (np.random.random() < self.flaprobs):
            return 0
        return 1 

    def act(self, state):
        """ Chọn action bằng thuật toán epsilon-greedy

        Args:
            state (string): trạng thái hiện tại

        Returns:
            [int]: action
        """
        p = np.random.random()
        if p < self.epsilon:
            return np.random.choice(len(self.actions))
        else:
            if (self.qTable[state][0] == self.qTable[state][1]):
                return self.randomFlap()
            else:
                return np.argmax(self.qTable[state])

    def test_act(self, state):
        """Chọn action trong quá trình test bằng thuật toán greedy

        Args:
            state ([type]): [description]

            state (string): trạng thái hiện tại

        Returns:
            [int]: action
        """
        # "Test chiến thuật bằng thuật toán tham lam  "
        return np.argmax(self.qTable[state])

    def test(self, numIters = 500):
        """Test mô hình

        Args:
            numIters (int, optional): Số lần test. Defaults to 500.

        Returns:
            [dictionary]: tần số các điểm đạt được
        """
        self.epsilon = 0
        self.env.seed(0)

        done = False
        maxScore = 0
        maxReward = 0
        output = defaultdict(int)
        t  = tqdm(range(numIters))
        for i in t:
            t.set_description("max Score = " + str(maxScore))
            score = 0
            totalReward = 0
            ob = self.env.reset()
            state = self.env.getGameState()
            
            while True:
                action = self.test_act(state)
                state, reward, done, _ = self.env.step(action)
                # self.env.render()  # Uncomment it to display graphics.
                totalReward += reward
                if reward >= 1:
                    score += 1
                if done:
                    break
                    
            output[score] += 1
            if score > maxScore: maxScore = score
            if totalReward > maxReward: maxReward = totalReward
    
        self.env.close()
        print("Max Score Test: ", maxScore)
        print("Max Reward Test: ", maxReward)
        print()
        return output
        
    def saveQValues(self, name):
        """Lưu kết quả dưới dạng json

        Args:
            name (string): tên file
        """
        toSave = {key : list(self.qTable[key]) for key in self.qTable}
        # print(toSave)
        with open(f'qValues_{name}.json', 'w') as fp:
            json.dump(toSave, fp)

    def saveOutput(self, output, iter , name):
        """Lưu lại kết quả đạt được

        Args:
            output (dictionary): tần xuất điểm đạt được
            iter (int): số vòng lặp
            name (string): tên file
        """
        if not os.path.isdir('scores'):
            os.mkdir('scores')
        with open(f'./scores/{name}/scores_{iter}.json', 'w') as fp:
            json.dump(output, fp)

    def loadQValues(self):
        """Nhập Q-value từ file Json

        Returns:
            [dictionary]: q-table
        """
        with open('qValues_grid20.json', 'r') as fp:
            toLoad = json.load(fp)
            qValues = {key : np.array(toLoad[key]) for key in toLoad}
            return qValues

    def play(self, numIters):
        """Chơi thử khi train xong

        Args:
            numIters (int): Số lần chơi

        Returns:
            [type]: [description]
        """
        q = self.loadQValues()
        for key in q:
            self.qTable[key] = q[key] 
        self.epsilon = 0
        self.env.seed(0)

        done = False
        maxScore = 0
        maxReward = 0
        output = defaultdict(int)
        t = tqdm(range(numIters))
        for i in t:
            score = 0
            totalReward = 0
            ob = self.env.reset()
            state = self.env.getGameState()
            
            while True:
                action = self.test_act(state)
                state, reward, done, _ = self.env.step(action)
                self.env.render()  
                totalReward += reward
                if reward >= 1:
                    score += 1
                if done:
                    break
            t.set_postfix({
                "score" : score
            }, refresh=True)      
            output[score] += 1
            if score > maxScore: maxScore = score
            if totalReward > maxReward: maxReward = totalReward
            t.set_description("Max Score = " + str(maxScore), refresh=True)
            self.env.save_output("./video/" + str(i))
    
        self.env.close()
        print("Max Score Test: ", maxScore)
        print("Max Reward Test: ", maxReward)
        print()
        return output


    def train(self, episodes = 2000, gamma = 0.9, lr = 0.8):
        """Train agent bằng thuật toán Q-learning

        Args:
            episodes (int, optional): Số training episode . Defaults to 2000.
            gamma (float, optional): Hệ số gamma. Defaults to 0.9.
            lr (float, optional): Learning rate. Defaults to 0.8.
        """
        done = False
        score = 0
        maxScore = 0
        maxReward = 0
        self.lr = lr
        self.gamma = gamma
        output = defaultdict(int)
        t = tqdm(range(episodes))
        for ep in t:
            t.set_description("Train Score = " + str(maxScore))
            if (ep + 1) % 300 == 0:
                output = self.test(numIters = 100)
                self.saveOutput(output, ep + 1 , "grid_20")
                self.saveQValues("grid20")
            score = 0
            totalReward = 0
            gameEP = []
            ob = self.env.reset()
            state = self.env.getGameState()
            
            while True:
                action = self.act(state)
                nextState , reward , done , _ = self.env.step(action)
                gameEP.append(
                    (state, action , reward, nextState)
                )
                state = nextState

                totalReward += reward
                if reward >= 1:
                    score += 1
                if done:
                    break
            t.set_postfix({
                "score" : score
            }, refresh=True) 

            output[score] += 1
            maxScore = max(score, maxScore)
            maxReward = max(reward, maxReward)

            for (state, action, reward, nextState) in gameEP[::-1]:
                self.updateQ(state, action, reward, nextState)      
        self.env.close()
        
            
    def updateQ(self, state, action, reward, nextState):
        """Cập nhật Q-value

        Args:
            state ([type]): trạng thái
            action ([type]): hành động
            reward ([type]): điểm thưởng
            nextState ([type]): trạng thái kế tiếp
        """
        nextQ = [self.qTable[(nextState)]]
        nextValue = np.max(nextQ)
        self.qTable[state][action] = (1-self.lr) * self.qTable[state][action] + \
                                        self.lr*(reward + self.gamma * nextValue)

  