from Agent.QLearning import Qlearning


bird = Qlearning(actions= [0,1],epsilon=0.1,flaprobs=0.4,rounding=12)


bird.train(episodes=10000, gamma=0.8, lr=0.8)