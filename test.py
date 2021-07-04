from Agent.QLearning import Qlearning

bird = Qlearning(actions = [0,1] , epsilon = 0.1, rounding= 12, flaprobs=0.1)


bird.play(1)