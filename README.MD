


# Final project for AI class - HCMUTE


<p align="center">
  <img width="250" height="300" src="./image/visual.png">
</p>

> by tsdocode


## **1. Installation**

**1.1 Cài đặt môi trường Anaconda**
```
conda create -n flappy python=3.7
conda activate flappy
```

**1.2 Cài đặt thư viện**
```
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip install -e .
cd ..
pip install -r requirements.txt
```

**1.3 Sửa lỗi xung đột phiên bản**
```python
#time.clock => 
time.process_time 
```

## **2. Training**
> In train.py
```python
from Agent.QLearning import Qlearning

bird = Qlearning(actions= [0,1],epsilon=0.1,flaprobs=0.4,rounding=12)

bird.train(episodes=10000, gamma=0.8, lr=0.8)
```
>In terminal

```bash
python train.py
```



## **3. Evaluation**
**3.1 Kiểm thử hiệu quả agent**
```python 
from Agent.QLearning import Qlearning

bird = Qlearning(actions = [0,1] , epsilon = 0.1, rounding= 12, flaprobs=0.1)

bird.play(1) # số lần thử
```



## **4. Reference**
Vu, Tai & Tran, Leon. (2020). FlapAI Bird: Training an Agent to Play Flappy
Bird Using Reinforcement Learning Techniques. [[paper](https://arxiv.org/abs/2003.09579)]