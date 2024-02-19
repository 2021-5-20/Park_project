import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
park_num = 1000
np.random.seed(5)
user_to_park_time = [np.random.randint(5,60) for j in range(park_num)] # 用户到停车场的时间
user_to_target = [np.random.randint(20,1000) for i in range(park_num)] # 用户停车后到目标地的距离
park_remainder = [np.random.randint(0,4000)/4000 for i in range(park_num)]  # 停车场剩余的车位数
park_cost = [np.random.randint(5,50) for i in range(park_num)]        # 停车场费用
#user_to_park_time = [15 for i in range(50)]
#park_cost = [15 for i in range(50)]
#park_remainder = [100 for i in range(50)]

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
scalar = StandardScaler()


user_to_park_time_ = np.round(normalization(user_to_park_time),3)
#user_to_target_ = np.round(normalization(user_to_target),3)
park_remainder_ = np.round(normalization(park_remainder),3)
park_cost_ = np.round(normalization(park_cost),3)


distance_x = [np.random.randint(0,3000) for i in range(park_num)] 
distance_y = [np.random.randint(0,3000) for i in range(park_num)] 
#停车场距离和信息素
distance_graph = [ [0.0 for col in range(park_num)] for raw in range(park_num)]
pheromone_graph = [ [1.0 for col in range(park_num)] for raw in range(park_num)]
for i in range(park_num):
        for j in range(park_num):    
                temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2)
                temp_distance = pow(temp_distance, 0.5)
                distance_graph[i][j] =float(int(temp_distance + 0.5))

#print(user_to_park_time_)
#print(user_to_target_)
#print(park_remainder_)
#print(park_cost_)

distance_graph_ = np.round(normalization(distance_graph),3)

#temp = (1/distance_graph_[self.current_city][i] + 1 / park_remainder_  + 1 / user_to_target_ + 1 / user_to_park_time_)
distance_to_park = [distance_graph[i][park_num - 1] for i in range(park_num)]
distance_to_park = np.array(distance_to_park)

rank_index = distance_to_park.argsort()

 # 获取前一百个靠的近的停车场排名
rank_index = rank_index[:100]
w = [[],[1,1,1],[1,1,5],[1,5,1]]
#print(distance_to_park)
#print(distance_graph_[0][12]) # 起点到12号停车场的距离
#print(park_remainder_[12])  # 停车场停车率
#print(user_to_park_time_[12]) # 用户到停车场的时间   
#print(park_cost_[12]) # 停车场费用
#park_cost_[12] = 0
#park_remainder_[12] = 0
#user_to_park_time_[12] = 0