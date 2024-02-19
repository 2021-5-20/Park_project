"""
    权重    
    1. 距离
    2. 拥堵情况(到达时间)
    3. 车位空余
    4. 收费
    5. 停车点到目标的距离
"""
import random
import copy
import time
from venv import create
import data
import sys
import math
import tkinter #//GUI模块
import threading
import numpy as np
import matplotlib.pyplot as plt

#各区域在图中的坐标
city_num = 35
start_index = 0
end_index = 1
w = [0.25,0.25,0.25,0.25]
distance_graph = [ [0.0 for col in range(city_num)] for raw in range(city_num)]
pheromone_graph = [ [1.0 for col in range(city_num)] for raw in range(city_num)]
park_x = [129, 165, 126, 130, 168, 248, 340, 339, 389, 509, 537, 561, 512, 156, 223, 144,
233, 322,359,670,705, 643, 712, 871, 899, 966, 989, 999, 969, 879, 903, 934, 949, 930, 951]

park_y = [126, 138, 194, 224, 210, 211, 133, 203, 219, 161, 152, 154, 222, 359, 386, 449, 
455, 505, 570, 323, 361, 411, 431, 285, 301, 291, 299, 318, 329, 364, 355, 352, 393, 486, 570]

traffic_x = [96, 337, 799, 374, 828, 1060, 878, 968, 53]
traffic_y = [277, 284, 199, 561, 474, 112, 576, 675, 601]

dest_x = [200,500]
dest_y = [60,400]

"""
distance_x = [
    178,272,176,171,650,499,267,703,408,437,491,74,532,
    416,626,42,271,359,163,508,229,576,147,560,35,714,
    757,517,64,314,675,690,391,628,87,240,705,699,258,
    428,614,36,360,482,666,597,209,201,492,294]
distance_y = [
    170,395,198,151,242,556,57,401,305,421,267,105,525,
    381,244,330,395,169,141,380,153,442,528,329,232,48,
    498,265,343,120,165,50,433,63,491,275,348,222,288,
    490,213,524,244,114,104,552,70,425,227,331]
"""
#各区域间的收费 yuan / hour   # 50
money_cost = [
    10,15,10,10,10,10,5,15,10,10,5,15,10,
    5,0,10,0,15,10,5,0,10,10,15,10,5,
    10,5,5,10,15,10,10,10,15,10,10,15,10,
    10,5,10,5,15,10,10,10,5,10,5]
#到达各个点的路程时间
time_cost = [
    10,15,10,10,10,10,5,15,10,10,5,15,10,
    5,0,10,0,15,10,5,0,10,10,15,10,5,
    10,5,5,10,15,10,10,10,15,10,10,15,10,
    10,5,10,5,15,10,10,10,5,10,5]


#各区域的车位空余
Park_nums = [
    178,5,189,65,0,485,267,52,69,437,265,74,532,
    254,423,100,265,412,163,125,229,254,147,168,35,714,
    757,517,64,124,258,625,391,628,87,240,423,555,258,
    894,562,36,547,178,160,255,345,201,258,294]

#均值归一化于标准化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
 

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

time_cost = np.array(time_cost)
time_cost = normalization(time_cost)

money_cost = np.array(money_cost)
money_cost = normalization(money_cost) # 收费均值归一完成

Park_nums = np.array(Park_nums)
Park_nums = normalization(Park_nums)
data = np.append(money_cost,Park_nums)
data = np.append(data,time_cost)
#print(data)



class TSP(object):
 
    def __init__(self, root, width = 1000, height = 600, n = city_num):
 
        # 创建画布
        self.root = root                               
        self.width = width      
        self.height = height
        # 城市数目初始化为city_num
        self.n = n
        # tkinter.Canvas
        self.canvas = tkinter.Canvas(
                root,
                width = self.width,
                height = self.height,
                bg = "#EBEBEB",             # 背景白色 
                xscrollincrement = 1,
                yscrollincrement = 1
            )
        self.canvas.pack(expand = tkinter.YES, fill = tkinter.BOTH)
        self.title("TSP蚁群算法(n:初始化 e:开始搜索 s:停止搜索 q:退出程序)")
        self.__r = 7
        self.__lock = threading.RLock()     # 线程锁
 
        self.__bindEvents()
        self.new()
 
        # 计算城市之间的距离
        for i in range(city_num):
            for j in range(city_num):
                temp_distance = pow((park_x[i] - park_x[j]), 2) + pow((park_y[i] - park_y[j]), 2)
                temp_distance = pow(temp_distance, 0.5)
                distance_graph[i][j] =float(int(temp_distance + 0.5)) # 已经计算好各点之间的距离
        
    
 
    # 按键响应程序
    def __bindEvents(self):
 
        self.root.bind("q", self.quite)        # 退出程序
        self.root.bind("n", self.new)          # 初始化
        self.root.bind("e", self.search_path)  # 开始搜索
        self.root.bind("s", self.stop)         # 停止搜索
 
    # 更改标题
    def title(self, s):
        self.root.title(s)
 
    # 初始化
    def new(self, evt = None):
 
        # 停止线程
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
 
        self.clear()     # 清除信息 
        self.nodes_park = []  # 节点坐标
        self.nodes2 = [] # 节点对象
        self.nodes_traffic = [] # 交通节点
        # 初始化城市节点
        for i in range(len(dest_x)):
            x = dest_x[i]
            y = dest_y[i]
            node = self.canvas.create_rectangle(x - self.__r,
                        y - self.__r, x + self.__r, y + self.__r,
                        fill = "#38B0DE",      # 填充红色
                        outline = "#000000",   # 轮廓白色
                        tags = "node",
                    )
            self.canvas.create_text(x,y-10,              # 使用create_text方法在坐标（302，77）处绘制文字
                        text = '('+str(x)+','+str(y)+')',    # 所绘制文字的内容
                        fill = 'black'                       # 所绘制文字的颜色为灰色
                    )       
        for i in range(len(park_x)):
            # 在画布上随机初始坐标
            x = park_x[i]
            y = park_y[i]
            self.nodes_park.append((x, y))
            # 生成节点椭圆，半径为self.__r
            node = self.canvas.create_oval(x - self.__r,
                    y - self.__r, x + self.__r, y + self.__r,
                    fill = "#ff0000",      # 填充红色
                    outline = "#000000",   # 轮廓白色
                    tags = "node",
                )
            self.nodes2.append(node)
            # 显示坐标
            self.canvas.create_text(x,y-10,              # 使用create_text方法在坐标（302，77）处绘制文字
                    text = '('+str(x)+','+str(y)+')',    # 所绘制文字的内容
                    fill = 'black'                       # 所绘制文字的颜色为灰色
                )
        for i in range(len(traffic_x)):
            # 在画布上随机初始坐标
            x = traffic_x[i]
            y = traffic_y[i]
            self.nodes_traffic.append((x, y))
            # 生成节点椭圆，半径为self.__r
            node = self.canvas.create_oval(x - self.__r,
                    y - self.__r, x + self.__r, y + self.__r,
                    fill = "#D9D9F3",      # 填充红色
                    outline = "#000000",   # 轮廓白色
                    tags = "node",
                )
            self.nodes2.append(node)
            # 显示坐标
            self.canvas.create_text(x,y-10,              # 使用create_text方法在坐标（302，77）处绘制文字
                    text = '('+str(x)+','+str(y)+')',    # 所绘制文字的内容
                    fill = 'black'                       # 所绘制文字的颜色为灰色
                )
        
        
        # 顺序连接城市
        #self.line(range(city_num))
        
    # 清除画布
    def clear(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)
 
    # 退出程序
    def quite(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.root.destroy()
        print (u"\n程序已退出...")
        sys.exit()
 
    # 停止搜索
    def stop(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        
    # 开始搜索
    def search_path(self, evt = None):
 
        # 开启线程
        self.__lock.acquire()
        self.__running = True
        self.__lock.release()
        
        while self.__running:
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path()
                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            # 更新信息素
            self.__update_pheromone_gragh()
            print (u"迭代次数：",self.iter,u"最佳路径总距离：",int(self.best_ant.total_distance-1000000))
            # 连线
            self.line(self.best_ant.path)
            # 设置标题
            self.title("TSP蚁群算法(n:随机初始 e:开始搜索 s:停止搜索 q:退出程序) 迭代次数: %d" % self.iter)
            # 更新画布
            self.canvas.update()
            self.iter += 1
 
    # 更新信息素
    def __update_pheromone_gragh(self):
 
        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        for ant in self.ants:
            for i in range(ant.path.__len__()):
                start, end = ant.path[i-1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += Q / ant.total_distance  #蚁圈模型（需改进）
                temp_pheromone[end][start] = temp_pheromone[start][end]
 
        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(ant.path.__len__()):
            for j in range(ant.path.__len__()):
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]
 
    # 主循环
    def mainloop(self):
        self.root.mainloop()
 
#----------- 程序的入口处 -----------
                
if __name__ == '__main__':
    TSP(tkinter.Tk()).mainloop()