import random
from data_help import *
import copy
import time
#import data
import sys
import math
from tkinter import ALL, EventType
import tkinter #//GUI模块
import threading
from functools import reduce
import torch 
model1 = torch.load('model_params_cost.pth')
model2 = torch.load('model_params2_time.pth')
model3 = torch.load('model_params3_distance.pth')
# 参数

'''
ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
      ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
BETA:Beta值越大,蚁群越就容易选择局部较短路径,这时算法收敛速度会
     加快，但是随机性不高，容易得到局部的相对最优
'''

# model1表示收费优先 model2表示时间优先，model3表示距离优先
n = int(input())
weight = [w[n][i] for i in range(0,3)]
(ALPHA, BETA, RHO, Q) = (1,5,2,100.0) #需要调整

#w1对应路径的长度，w2对应用户前往目的地时间，w3为停车位收费情况

#   w0*d+w1*r+w2*t+w3*c
# 城市数，蚁群 
(city_num, ant_num) = (50,50)
#城市距离和信息素 
pheromone_graph = [ [1.0 for col in range(park_num)] for raw in range(park_num)]
start_index = 0
end_index = 999
#停车场到目标物的距< 2km

#----------- 蚂蚁 -----------
class Ant(object):
 
    # 初始化
    def __init__(self,ID,start_index,end_index):
        
        self.ID = ID                 # ID
        self.__clean_data()          # 随机初始化出生点
        self.start_index = start_index
        self.end_index = end_index
 
    # 初始数据
    def __clean_data(self):
    
        self.path = []               # 当前蚂蚁的路径           
        self.total_distance = 0.0    # 当前路径的总距离
        self.move_count = 0          # 移动次数
        self.current_city = -1       # 当前停留的城市
        self.open_table_city = [True for i in range(park_num)] # 探索城市的状态
        
        city_index = 0 # 随机初始出生点 random.randint(0,city_num - 1) last version
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1
    

    # 选择下一个
    def __choice_next_city(self):
        next_city = -1
        select_citys_prob = [0.0 for i in range(park_num)]  #存储去下个城市的概率
        total_prob = 0.0
 
        # 获取去下一个的概率
        for i in rank_index:
            if self.open_table_city[i]:
                try :
                    # 计算概率：与信息素浓度成正比，与距离成反比（需修改）
                    #temp = (distance_graph_[self.current_city][i])
                    temp = (w[n][0] * distance_graph_[self.current_city][i]  + w[n][1] * user_to_park_time_[i] + w[n][2] * park_cost_[i])
                    #select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow((1.0/distance_graph[self.current_city][i]), BETA)
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(1/temp, BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print ('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID = self.ID, current = self.current_city, target = i))
                    sys.exit(1)
        
        #轮盘选择(小技巧)
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(park_num):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break
 
        # 未从概率产生，顺序选择一个未访问城市
        # if next_city == -1:
        #     for i in range(city_num):
        #         if self.open_table_city[i]:
        #             next_city = i
        #             break
 
        if (next_city == -1):
            next_city = random.randint(0, park_num - 1)
            while ((self.open_table_city[next_city]) == False):  # if==False,说明已经遍历过了
                next_city = random.randint(0, park_num - 1)
    
        # 返回下一个序号
        return next_city
    
    # 计算路径总距离
    def __cal_total_distance(self):
        
        temp_distance = 0.0
        for i in range(self.path.__len__()):
            start, end = self.path[i], self.path[i-1]
            temp_distance += distance_graph[start][end]
 
        # 回路
        end = self.path[0]
        temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance
        
    
    # 移动操作
    def __move(self, next_city):
        
        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1
        
    # 搜索路径
    def search_path(self):
 
        # 初始化数据
        self.__clean_data()
 
        # 搜素路径，遍历完所有城市为止
        while self.move_count < park_num:
            # 移动到下一个城市
            next_city =  self.__choice_next_city()
            self.__move(next_city)
            if(next_city == self.end_index): break #修改处
 
        # 计算路径总长度
        self.__cal_total_distance()
 
#----------- TSP问题 -----------
        
class TSP(object):
 
    def __init__(self, root, width = 3000, height = 3000, n = park_num):
 
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
        
        distance_graph[start_index][end_index] = 1000000
        distance_graph_[start_index][end_index] = 1000000
        user_to_park_time_[end_index] = 0
        park_cost_[end_index] = 0
    
 
    # 按键响应程序
    def __bindEvents(self):
 
        self.root.bind("q", self.quite)        # 退出程序
        self.root.bind("n", self.new)          # 初始化
        self.root.bind("e", self.search_path)  # 开始搜索
        self.root.bind("s", self.stop)         # 停止搜索
        self.canvas.bind('<ButtonPress-1>', lambda event: self.canvas.scan_mark(event.x, event.y))
        self.canvas.bind("<B1-Motion>", lambda event: self.canvas.scan_dragto(event.x, event.y, gain=1))
   
    def do_zoom(self,event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        factor = 1.001 ** event.delta
        self.canvas.scale(ALL, x, y, factor, factor)
        
 
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
        self.nodes = []  # 节点坐标
        self.nodes2 = [] # 节点对象
        self.nodes3 = [] # 出发点到目标点
        # 初始化城市节点
        for i in range(len(distance_x)):
            # 在画布上随机初始坐标
            x = distance_x[i]
            index = i
            y = distance_y[i]
            self.nodes.append((x, y))
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
                    text = '('+str(x)+','+str(y)+','+str(index)+')',    # 所绘制文字的内容
                    fill = 'black'                       # 所绘制文字的颜色为灰色
                )
        x_green = [distance_x[start_index],distance_x[end_index]]
        y_green = [distance_y[start_index],distance_y[end_index]]
        for i in range(len(x_green)):
            x = x_green[i]
            y = y_green[i]
            self.nodes3.append((x, y))
            node = self.canvas.create_oval(x - self.__r,
                    y - self.__r, x + self.__r, y + self.__r,
                    fill = "#00FF00",      # 填充绿色
                    outline = "#000000",   # 轮廓白色
                    tags = "node",
                )

        
            
        # 顺序连接城市
        #self.line(range(city_num))
        
        # 初始城市之间的距离和信息素
        for i in range(park_num):
            for j in range(park_num):
                pheromone_graph[i][j] = 1.0
        self.ants = [Ant(ID,start_index,end_index) for ID in range(ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1,start_index,end_index)                          # 初始最优解
        self.best_ant.total_distance = 1 << 31           # 初始最大距离
        self.iter = 1                                    # 初始化迭代次数 
            
    # 将节点按order顺序连线
    def line(self, order):
        # 删除原线
        self.canvas.delete("line")
        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            self.canvas.create_line(p1, p2, fill = "#FF00FF", tags = "line")
            return i2
        
        # order[-1]为初始值
        n = len(order)
        reduce(line2, order, order[n - 2])
 
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
            find_index = self.best_ant.path[1]
            print(u"迭代次数：",self.iter,u"最佳路径总距离：(m)",int(self.best_ant.total_distance-1000000),u"路径",self.best_ant.path)
            print(u"               到达目的地时间",user_to_park_time[find_index],u"停车场收费",park_cost[find_index])
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
        temp_pheromone = [[0.0 for col in range(park_num)] for raw in range(park_num)]
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

