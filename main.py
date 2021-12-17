# -*- coding: utf-8 -*-


import numpy as np
import gmap as gp
from center_dqn import Center_DQN
from uav import UAV_agent
from sensor import sensor_agent
import matplotlib.pyplot as plt

Ed = 10000                             # total slot
ep0 = 0.97
batch_size = 12                       # training samples per batch
pl_step = 5                           # How many steps will The system plan the next destination
T = 300                               # How many steps will the epsilon be reset and the trained weights will be stored
E_wait = np.ones([401, 601])
P_cen = np.array([300, 200])
N0 = 2e-20
alfmin = 1e-3
v = 8
V = 10e9
v1 = v*np.sin(np.pi/4)
# ------------------------------------------------------------------------------------------------------------------------------

# Create regions in map
num1 = 5
num2 = 4
num_region = num1*num2
region = gp.genmap(600, 400, num1, num2)  # returns a num1*num2 number of tuples {bl,br,bd,bu}
# ------------------------------------------------------------------------------------------------------------------------------

# UAVlist Parameters
num_UAV = 6
com_r = 60
region_obstacle = gp.gen_obs(num_region)
region_rate = np.zeros([num_region])
omeg = 1/num_UAV
slot = 0.5
t_bandwidth = 2e6
cal_L = 3000
k = 1e-26
f_max = 2e9    # the max cal frequency of UAV
p_max = 5
# ------------------------------------------------------------------------------------------------------------------------------

# Sensor parameters
num_sensor = 20000
C = 2e3
averate = np.random.uniform(280, 300, [num_region])
p_sensor = gp.position_sensor(region, num_sensor)
# ------------------------------------------------------------------------------------------------------------------------------

vlist = [[0, 0], [v, 0], [v1, v1], [0, v], [-v1, v1], [-v, 0], [-v1, -v1], [0, -v], [v1, -v1]]
g0 = 1e-4
d0 = 1
the = 4
OUT = np.zeros([num_UAV])
reward = np.zeros([num_UAV])
reset_p_T = 800

# jud=70000
gammalist = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
Mentrd =np.zeros([num_UAV, Ed])

# generate UAV agent
UAVlist = []
for i in range(num_UAV): # 6
    UAVlist.append(UAV_agent(i, com_r, region_obstacle, region, omeg, slot, t_bandwidth, cal_L, k, f_max, p_max))

# generate sensor agent
sensorlist = []
for i in range(num_sensor): # 20,000
    sensorlist.append(sensor_agent([p_sensor['W'][i], p_sensor['H'][i]], C, region, averate, slot))

# initailize a DQN
Center = Center_DQN((84, 84, 1), 9, num_UAV, batch_size)
#reference network weights dont know where it is initialized
#alpha = 0.1
#gamma = 0.8
#rj =0
#what is rowj and alpha max

# Center.load("./save/center-dqn.h5")
prebuf = np.zeros([num_UAV])
data = np.zeros([num_UAV])
# pre_data = np.zeros([num_UAV])

# define record data buf
cover = np.zeros([Ed])

# init plt
plt.close()  # clf() # 清图  cla() # 清坐标轴 close() # 关窗口
fig=plt.figure()
ax=fig.add_subplot(1, 1, 1)
plt.xlim((0, 600))
plt.ylim((0, 400))
plt.grid(True) #添加网格
plt.ion()  # interactive mode on
X = np.zeros([num_UAV])
Y = np.zeros([num_UAV])
fg = 1

# for each epoc do
for t in range(Ed):  # move first, get the data, offload collected data
    gp.gen_datarate(averate, region_rate)
#    print(t)
#    epsilon be reset and the trained weights will be stored after every T steps
    if t % T == 0 and t > 0:
        Center.epsilon = ep0
        Center.save("./save/center-dqn.h5")

# The system plan the next destination for each UAV after every pl_step

    if t % pl_step == 0:
        pre_feature = []
        aft_feature = []
        act_note = []

        # for each uk do
        #    collect around service requirements and generate observations Ok(tp)
        #    randomnly generate epsilon(tp)
        #    choose action a(tp) by
        #    if p< (epsilon(tp) then
        #       randomnly select an action a(tp)
        #    else
        #       a(tp) = argmax_a Q(okt(p),a,theta)
        #    end if

        for uk in range(num_UAV):
            pre_feature.append(UAVlist[uk].map_feature(region_rate, UAVlist, E_wait))    # record former feature
            act=Center.act(pre_feature[uk],fg)          # get the action V
            act_note.append(act)                       # record the taken action
    
    for uk in range(num_UAV):
        # execute the action a(t)
        OUT[uk]=UAVlist[uk].fresh_position(vlist[act_note[uk]],region_obstacle)

        #calculate(x,y,h)
        UAVlist[uk].cal_hight() # calculate optimal UAV height
        X[uk]=UAVlist[uk].position[0] #updating the X coordinate
        Y[uk]=UAVlist[uk].position[1] # updating the y coordinate

        # do edge processing and add the data to the queue
        UAVlist[uk].fresh_buf()
        prebuf[uk]=UAVlist[uk].data_buf   #the buf after fresh by server

    # update the gamma values for each drone
    gp.list_gama(g0,d0,the,UAVlist,P_cen)

    #collect data from covered sensors
    for dl in range(num_sensor):          #fresh buf send data to UAV
        sensorlist[dl].data_rate=region_rate[sensorlist[dl].rNo]
        sensorlist[dl].fresh_buf(UAVlist) #accumulate data in the former slot, transmit to UAV
        cover[t]=cover[t]+sensorlist[dl].wait
    cover[t]=cover[t]/num_sensor
    print(cover[t])

    # storing reward for each timestamp for drone uk
    for uk in range(num_UAV):
        reward[uk]=reward[uk]+UAVlist[uk].data_buf-prebuf[uk]
        Mentrd[uk,t]=reward[uk]
#    if sum(OUT)>=num_UAV/2:
#        fg=0
#    if np.random.rand()>0.82 and fg==0:
#        fg=1

# after every pl_step store the reward generated for each of the drones
    if t%pl_step==0:
        #generate observations
        E_wait=gp.W_wait(600,400,sensorlist)
        rdw=sum(sum(E_wait))
        #print(t)
        for i in range(num_UAV):  # calculate the reward : need the modify
            aft_feature.append(UAVlist[i].map_feature(region_rate,UAVlist,E_wait))    #recode the current feature
            rd = reward[i] / 1000
            reward[i] = 0
            UAVlist[i].reward=reward[i]
            reward[i]=data[i]/(prebuf[i]+1)
            if OUT[i]>0:
               rd=-200000

            if data[i]<700:
                reward=-1
            prebuf[i]= data[i]
            UAVlist[i].reward = rd
            # l_queue[t]=l_queue[t]+UAVlist[i].data_buf
            print("%f, %f, %f, %f"%(rd,UAVlist[i].data_buf,UAVlist[i].D_l,UAVlist[i].D_tr))
            #if UAVlist[i].data_buf>jud:
            #    reward=reward/(reward-jud)
            if t>0:
                Center.remember(pre_feature[i],act_note[i],rd,aft_feature[i],i)  #record the training data

        if t>1000:
            Center.epsilon=ep0
            Center.epsilon_decay=1

        if t>batch_size*pl_step and t%pl_step==0:
            for turn in range(num_UAV):
                Center.replay(batch_size,turn,t%reset_p_T)
                Center.replay(batch_size,turn,t-batch_size*pl_step)


    if t>0:
        ax.clear()
    plt.xlim((0,600))
    plt.ylim((0,400))
    plt.grid(True) #添加网格

    ax.scatter(X,Y,c='b',marker='.')  #散点图
#    if t>0:
    plt.pause(0.1)

    
#np.save("record_rd3",Mentrd)
np.save("cover_hungry_10",cover)
fig=plt.figure()
plt.plot(cover)


plt.show()

