
import numpy as np
import matplotlib.pyplot as plt
import time
import gmap as gp
import graph as g
from ssl import get_default_verify_paths
from center_dqn import Center_DQN
from uav import UAV_agent
from sensor import sensor_agent



Ed = 100                        # Total iterations
pl_step = 5                    # How many steps will the system plan the next destination

#---------------------------------------------------------------------------------------------------------------------------
# Create Regions in Map
#---------------------------------------------------------------------------------------------------------------------------

# st = time.time()
num1 = 5
num2 = 4
num_region = num1*num2             # there are now 20 regions in the map
region = gp.genmap(600, 400, num1, num2)  # let it be something
# print("genmap took", time.time() - st, "to run \n")

#---------------------------------------------------------------------------------------------------------------------------
# Actions
#---------------------------------------------------------------------------------------------------------------------------

m = 8
v1 = m*np.sin(np.pi/4)   #equation 20, assumed to be constant
vlist = [[0, 0], [m, 0], [v1, v1], [0, m], [-v1, v1], [-m, 0], [-v1, -v1], [0, -m], [v1, -v1]] # movements/directions possible for each UAV
# vlist = [[0,0],[8,0],[5.6,5.6],[0,8],[-5.6,5.6],[-8,0],[-5.6,-5.6],[0,-8],[5.6,-5.6]]

#---------------------------------------------------------------------------------------------------------------------------
# UAVlist parameters
#---------------------------------------------------------------------------------------------------------------------------

num_UAV = 6                                        # number of UAVs
com_r = 60                                         # communication radius
# st = time.time()
region_obstacle = gp.gen_obs(num_region)           # generates locations for 20 obstacles
region_rate = np.zeros([num_region])               # array of 20 values each initilized to 0
# print("gen_obs took", time.time() - st, "to run \n")
omeg = 1/num_UAV                                   # omeg = 1/6 = 0.167 weight in eq 22.
slot = 0.5                                         # system time slot(τ)
t_bandwidth = 2e6
OUT    = np.zeros([num_UAV])
reward = np.zeros([num_UAV]) 
E_wait = np.ones([401, 601])                       # Storing the local observations
UAVlist = []                                       # contains the list of UAV objects created
X = np.zeros([num_UAV])
Y = np.zeros([num_UAV])

#---------------------------------------------------------------------------------------------------------------------------
# Edge data processing parameters
#---------------------------------------------------------------------------------------------------------------------------

cal_L = 3000                                       # L_k = 3000 bits per cycle
k = 1e-26
f_max = 2e9                                        # Max CPU cycle frequency of UAV from eq 8
p_max = 5

#---------------------------------------------------------------------------------------------------------------------------
# Sensor parameters
#---------------------------------------------------------------------------------------------------------------------------

num_sensor = 20000                                  # reduce this value 50
averate = np.random.uniform(250, 300, [num_region])
# st = time.time()
p_sensor = gp.position_sensor(region, num_sensor)   # position of sensors
# print("position_sensor took", time.time() - st, "to run \n")
C = 2e3                                             # Data Transmission rate of sensors
sensorlist=[]                       # contains the list of sensor objects created
sX =np.zeros([num_sensor])
sY =np.zeros([num_sensor])

#---------------------------------------------------------------------------------------------------------------------------
# DQN Model parameters
#---------------------------------------------------------------------------------------------------------------------------

fg = 1
T = 100                            # How many steps will the epsilon be reset and the trained weights will be stored
G = 800                            # after how many intervals we need to update the target network
ep0 = 1                         # initial exploration rate
batch_size = 12                    # training samples per batch
update_target = 10
losses = np.empty((num_UAV, 0), int)     # to store loss for each timestamp
Mentrd = np.zeros([num_UAV, Ed])  # stores the reward for each UAV at each timestamp
prebuf = np.zeros([num_UAV])
data = np.zeros([num_UAV])
cover = np.zeros([Ed])             # record data buffer to store average data collected in each timestamp

#---------------------------------------------------------------------------------------------------------------------------
# Gamma value parameters
#---------------------------------------------------------------------------------------------------------------------------

g0 = 1e-4
d0 = 1
the = 4
P_cen = np.array([300, 200])
gammalist = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

fp = open("output_file.txt","w+")

#---------------------------------------------------------------------------------------------------------------------------
# Generate UAV agent
#---------------------------------------------------------------------------------------------------------------------------
# st = time.time()

for uk in range(num_UAV):            # 6
    UAVlist.append(UAV_agent(uk, com_r, region_obstacle, region, omeg, slot, t_bandwidth, cal_L, k, f_max, p_max))
                             #6 , 60, 20 {}, 20 {}, 1/6, 0.5, 2e6, 3000, 1e-26, 2e9, 5
# print("UAV initialization took", time.time() - st, "to run \n")
#---------------------------------------------------------------------------------------------------------------------------
# Generate sensor agent
#---------------------------------------------------------------------------------------------------------------------------
# st = time.time()

for dl in range(num_sensor):        # 20,000
    sensorlist.append(sensor_agent([p_sensor['W'][dl], p_sensor['H'][dl]], C, region, averate, slot))
    sX[dl] = sensorlist[dl].position[0]
    sY[dl] = sensorlist[dl].position[1]
    

# print("Sensor initialization took", time.time() - st, "to run \n")
# C = communication rate = 2000
# averate = local data rate = array having values between 280 to 300
# system time slot interval =0.5

#---------------------------------------------------------------------------------------------------------------------------
# Initialize a DQN
#---------------------------------------------------------------------------------------------------------------------------

# input size = 84 *84*1
# number of actions = 9
# num_UAV = 6
# batch_size = 12
# gamma = 0.8
# alpha =0.1
# rj =0
# st = time.time()
Center = Center_DQN((84, 84, 1), 9, num_UAV, batch_size)
# print("DQN initialization took", time.time() - st, "to run \n")
# Center.load("./save/center-dqn.h5")

#---------------------------------------------------------------------------------------------------------------------------
# Initialize Plot
#---------------------------------------------------------------------------------------------------------------------------

y = []
for t in range(0, Ed):            # To store all timestamps for plotting
    y.append(t)
plt.close()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xlim((0, 300))
plt.ylim((0, 400))
plt.grid(True)
plt.ion()  # interactive mode on


#---------------------------------------------------------------------------------------------------------------------------
# move first, get the data, offload collected data
#---------------------------------------------------------------------------------------------------------------------------

for t in range(Ed):                                     # for each epoc do 10000
    fp.write("t is ")
    fp.write(repr(t))
    fp.write("\n")
    print("t is ", t, "\n")
    # st = time.time()
    gp.gen_datarate(averate, region_rate)               # for each region generate the data rate
    # print("gen_datarate took", time.time() - st, "to run \n")
    # ---------------------------------------------------------------------------------------------------------------------

    if t % T == 0 and t > 0:                            # After every T steps
        Center.epsilon = ep0                            # reset epsilon
        Center.save("./save/center-dqn.h5")             # save trained weights
        np.save("./save/record_rd3", Mentrd)            # save the rewards

    # --------------------------------------------------------------------------------------------------------------------

    if t % pl_step == 0:                                # system plans the next action after every planning step
        pre_feature = []                                # ok(tp)
        aft_feature = []                                # ok(tp+1)
        act_note = []

        # for each uk do
        #    collect around service requirements and generate observations Ok(tp)
        #    randomly generate epsilon(tp)
        #    choose action a(tp) by
        #    if p< epsilon(tp) then
        #       randomly select an action a(tp)
        #    else
        #       a(tp) = argmax_a Q(okt(p),a,theta)
        #    end if
        
        for uk in range(num_UAV):                                                        # for each UAV
            # st = time.time()
            pre_feature.append(UAVlist[uk].map_feature(region_rate, UAVlist, E_wait))    # record former feature of each drone 84x84x1
            # Ok tp = { o1 tp , O2 tp, .... Ok tp}
            
            act = Center.act(pre_feature[uk], fg)                                        # get the action V
            act_note.append(act)                                                         # record the taken action
            # print("calculating action took", time.time() - st, "to run \n")

        # print("pre_feature /n", pre_feature, "/n")
        # print("act values:", act_note, "\n")
    # ----------------------------------------------------------------------------------------------------------------------
    # st = time.time()
    for uk in range(num_UAV):                                                           # execute the action a(t)
        OUT[uk]=UAVlist[uk].fresh_position(vlist[act_note[uk]], region_obstacle)         # Update the UAV position accroding to the path
        # out = (1 means moving out of the map), (0 means inside the map)
        UAVlist[uk].cal_hight()                                                            # calculate h
        X[uk] = UAVlist[uk].position[0]                                                    # calculate x
        Y[uk] = UAVlist[uk].position[1]                                                    # calculate y
        UAVlist[uk].fresh_buf()                                                            # do edge processing and add the data to the queue
        prebuf[uk] = UAVlist[uk].data_buf                                                  # Update the data left in buffer of UAV
    # print("fresh position and fresh buf took", time.time() - st, "to run \n")
    

    # print("out values:", OUT, "\n")
    # print("prebuf values:", prebuf, "\n")
    # ----------------------------------------------------------------------------------------------------------------------
    # st = time.time()
    gp.list_gama(g0, d0, the, UAVlist, P_cen)                                               # update discount rate for each UAV
    # print("list_gama took", time.time() - st, "to run \n")
    # ----------------------------------------------------------------------------------------------------------------------
    # st = time.time()
    for dl in range(num_sensor):                                                # for each sensor do
        sensorlist[dl].data_rate = region_rate[sensorlist[dl].rNo]              # find data rate at region r
        sensorlist[dl].fresh_buf(UAVlist)                          # calculate data present in the buffer of that sensor
        cover[t] = cover[t]+sensorlist[dl].wait                                 # collect the data from covered sensors

    cover[t] = cover[t]/num_sensor                                              # average service urgency at time t
    # print("cover values" , cover[t], "/n")
    # print("calculating cover took", time.time() - st, "to run \n")
    # ----------------------------------------------------------------------------------------------------------------------

    for uk in range(num_UAV):
        reward[uk] = reward[uk]+UAVlist[uk].data_buf-prebuf[uk]   # reward is 1D array that stores reward for each drone
        Mentrd[uk, t]=reward[uk]                                     # at time t what is reward set generated
    
    # print("reward values" , reward, "/n")
    # ----------------------------------------------------------------------------------------------------------------------

#    if sum(OUT)>=num_UAV/2:                                           # if more number of UAV have gone out of the map
#        fg=0
#    if np.random.rand()>0.82 and fg==0:
#        fg=1
    # ----------------------------------------------------------------------------------------------------------------------
    # st = time.time()
    if t%pl_step==0:
        E_wait = gp.W_wait(600,400,sensorlist)                                         # generate observatiosn after every pl_step

        for uk in range(num_UAV):                                                      # For each UAV
            aft_feature.append(UAVlist[uk].map_feature(region_rate,UAVlist,E_wait))    #    Obtain the observations Ok(tp+1)
            # Ok ( tp+1)
            rd=reward[uk]/1000                                                         #    Obtain the reward r(tp)
            reward[uk]=0                                                               #    make the reward array 0 so that it can be used again in next iteration
            UAVlist[uk].reward = rd                                                    #    update the reward for the UAV uk

            # Transmit e(tp) = ((ok(tp), a(tp), r(tp), ok(tp+1)) to the central relay memory
            if t>0:
                Center.remember(pre_feature[uk], act_note[uk], rd, aft_feature[uk], uk)     # record the training data
     
    # print("calculating remember took", time.time() - st, "to run \n")
    # ----------------------------------------------------------------------------------------------------------------------

#    if t>1000:
#        Center.epsilon=ep0
#        Center.epsilon_decay=1
    # ----------------------------------------------------------------------------------------------------------------------
    # st = time.time()
    if t>batch_size*pl_step and t%pl_step==0:
       larr = []
       for turn in range(num_UAV):
#            Center.replay(batch_size,turn,t%G)
             larr = np.append( larr, Center.replay(batch_size,turn,t-batch_size*pl_step))
       #print("larr",larr)
       losses = np.append(losses, np.array([larr]).transpose(), axis=1)
       #larr = np.delete(larr, axis=0)
    # print("calculating loss took", time.time() - st, "to run \n")
    # ----------------------------------------------------------------------------------------------------------------------
    if t%update_target==0:                    # Update the reference network weights after every G steps
        Center.update_target_network()

    # ----------------------------------------------------------------------------------------------------------------------
    # Plot the UAVs in the map
    # ----------------------------------------------------------------------------------------------------------------------

    if t>0:
      ax.clear()
    plt.xlim((0,600))
    plt.ylim((0,400))
    plt.grid(True)
    colors = np.array(
        ["red", "green", "blue", "pink",   "purple", "magenta"])
    ax.scatter(X,Y,c= colors,marker='H',label= "UAVs")
   # ax.scatter(sX,sY, marker='*')
    if t>0:
      plt.pause(0.1)

fp.close()
# ----------------------------------------------------------------------------------------------------------------------
# Save the reward and data buffer
# ----------------------------------------------------------------------------------------------------------------------

np.save("./save/record_rd3",Mentrd)
np.save("./save/cover_hungry_10",cover)
fp = open("output_file.txt","w+")
fp.write("testing process \n \n ")
#----------------------------------------
#  Print the graphs
#-----------------------------------------

g.Avg_Service_Urgency(cover,y)
g.Avg_Loss(losses)
g.UAV_Loss(losses)
g.Avg_Reward_Values(Mentrd)
g.UAV_Rewards(Mentrd)
g.Sensor_Graph(sX, sY)

# ----------------------------------------------------------------------------------------------------------------------
# Validating the DQN Model
# ----------------------------------------------------------------------------------------------------------------------

Center.load("./save/center-dqn.h5")

for t in range(Ed):
    fp.write("t is ")
    fp.write(repr(t))
    fp.write("\n")
    print("t is ", t, "\n")
    # st = time.time()
    gp.gen_datarate(averate, region_rate)               # for each region generate the data rate

    if t % T == 0 and t > 0:                            # After every T steps
        Center.epsilon = ep0                            # reset epsilon
        #Center.save("./save/center-dqn.h5")             # save trained weights
        #np.save("./save/record_rd3", Mentrd)            # save the rewards
    
    if t % pl_step == 0:                                # system plans the next action after every planning step
        pre_feature = []                                # ok(tp)
        aft_feature = []                                # ok(tp+1)
        act_note = []

        # for each uk do
        #    collect around service requirements and generate observations Ok(tp)
        #    randomly generate epsilon(tp)
        #    choose action a(tp) by
        #    if p< epsilon(tp) then
        #       randomly select an action a(tp)
        #    else
        #       a(tp) = argmax_a Q(okt(p),a,theta)
        #    end if
        
        for uk in range(num_UAV):                                                        # for each UAV
            # st = time.time()
            pre_feature.append(UAVlist[uk].map_feature(region_rate, UAVlist, E_wait))    # record former feature of each drone 84x84x1
            # Ok tp = { o1 tp , O2 tp, .... Ok tp}
            
            act = Center.actv(pre_feature[uk], fg)                                        # get the action V
            act_note.append(act)

    for uk in range(num_UAV):                                                           # execute the action a(t)
        OUT[uk]=UAVlist[uk].fresh_position(vlist[act_note[uk]], region_obstacle)         # Update the UAV position accroding to the path
        # out = (1 means moving out of the map), (0 means inside the map)
        UAVlist[uk].cal_hight()                                                            # calculate h
        X[uk] = UAVlist[uk].position[0]                                                    # calculate x
        Y[uk] = UAVlist[uk].position[1]                                                    # calculate y
        UAVlist[uk].fresh_buf()                                                            # do edge processing and add the data to the queue
        prebuf[uk] = UAVlist[uk].data_buf 

    gp.list_gama(g0, d0, the, UAVlist, P_cen)                                               # update discount rate for each UAV
    
    for dl in range(num_sensor):                                                # for each sensor do
        sensorlist[dl].data_rate = region_rate[sensorlist[dl].rNo]              # find data rate at region r
        sensorlist[dl].fresh_buf(UAVlist)                          # calculate data present in the buffer of that sensor
        cover[t] = cover[t]+sensorlist[dl].wait                                 # collect the data from covered sensors

    cover[t] = cover[t]/num_sensor 

    for uk in range(num_UAV):
        reward[uk] = reward[uk]+UAVlist[uk].data_buf-prebuf[uk]   # reward is 1D array that stores reward for each drone
        Mentrd[uk, t]=reward[uk]

    if t>batch_size*pl_step and t%pl_step==0:
       larr = []
       for turn in range(num_UAV):
#            Center.replay(batch_size,turn,t%G)
             larr = np.append( larr, Center.vreplay(batch_size,turn,t-batch_size*pl_step))
       #print("larr",larr)
       losses = np.append(losses, np.array([larr]).transpose(), axis=1)

    if t>0:
      ax.clear()
      plt.xlim((0,600))
      plt.ylim((0,400))
      plt.grid(True)
      colors = np.array(
        ["red", "green", "blue", "pink",   "purple", "magenta"])
      ax.scatter(X,Y,c= colors,marker='H',label= "UAVs")
      # ax.scatter(sX,sY, marker='*')
      if t>0:
        plt.pause(0.1)

fp.close()

#----------------------------------------
#  Print the graphs
#-----------------------------------------

g.Avg_Service_Urgencyt(cover,y)
g.Avg_Reward_Valuest(Mentrd)
g.Avg_Losst(losses)
g.UAV_Losst(losses)
g.UAV_Rewardst(Mentrd)

    
