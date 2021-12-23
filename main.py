import numpy as np
import gmap as gp
from center_dqn import Center_DQN
from uav import UAV_agent
from sensor import sensor_agent
import matplotlib.pyplot as plt

Ed = 10000                          # total slot
pl_step = 7                        # How many steps will The system plan the next destination
# jud = 70000

#---------------------------------------------------------------------------------------------------------------------------
# Create Regions in Map
#---------------------------------------------------------------------------------------------------------------------------

num1 = 5
num2 = 4
num_region = num1*num2          # there are now 20 regions in the map
region = gp.genmap(600, 400, num1, num2)
m = 8
v1 = m*np.sin(np.pi/4)   #equation 20, assumed to be constant
vlist = [[0, 0], [m, 0], [v1, v1], [0, m], [-v1, v1], [-m, 0], [-v1, -v1], [0, -m], [v1, -v1]] # movements/directions possible for each UAV
# vlist = [[0,0],[8,0],[5.6,5.6],[0,8],[-5.6,5.6],[-8,0],[-5.6,-5.6],[0,-8],[5.6,-5.6]]

#---------------------------------------------------------------------------------------------------------------------------
# UAVlist parameters
#---------------------------------------------------------------------------------------------------------------------------

num_UAV = 6                                        # number of UAVs
com_r = 60                                         # communication radius
region_obstacle = gp.gen_obs(num_region)           # generates locations for 20 obstacles
region_rate = np.zeros([num_region])
omeg = 1/num_UAV
slot = 0.5
t_bandwidth = 2e6
cal_L = 3000
k = 1e-26
f_max = 2e9                                         #the max CPU cycle frequency of UAV from eq 8
p_max = 5
OUT = np.zeros([num_UAV])
reward = np.zeros([num_UAV])
reset_p_T = 800
E_wait = np.ones([401, 601])                        # Storing the local observations ?

#---------------------------------------------------------------------------------------------------------------------------
# Sensor parameters
#---------------------------------------------------------------------------------------------------------------------------

num_sensor = 20000
averate = np.random.uniform(280, 300, [num_region])
p_sensor = gp.position_sensor(region, num_sensor)   # position of sensors
# sX = p_sensor['W']
# sY = p_sensor['H']
# plt.scatter(sX, sY)
# plt.show()
C = 2e3                            # Data Transmission rate
prebuf = np.zeros([num_UAV])
data = np.zeros([num_UAV])
cover = np.zeros([Ed])             # record data buffer to store average data collected in each timestamp

#---------------------------------------------------------------------------------------------------------------------------
# DQN Model parameters
#---------------------------------------------------------------------------------------------------------------------------

T = 100                            # How many steps will the epsilon be reset and the trained weights will be stored
ep0 = 0.97                         # initial exploration rate
batch_size = 12                    # training samples per batch
update_target = 10
losses = np.empty((num_UAV, 0), int)     # to store loss for each timestamp
Mentrd = np.zeros([num_UAV, Ed])  # stores the reward for each UAV at each timestamp

#---------------------------------------------------------------------------------------------------------------------------
# Gamma value parameters
#---------------------------------------------------------------------------------------------------------------------------

g0 = 1e-4
d0 = 1
the = 4
P_cen = np.array([300, 200])
gammalist = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

#---------------------------------------------------------------------------------------------------------------------------
# Generate UAV agent
#---------------------------------------------------------------------------------------------------------------------------

UAVlist = []                         # contains the list of UAV objects created
for uk in range(num_UAV):            # 6
    UAVlist.append(UAV_agent(uk, com_r, region_obstacle, region, omeg, slot, t_bandwidth, cal_L, k, f_max, p_max))

#---------------------------------------------------------------------------------------------------------------------------
# Generate sensor agent
#---------------------------------------------------------------------------------------------------------------------------

sensorlist=[]                       # contains the list of sensor objects created
for dl in range(num_sensor):        # 20,000
    sensorlist.append(sensor_agent([p_sensor['W'][dl], p_sensor['H'][dl]], C, region, averate, slot))

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

Center = Center_DQN((84, 84, 1), 9, num_UAV, batch_size)

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
plt.xlim((0, 600))
plt.ylim((0, 400))
plt.grid(True)
plt.ion()  # interactive mode on
X = np.zeros([num_UAV])
Y = np.zeros([num_UAV])
fg = 1

#---------------------------------------------------------------------------------------------------------------------------
# move first, get the data, offload collected data
#---------------------------------------------------------------------------------------------------------------------------

for t in range(Ed):                                     # for each epoc do
    gp.gen_datarate(averate, region_rate)               # for each region generate the data rate
    print("t is ", t, "\n")

    # ---------------------------------------------------------------------------------------------------------------------

    if t % T == 0 and t > 0:                            # After every T steps
        Center.epsilon = ep0                            # reset epsilon
        Center.save("./save/center-dqn.h5")             # save trained weights
        np.save("record_rd3", Mentrd)                   # save the rewards

    # --------------------------------------------------------------------------------------------------------------------

    if t % pl_step == 0:                                    # system plans the next action after every planning step
        pre_feature = []
        aft_feature = []
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
            pre_feature.append(UAVlist[uk].map_feature(region_rate, UAVlist, E_wait))    # record former feature of each drone 84x84x1
            act = Center.act(pre_feature[uk], fg)                                        # get the action V
            act_note.append(act)                                                         # record the taken action

    # ----------------------------------------------------------------------------------------------------------------------

    for uk in range(num_UAV):                                                           # execute the action a(t)
        OUT[uk]=UAVlist[uk].fresh_position(vlist[act_note[uk]], region_obstacle)         # Update the UAV position accroding to the path
        # out = (1 means moving out of the map), (0 means inside the map)
        UAVlist[uk].cal_hight()                                                            # calculate h
        X[uk] = UAVlist[uk].position[0]                                                    # calculate x
        Y[uk] = UAVlist[uk].position[1]                                                    # calculate y
        UAVlist[uk].fresh_buf()                                                            # do edge processing and add the data to the queue
        prebuf[uk] = UAVlist[uk].data_buf                                                  # Update the data left in buffer

    # ----------------------------------------------------------------------------------------------------------------------

    gp.list_gama(g0, d0, the, UAVlist, P_cen)                                               # update discount rate for each UAV

    # ----------------------------------------------------------------------------------------------------------------------

    for dl in range(num_sensor):                                                # for each sensor do
        sensorlist[dl].data_rate = region_rate[sensorlist[dl].rNo]              # find data rate at region r
        sensorlist[dl].fresh_buf(UAVlist)                          # calculate data present in the buffer of that sensor
        cover[t] = cover[t]+sensorlist[dl].wait                                 # collect the data from covered sensors

    cover[t] = cover[t]/num_sensor                                              # average data generated at time t
    # print(cover[t])

    # ----------------------------------------------------------------------------------------------------------------------

    for uk in range(num_UAV):
        reward[uk] = reward[uk]+UAVlist[uk].data_buf-prebuf[uk]   # reward is 1D array that stores reward for each drone
        Mentrd[uk, t]=reward[uk]                                     # at time t what is reward set generated

    # ----------------------------------------------------------------------------------------------------------------------

#    if sum(OUT)>=num_UAV/2:                                           # if more number of UAV have gone out of the map
#        fg=0
#    if np.random.rand()>0.82 and fg==0:
#        fg=1
    # ----------------------------------------------------------------------------------------------------------------------

    if t%pl_step==0:
        E_wait = gp.W_wait(600,400,sensorlist)                                         # generate observatiosn after every pl_step

        for uk in range(num_UAV):                                                      # For each UAV
            aft_feature.append(UAVlist[uk].map_feature(region_rate,UAVlist,E_wait))    #    Obtain the observations Ok(tp+1)
            rd=reward[uk]/1000                                                         #    Obtain the reward r(tp)
            reward[uk]=0                                                               #    make the reward array 0 so that it can be used again in next iteration
            UAVlist[uk].reward = rd                                                    #    update the reward for the UAV uk

            # Transmit e(tp) = ((ok(tp), a(tp), r(tp), ok(tp+1)) to the central relay memory
            if t>0:
                Center.remember(pre_feature[uk], act_note[uk], rd, aft_feature[uk], uk)     # record the training data

    # ----------------------------------------------------------------------------------------------------------------------

#    if t>1000:
#        Center.epsilon=ep0
#        Center.epsilon_decay=1
    # ----------------------------------------------------------------------------------------------------------------------

    if t>batch_size*pl_step and t%pl_step==0:
       larr = []
       for turn in range(num_UAV):
#            Center.replay(batch_size,turn,t%reset_p_T)
             larr = np.append( larr, Center.replay(batch_size,turn,t-batch_size*pl_step))
       #print("larr",larr)
       losses = np.append(losses, np.array([larr]).transpose(), axis=1)
       #larr = np.delete(larr, axis=0)

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
    if t>0:
      plt.pause(0.1)

# ----------------------------------------------------------------------------------------------------------------------
# Save the reward and data buffer
# ----------------------------------------------------------------------------------------------------------------------

np.save("record_rd3",Mentrd)
np.save("cover_hungry_10",cover)

# ----------------------------------------------------------------------------------------------------------------------
# plotting the data buffer values
# ----------------------------------------------------------------------------------------------------------------------

fig=plt.figure()
#print(cover)
plt.xlabel('Timestamp(t)')
plt.ylabel('buffer data')
plt.title('Data Vs. Time')
plt.grid(True)

plt.plot( y,cover, label= "stars", color= "blue",linestyle='--',marker="." )
#for a,b in zip(y ,cover):
#    plt.text(a, b, str(b))
#plt.plot(cover)

plt.savefig('Buffer_data.png')
plt.close(fig)


# plotting the average loss value

#calculate the average loss
avg_loss = []
for i in range(len(losses[0])):
    summ = 0
    for j in range(len(losses)):
        summ += losses[j][i]
    avg_loss.append(summ / len(losses))

fal = plt.figure()

x = []
for i in range(1, len(avg_loss) + 1):
    x.append(i)
print(x)

y = avg_loss
plt.plot(x, y, label="loss_values ")
# naming the x axis
plt.xlabel('iterations')
# naming the y axis
plt.ylabel('loss_value')
# giving a title to my graph
plt.title('average loss values')
plt.grid(True)
# show a legend on the plot
plt.legend()

# function to show the plot
plt.savefig('Average_Loss_Values.png')
plt.close(fal)



#print("losses: ", losses)

# plotting the losses for each UAVs

fi = plt.figure()

x = []
for i in range(1, len(losses[1]) + 1):
    x.append(i)
print(x)

it = 0
for i in losses:
    it = it + 1
    y = i
    plt.plot(x, y, label="UAV " + str(it), marker="." )
# naming the x axis
plt.xlabel('iterations')
# naming the y axis
plt.ylabel('loss_value')
# giving a title to my graph
plt.title('loss values for each UAVs')
plt.grid(True)
# show a legend on the plot
plt.legend()

# function to show the plot
plt.savefig('Loss_Values.png')
plt.close(fi)


#calculate the average reward
avg_reward = []
for i in range(len(Mentrd[0])):
    summ = 0
    for j in range(len(Mentrd)):
        summ += Mentrd[j][i]
    avg_reward.append(summ / len(Mentrd))

far = plt.figure()

x = []
for i in range(1, len(avg_reward) + 1):
    x.append(i)
print(x)

y = avg_reward
plt.plot(x, y, label="reward_values ")
# naming the x axis
plt.xlabel('iterations')
# naming the y axis
plt.ylabel('reward_value')
# giving a title to my graph
plt.title('average reward values')
plt.grid(True)
# show a legend on the plot
plt.legend()

# function to show the plot
plt.savefig('Average_Reward_Values.png')
plt.close(far)



# plotting the rewards

fr = plt.figure()

x = []
for i in range(1, len(Mentrd[1]) + 1):
    x.append(i)
print(x)

it = 0
for i in Mentrd:
    it = it + 1
    y = i
    plt.plot(x, y, label="UAV " + str(it), marker="." )
# naming the x axis
plt.xlabel('iterations')
# naming the y axis
plt.ylabel('Reward_value')
# giving a title to my graph
plt.title('Reward values for each UAVs')
plt.grid(True)
# show a legend on the plot
plt.legend()

# function to show the plot
plt.savefig('Reward_Values.png')
plt.close(fr)



