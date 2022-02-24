import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# plotting the Average Service Urgency
# ----------------------------------------------------------------------------------------------------------------------

def Avg_Service_Urgency(cover,y):

    fig=plt.figure()
    plt.xlabel('Timestamp(t)')
    plt.ylabel('Average Service Urgency')
    plt.title('Average Service Urgency Vs. Time')
    plt.grid(True)
    plt.plot( y,cover, label= "stars", color= "blue",marker=".") #linestyle='--'
    plt.savefig('Average Service Urgency for equal alpha.png')
    plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# plotting the average loss value
# ----------------------------------------------------------------------------------------------------------------------

def Avg_Loss(losses):  

    #calculate the average loss
    avg_loss = []
    for i in range(len(losses[0])):
        summ = 0
        for j in range(len(losses)):
            summ += losses[j][i]
        avg_loss.append(summ / len(losses))

    x = []
    for i in range(1, len(avg_loss) + 1):
        x.append(i)
  
    y = avg_loss

    fal = plt.figure()
    plt.plot(x, y, label="loss_values ")
    plt.xlabel('iterations')
    plt.ylabel('loss_value')
    plt.title('average loss values')
    plt.grid(True)
    plt.legend()
    plt.savefig('Average_Loss_Values.png')
    plt.close(fal)

   
# ----------------------------------------------------------------------------------------------------------------------
# plotting the average loss value
# ----------------------------------------------------------------------------------------------------------------------

def UAV_Loss(losses):
    fi = plt.figure()
    x = []
    for i in range(1, len(losses[1]) + 1):
        x.append(i)
    it = 0
    for i in losses:
        it = it + 1
        y = i
        plt.plot(x, y, label="UAV " + str(it), marker="." )
   
    plt.xlabel('iterations')
    plt.ylabel('loss_value')
    plt.title('loss values for each UAVs')
    plt.grid(True)
    plt.legend()
    plt.savefig('Loss_Values.png')
    plt.close(fi)


# ----------------------------------------------------------------------------------------------------------------------
# plotting the average reward values
# ----------------------------------------------------------------------------------------------------------------------
def Avg_Reward_Values(Mentrd):
    avg_reward = []
    for i in range(len(Mentrd[0])):
        summ = 0
        for j in range(len(Mentrd)):
            summ += Mentrd[j][i]
        avg_reward.append(summ / len(Mentrd))

    x = []
    for i in range(1, len(avg_reward) + 1):
        x.append(i)
    y = avg_reward
    
    far = plt.figure()
    plt.plot(x, y, label="reward_values ")
    plt.xlabel('iterations')
    plt.ylabel('reward_value')
    plt.title('average reward values')
    plt.grid(True)
    plt.legend()
    plt.savefig('Average_Reward_Values.png')
    plt.close(far)

#---------------------------------------------------------------------------------------------------------------------------
# Plotting the rewards
#---------------------------------------------------------------------------------------------------------------------------

def UAV_Rewards(Mentrd):
    fr = plt.figure()

    x = []
    for i in range(1, len(Mentrd[1]) + 1):
        x.append(i)

    it = 0
    for i in Mentrd:
        it = it + 1
        y = i
        plt.plot(x, y, label="UAV " + str(it), marker="." )

    plt.xlabel('iterations')
    plt.ylabel('Reward_value')
    plt.title('Reward values for each UAVs')
    plt.grid(True)
    plt.legend()
    plt.savefig('Reward_Values.png')
    plt.close(fr)

#---------------------------------------------------------------------------------------------------------------------------
# Plotting the sensors
#---------------------------------------------------------------------------------------------------------------------------

def Sensor_Graph(sX, sY):

    x = sX
    y = sY
    fs = plt.figure()
    plt.scatter(x, y, marker="*", label ="sensors" )
    plt.xlim((0, 300))
    plt.ylim((0, 600))
    plt.grid(True)
    plt.xlabel('iterations')
    plt.ylabel('Sensors')
    plt.title('Sensor locations')
    plt.grid(True)
    plt.legend()
    plt.savefig('Sensors.png')
    plt.close(fs)