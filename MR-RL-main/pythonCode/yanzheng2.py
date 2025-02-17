import numpy as np
import matplotlib.pyplot as plt



# x = np.linspace(0, 1000, 100)
# y = 1000*1 / (1 + np.exp(-x/50))
#
# plt.figure(figsize=(10, 6))
# plt.plot(x, y)
# plt.title('sigmoid(x_dist / 2500) * 8000')
# plt.xlabel('x_dist')
# plt.ylabel('sigmoid(x_dist / 2500) * 8000')
# plt.grid(True)
# plt.show()
# a=np.load('rewards.npy')
# np.savetxt('array.txt',a)

# import pandas as pd
#
# # 假设你的数据存储在一个名为"data.txt"的文件中
# data = pd.read_csv("scale1.txt", delim_whitespace=True, header=None)
#
# # 将数据保存到Excel文件中
# data.to_excel("output3.xlsx", header=False, index=False)



def getdata():
    global first_run
    global cluster
    global x_pre
    global y_pre
    global z_pre
    global eulerX_pre,eulerY_pre
    global x, y, z, eulerX,eulerY
    global addx, addy, addz
    global position, rotation
    global step
    global timing
    global total_distance_hand
    global total_distance
    global effiency
    global x_dist
    global y_dist
    global z_dist
    global deepvalue
    global reward
    global G
    global S
    global log_dimensionless_jerk
    global sal
    global av
    global scale_addx
    global scale_addy
    global scale_addz
    global scale_addex
    global scale_rotation
    global observations
    global features
    global actions
    global rewards
    global terminals
    global feedback
    global feedbacks

    ##offline RL 参数
    observations = np.zeros((1000, 9))
    features = np.zeros((1000,8))
    actions = np.zeros((1000, 4))
    rewards = np.zeros(1000)
    feedbacks = np.zeros(1000)
    terminals = np.random.randint(2, size=1000)

    total_distance = 0
    feedback = 0
    total_distance_hand = 0.0001
    G = 0
    S = 0
    log_dimensionless_jerk = 3
    av= 200
    sal =200
    startime = time.time()
    scale_rotation = 398
    print(111)
    for i in range(1000):
        data, addr = sock.recvfrom(1024)  # 接收最大1024字节的数据
        message_data = data.decode('utf-8')
        if "HandPosition:" in message_data:
            message_data = message_data.replace("HandPosition:", "")
            x, y, z, eulerX, eulerY, eulerZ = map(lambda s: float(s.replace(',', '')), message_data.split())
            #print('Hand position x:', x, 'y:', y, 'z:', z)
            #print('Hand rotation Euler angles x:', eulerX, 'y:', eulerY, 'z:', eulerZ)
            # eye tracking data
        # elif "GazeOrigin:" in message_data and "GazeDirection:" in message_data:
        #     gaze_data = message_data.split("GazeDirection:")
        #     gaze_origin_data = gaze_data[0].replace("GazeOrigin:", "").split()
        #     gaze_direction_data = gaze_data[1].split()
        #     x_origin, y_origin, z_origin = map(lambda s: float(s.replace(',', '')), gaze_origin_data)
        #     x_direction, y_direction, z_direction = map(lambda s: float(s.replace(',', '')), gaze_direction_data)
        #     print('Gaze origin x:', x_origin, 'y:', y_origin, 'z:', z_origin)
        #     print('Gaze direction x:', x_direction, 'y:', y_direction, 'z:', z_direction)
        elif message_data == "1":
            cluster = True
            first_run = False
            print('Received command: cluster')
        elif message_data == "increase":
            print('Received command: increase')
            feedback = 5
        elif message_data == "decrease":
            print('Received command: decrease')
            feedback = -5
        elif "deepvalue:" in message_data:
            deepvalue = float(message_data.replace("deepvalue:", ""))
        if x == 0:
            first_run = True
        if first_run or cluster:
            if not first_run:
                cluster = False
                first_run = True
                x_pre = 0
                y_pre = 0
                z_pre = 0
            else:
                first_run = False
        else:
            dx = x - x_pre
            dy = y - y_pre
            dz = z - z_pre
            deulerX = eulerX - eulerX_pre
            deulerY = eulerY - eulerY_pre
            timestamps.append(time.time())
            #print('dx:', dx, 'dy:', dy, 'dz:', dz)
            # dx_filtered = dx_filter.update(dx)
            # dy_filtered = dy_filter.update(dy)
            # dz_filtered = dz_filter.update(dz)
            dx_filtered = dx_filter.update(dx)
            dy_filtered = dy_filter.update(dy)
            dz_filtered = dz_filter.update(dz)
            deulerX_filtered = deulerX_filter.update(deulerX)
            deulerY_filtered = deulerY_filter.update(deulerY)
            points_hand.append([dx_filtered, dy_filtered, dz_filtered])
            scale_addx = np.random.uniform(-3000,3000)
            scale_addy = np.random.uniform(-3000, 3000)
            scale_addz = np.random.uniform(-4000, 4000)
            #范围？
            scale_addex = np.random.uniform(-3000,3000)
            #
            scale_ada_x = 6000 + scale_addx + 10000*1 / (1 + np.exp(-abs(x_dist)/500))
            scale_ada_y = 6000 + scale_addy + 10000*1 / (1 + np.exp(-abs(y_dist)/500))
            scale_ada_z = 6000 + scale_addz + 10000*1 / (1 + np.exp(-abs(z_dist)/500))
            #第四个关节
            scale_ada_ex = 60 + scale_addex/100
            #
            scale1_list.append(scale_addx)
            scale2_list.append(scale_addy)
            scale3_list.append(scale_addz)
            disx = dx_filtered * scale_ada_x
            disy = dy_filtered * scale_ada_y
            disz = dz_filtered * scale_ada_z
            diseulerX = deulerX_filtered*scale_ada_ex
            diseulerY = deulerY_filtered*scale_rotation
            print("电机的转角：",diseulerX)
            # if move_ahead ==1:
            #     disz = disz + 500
            #     move_ahead = 0
            #real robot
            # if dx_filtered > 0.014284:
            #     disx = 9999
            # else:
              #     disx = dx_filtered*700000
            # if dy_filtered > 0.014284:
            #     disy = 9999
            # else:
            #     disy = dy_filtered*700000
            # if dz_filtered > 0.014284:
            #     disz = 9999
            # else:
            #     disz = dz_filtered*700000
            addx = addx + disx
            addy = addy + disy
            addz = addz + disz
            position = position + diseulerX
            rotation = rotation + diseulerY
            dx_list.append(addx)
            dy_list.append(addy)
            dz_list.append(addz)
            eulerX_list.append(diseulerX)
            eulerY_list.append(diseulerY)
            #dev.goto_pos([9999+addx, 9999-addz, 9999-addy, 19999], 10000 )
            #dev.goto_pos([9999 -addy, 9999 - addx, 9999 - addz, 19999], 10000)
            dev.goto_pos([9999 - addy, 9999 - addx, 9999 - addz, 9999+position], 10000)
            cmd = 'LA' + str(rotation) + '\n'
            ser.write(cmd.encode())
            #time.sleep(0.05)
            cmd = 'M' + '\n'
            ser.write(cmd.encode())
            #  time.sleep(0.05)
            #dev.goto_pos([9999 + addx, 9999 - addy, 9999 - addz, 19999], 10000)
            print(addx,addy,addz)
            point_now = np.array([[addx, addy-position*0.707107, addz+position*0.707107]])
            trajectory_error, x_dist, y_dist, z_dist = calculate_error(resampled_track, point_now)
            print("轨迹误差值：", trajectory_error)
            print("最近点在 x 方向上的距离：", x_dist)
            print("最近点在 y 方向上的距离：", y_dist)
            print("最近点在 z 方向上的距离：", z_dist)
            #time.sleep(0.5)  # sleep 0.5 sec
            #[+-0.0275]
            #(disx/9999)*0.0275
            # startPos[0] += (dz_filtered/9999)  # increase z by one
            # startPos[1] += (dx_filtered/20)
            # startPos[2] += (dy_filtered/20)
            startPos[0] += (disx/9999)*0.0275 # increase z by one
            startPos[1] += (disy/9999)*0.0275
            startPos[2] += (disz/9999)*0.0275
            posString = '({})'.format(','.join(map(str, startPos)))  # Adding parentheses to the string, example "(0,0,0)"
            print(posString)
            # Add the new point to the list
            points.append([addy-position*0.707107, addx, addz+position*0.707107])
            if len(points) >= 10:  # Replace 4 with the actual number of points you need
                points_array = np.array(points)
                G = calculate_G(points_array)
                S = calculate_S(points_array, sigma, vp)
                sal = calculate_sal(points_array, freq_range)
                log_dimensionless_jerk = calculate_log_dimensionless_jerk(points_array, sigma, vp)
                av = calculate_av(points_array)
                G_values.append(G)
                S_values.append(S)
                sal_val.append(sal)
                log_jek.append(log_dimensionless_jerk)
                av_val.append(av)
               # print("G:", G)
               # print("S:", S)
            if len(points) >= 2:
                # Get the last two points
                last_point = points[-1]
                second_last_point = points[-2]

                # Calculate the Euclidean distance between the two points
                distance_robot = ((last_point[0] - second_last_point[0]) ** 2 +
                            (last_point[1] - second_last_point[1]) ** 2 +
                            (last_point[2] - second_last_point[2]) ** 2) ** 0.5

                # Add the distance to the total distance
                total_distance += distance_robot
            if len(points_hand) >= 2:
                # Get the last two points
                last_point_hand = points_hand[-1]
                second_last_point_hand = points_hand[-2]

                # Calculate the Euclidean distance between the two points
                distance_hand = ((last_point_hand[0] - second_last_point_hand[0]) ** 2 +
                                 (last_point_hand[1] - second_last_point_hand[1]) ** 2 +
                                 (last_point_hand[2] - second_last_point_hand[2]) ** 2) ** 0.5

                # Add the distance to the total distance
                total_distance_hand += distance_hand
                total_distance_hand_list.append(total_distance_hand)
            effiency = total_distance / total_distance_hand
            effiency_list.append(effiency)
            print("机器人轨迹长度：", total_distance)
            print("手轨迹长度：", total_distance_hand)
            print("效率：", effiency)
            #print("时间：", timing)
            reward = (-G - S)*15  + feedback + (av-120)/50 + (200-sal)/50 #+ (200000-effiency)/3000
            endtime = time.time()
            timing = endtime - startime
            feedback = 0

            #手部运动距离作为reward一部分
            # if timing > 10:
            #     reward += (0.04 -total_distance_hand) * 300
            #     startime = time.time()

            rewards[i] = reward
            observations[i] = [addx, addy, addz, scale_ada_x, scale_ada_y, scale_ada_z,x_dist,y_dist,z_dist]
            actions[i] = [scale_addx, scale_addy, scale_addz,scale_ada_ex]
            feedbacks[i] = feedback
            features[i] = [addx,addy,addz,sal,log_dimensionless_jerk,av,G,S]


            sock.sendto(posString.encode("UTF-8"), (host, port))  # Sending string to Unity
        #startPos = [0.2955, 0.027, 0.04105353]
        x_pre = x
        y_pre = y
        z_pre = z
        eulerX_pre = eulerX
        eulerY_pre = eulerY
    np.save('rewards.npy', rewards)
    np.save('observations.npy', observations)
    np.save('actions.npy', actions)
    np.save('terminals.npy', terminals)
    np.save('feedbacks.npy',feedbacks)