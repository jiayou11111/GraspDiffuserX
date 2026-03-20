from Robotic_Arm.rm_robot_interface import *
import numpy as np

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

handle = arm.rm_create_robot_arm("169.254.128.19", 8080)

weita_init_state = [-162.3698, -119.6334,  155.1919,  -88.6483,   39.4240,  108.4789, -97.4292]


pink_weita_init_state = [-157.79257677902874, -113.23364905170459, 156.54925836359482, -78.6441869596568, -140.68978659437363, -115.77185208413414, 66.60634368395]#demo1


# for res in pink_weita_init_state:

    # 关节阻塞运动到[-114.4785,  -83.8620,  143.5297,    4.4633, -164.5899,  -76.9528,
    #       93.2380]，速度20， 加速度0，时间0， 阻塞模式1

    # res = res / 180 * np.pi
    # print(res)

    # arm.rm_movej(res[:7], 10, 0, 0, 1)

arm.rm_movej(pink_weita_init_state, 10, 0, 0, 1) #grasp_weita_success_init_state

arm.rm_set_gripper_position(1000, True, 10)

# arm.rm_delete_robot_arm()

