from Robotic_Arm.rm_robot_interface import *
import numpy as np
import time

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

handle = arm.rm_create_robot_arm("169.254.128.19", 8080)

weita_init_state = [-162.3698, -119.6334,  155.1919,  -88.6483,   39.4240,  108.4789, -97.4292]


pink_weita_init_state = [-157.79257677902874, -113.23364905170459, 156.54925836359482, -78.6441869596568, -140.68978659437363, -115.77185208413414, 66.60634368395]#demo1

# arm.rm_movej(pink_weita_init_state, 10, 0, 0, 1) #grasp_weita_success_init_state

# # for res in init_state:

# #     # 关节阻塞运动到[-114.4785,  -83.8620,  143.5297,    4.4633, -164.5899,  -76.9528,
# #     #       93.2380]，速度20， 加速度0，时间0， 阻塞模式1

# #     res = res * 180 /np.pi
# #     print(res)

#     # arm.rm_movej(res[:7], 10, 0, 0, 1)



# arm.rm_movej(pink_weita_init_state, 10, 0, 0, 1) #grasp_weita_success_init_state

# arm.rm_set_gripper_position(1000, True, 10)

# arm.rm_delete_robot_arm()



# 轨迹数据 (Step 0 - Step 83)
# 包含 7个关节角度 + 1个夹爪维度

trajectory_data = [
    [-157.3070, -112.0590, 154.4470, -75.0290, -137.5950, -109.9550, 66.4780, 57.2958],
    [-157.2210, -112.0600, 154.3570, -75.1430, -137.7020, -109.9630, 66.4720, 57.2958],
    [-156.9130, -112.1480, 154.1580, -75.5350, -138.0760, -109.9860, 66.4650, 57.2958],
    [-156.6030, -112.2290, 153.9220, -75.8980, -138.4330, -110.0450, 66.4670, 57.2958],
    [-156.2870, -112.3180, 153.6540, -76.1980, -138.7680, -110.0990, 66.5040, 57.2958],
    [-155.6660, -112.4390, 153.4650, -76.4400, -139.4110, -110.2160, 66.6190, 57.2958],
    [-155.4590, -112.4840, 153.4060, -76.4780, -139.6670, -110.2820, 66.6800, 57.2958],
    [-154.7760, -112.6410, 153.2640, -76.5960, -140.3980, -110.5070, 66.8090, 57.2958],
    [-153.8190, -112.9170, 153.0720, -76.9350, -141.5570, -110.7250, 67.1630, 57.2958],
    [-152.4770, -113.4020, 152.4350, -77.6550, -143.2320, -111.0950, 68.0890, 57.2958],

]


if __name__ == "__main__":

    start_obs = trajectory_data[0]
    # 前7个是关节角度
    start_joints = start_obs[:7]
        # 第8个是夹爪(假设)
    start_gripper = start_obs[7] 

    # 运动到初始关节, 速度20, 阻塞等待
    ret = arm.rm_movej(start_joints, 20, 0, 0, 1)
    print(f"[Info] Reached start position. Ret: {ret}")

    time.sleep(1.0)
        
    # 2. 依次执行轨迹
    input("Press Enter to continue trajectory execution...")
        
    for i, step in enumerate(trajectory_data):
        joints = step[:7]
        gripper_val = step[7]
            
        print(f"Step {i}/{len(trajectory_data)}: joints={joints}")
        arm.rm_movej(joints, 30, 0, 0, 1)

        if gripper_val < 30:
            arm.rm_set_gripper_position(0, True, 10) # Close
        else:
            arm.rm_set_gripper_position(1000, True, 10) # Open

        print("[Info] Trajectory finished.")

    # 断开连接
    arm.rm_delete_robot_arm()