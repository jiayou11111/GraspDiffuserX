from Robotic_Arm.rm_robot_interface import *
import socket
import struct
import cv2
import numpy as np
from collections import deque
import threading


class RealSenseCameraBuffer:
    def __init__(self, maxlen=300):

        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def get_closest_before(self, target_ts: float):
        with self.lock:
            candidates = [(ts, img) for ts, img in self.buffer if ts <= target_ts]
            if not candidates:
                return None
            ts, img = min(candidates, key=lambda x: abs(x[0] - target_ts))
            return img

class CameraSocket(threading.Thread):
    def __init__(self, cfg, cam_buf: RealSenseCameraBuffer):
        super().__init__(daemon=True)

        self.c_ip = cfg["c_ip"]
        self.name = cfg["name"]
        self.port = cfg["port"]
    
        self.deque = cam_buf.buffer
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.c_ip, self.port))
        print("Connected to server")



    def recv_all(self, sock, length):
        data = b""
        while len(data) < length:
            packet = sock.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _recv_one_frame(self):
        # ===== 时间戳 =====
        timestamp_bytes = self.recv_all(self.sock, 8)
        if timestamp_bytes is None:
            return None
        timestamp = struct.unpack("!d", timestamp_bytes)[0]

        # ===== RGB =====
        rgb_len = struct.unpack("!I", self.recv_all(self.sock, 4))[0]
        rgb_bytes = self.recv_all(self.sock, rgb_len)

        # ===== Depth =====
        depth_len = struct.unpack("!I", self.recv_all(self.sock, 4))[0]
        depth_bytes = self.recv_all(self.sock, depth_len)

        color = cv2.imdecode(
            np.frombuffer(rgb_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        
        depth = cv2.imdecode(
            np.frombuffer(depth_bytes, np.uint8),
            cv2.IMREAD_UNCHANGED
        )

        return timestamp, color, depth

    
    def run(self):

        while True:
            ts, img ,depth = self._recv_one_frame()
            # cv2.imshow(f"{self.name}_RGB", img)

            # print("img_success")
            self.deque.append((ts, img))
            # self.deque.append((ts, img, depth))
            # print("deque_length:", len(self.deque))
