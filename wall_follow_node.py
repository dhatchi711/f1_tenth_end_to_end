#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TLNModel(nn.Module):
    def __init__(self, input_length):
        super(TLNModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(in_channels=24, out_channels=36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(in_channels=36, out_channels=48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            self.flattened_size = x.numel()

        self.fc1 = nn.Linear(self.flattened_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_length = 1080  # Adjust this if your lidar vector length differs
        self.model = self.load_model("/sim_ws/src/wall_follow/scripts/all_csvs_model.pth")
        self.model.eval()

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Create subscribers and publishers
        self.sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Set PID gains
        self.kp = 2
        self.kd = 0
        self.ki = 0

        # Store history
        self.integral = 0.0
        self.prev_error = 0.0
        self.error = 0.0

        # Store any necessary values
        self.start_t = -1
        self.curr_t = 0.0
        self.prev_t = 0.0
        self.lookahead = 0.5

        self.get_logger().info("DriveNode initialized and listening to /scan")
    
    def load_model(self, path):
        model = TLNModel(input_length=self.input_length)
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        return model


    def preprocess_lidar(self, ranges):
        proc = np.array(ranges, dtype=np.float32)
        proc = np.nan_to_num(proc, nan=0.0, posinf=10.0, neginf=0.0)
        proc = np.clip(proc, 0.0, 10.0)
        proc = proc.reshape(1, 1, -1)
        return torch.tensor(proc, dtype=torch.float32).to(self.device)

    def scan_callback(self, msg: LaserScan):
        if len(msg.ranges) != self.input_length:
            self.get_logger().warn(f"Unexpected LiDAR length: {len(msg.ranges)} (expected {self.input_length})")
            return

        lidar_tensor = self.preprocess_lidar(msg.ranges)

        with torch.no_grad():
            output = self.model(lidar_tensor)

        steering_angle = float(output[0][0].cpu())
        speed = float(output[0][1].cpu())

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.pub.publish(drive_msg)

        self.get_logger().info(f"Published - Steering: {steering_angle:.3f}, Speed: {speed:.3f}")



def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
