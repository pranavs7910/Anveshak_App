from robot import Robot
from sensors.lidar import LidarScan
import matplotlib.pyplot as plt
import math
import csv

path = []
with open(r"C:\Users\PRANAV\OneDrive\Desktop\odometry\Autonomous-Rover-Simulation\path.csv") as f:
    reader = csv.reader(f)
    next(reader) 
    for row in reader:
        x = float(row[0])
        y = float(row[1])
        path.append((x, y))

# -----------------------------------------------------
# Runtime modes & visualization toggles
# -----------------------------------------------------
MODE = "MANUAL"        # MANUAL | AUTO
SHOW_LIDAR = True
SHOW_ODOM = True


# -----------------------------------------------------
# Control commands (shared state)
# -----------------------------------------------------
v = 0.0   # linear velocity [m/s]
w = 0.0   # angular velocity [rad/s]


def on_key(event):
    """Keyboard control & visualization toggles."""
    global v, w, MODE, SHOW_LIDAR, SHOW_ODOM

    # --- visualization toggles ---
    if event.key == 'o':
        SHOW_ODOM = not SHOW_ODOM
        print(f"Odometry visualization: {'ON' if SHOW_ODOM else 'OFF'}")
        return

    if event.key == 'l':
        SHOW_LIDAR = not SHOW_LIDAR
        print(f"LiDAR visualization: {'ON' if SHOW_LIDAR else 'OFF'}")
        return

    # --- mode switching ---
    if event.key == 'm':
        MODE = "MANUAL"
        v = 0.0
        w = 0.0
        print("Switched to MANUAL mode")
        return

    if event.key == 'a':
        MODE = "AUTO"
        print("Switched to AUTO mode")
        return

    # --- manual control ---
    if MODE != "MANUAL":
        return

    if event.key == 'up':
        v += 1.5
    elif event.key == 'down':
        v -= 1.5
    elif event.key == 'left':
        w += 2.0
    elif event.key == 'right':
        w -= 2.0
    elif event.key == ' ':
        v = 0.0
        w = 0.0

    # clamp commands
    v = max(min(v, 6.0), -6.0)
    w = max(min(w, 6.0), -6.0)
import os
print(os.getcwd())

if __name__ == "__main__":

    lidar = LidarScan(max_range=4.0)
    robot = Robot()

    plt.close('all')
    fig = plt.figure(num=2)
    fig.canvas.manager.set_window_title("Autonomy Debug View")
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=False)

    dt = 0.01 

    # -------------------------------------------------
    # Main simulation loop
    # -------------------------------------------------
    while plt.fignum_exists(fig.number):

        # ground truth pose
        real_x, real_y, real_theta = robot.get_ground_truth()
        # odometry estimate
        ideal_x, ideal_y, ideal_theta = robot.get_odometry()
        # LiDAR scan 
        lidar_ranges, lidar_points, lidar_rays, lidar_hits = lidar.get_scan((real_x, real_y, real_theta))

        
        if MODE == "AUTO":
            print("AUTO running", v, w)
            # ---------------------------------------------
            # write your autonomous code here!!!!!!!!!!!!!
                        
            lookahead = 1.0
            speed = 5.0
            
            # find closest point on path
            closest_i = 0
            min_d = float("inf")
            
            for i, (px, py) in enumerate(path):
                d = math.hypot(ideal_x - px, ideal_y - py)
                if d < min_d:
                    min_d = d
                    closest_i = i
            
            target_i = min(closest_i + 5, len(path) - 1)
            tx, ty = path[target_i]
            
            dx = tx - ideal_x
            dy = ty - ideal_y
            
            angle_to_target = math.atan2(dy, dx)
            alpha = angle_to_target - ideal_theta
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))
            
            # steering
            w = 2 * speed * math.sin(alpha) / lookahead
            v = speed
            
            
            if closest_i >= len(path) - 3:
                v = 0
                w = 0
            # ---------------------------------------------
            # Allowed inputs:
            #   - real_x, real_y, real_theta
            #   - ideal_x, ideal_y, ideal_theta (odometry you have to use for logic)
            #   - lidar_ranges (lidar data you have to use for logic, array of length 36 corresponding to 36 beams)
            #
            # Required outputs:
            #   - v, w (linear and angular velocity commands)


            

            # ---------------------------------------------
            # don't edit below this line (visualization & robot stepping)
            # ---------------------------------------------
        robot.step(
            lidar_points,
            lidar_rays,
            lidar_hits,
            v,
            w,
            dt,
            show_lidar=SHOW_LIDAR,
            show_odom=SHOW_ODOM
        )

        plt.pause(dt)
