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

L_FREE = 0.5
L_OCC = 1.0
OCCUPIED_STRONG = 3.0
L_MIN = -5.0
L_MAX = 5.0
DIST_SCALE = 1.0  
CELL_SIZE = 0.2
ANGULAR_THRESHOLD = 0.1  # Example threshold# Initialize occupancy grid (assume a size and initialize log-odds)
grid_size = (20, 20)  # Adjust based on your simulation 
# Constants for occupancy grid update
GRID_WIDTH = int(20/CELL_SIZE)   # Example width of the grid
GRID_HEIGHT = int(20/CELL_SIZE)  # Example height of the grid
INITIAL_LOG_ODDS = 0.0  # Initial log-odds for cells
grid = [[INITIAL_LOG_ODDS for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

def clamp(value, min_value, max_value):   
     return max(min(value, max_value), min_value)
# -----------------------------------------------------
# Control commands (shared state)
# -----------------------------------------------------
v = 0.0   # linear velocity [m/s]
w = 0.0   # angular velocity [rad/s]


def RAYTRACE(start, end):
    x0, y0 = start
    x1, y1 = end
    cells = {"before_hit": [], "hit_cell": None}

    x0_idx, y0_idx = int(x0), int(y0)
    x1_idx, y1_idx = int(x1), int(y1)

    dx = abs(x1_idx - x0_idx)
    dy = abs(y1_idx - y0_idx)
    sx = 1 if x0_idx < x1_idx else -1
    sy = 1 if y0_idx < y1_idx else -1
    err = dx - dy

    while True:
        if x0_idx == x1_idx and y0_idx == y1_idx:
            cells["hit_cell"] = (x1_idx, y1_idx)  # DON'T add to before_hit
            break

        cells["before_hit"].append((x0_idx, y0_idx))  # only free cells

        err2 = 2 * err
        if err2 > -dy:
            err -= dy
            x0_idx += sx
        if err2 < dx:
            err += dx
            y0_idx += sy

    return cells


def world_to_grid(x, y):
    gx = int(x / CELL_SIZE)
    gy = int(y / CELL_SIZE)
    return gx, gy

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
import numpy as np
print(os.getcwd())



if __name__ == "__main__":

    lidar = LidarScan(max_range=4.0)
    plt.close('all')

    fig = plt.figure(num=2)
    fig.canvas.manager.set_window_title("Autonomy Debug View")
    fig.canvas.mpl_connect("key_press_event", on_key)

    robot = Robot(fig)


    occ_fig = plt.figure(num=3)
    occ_ax = occ_fig.add_subplot(111)
    occ_img = occ_ax.imshow(
       grid,
       origin="lower",
       cmap="gray",
       vmin=L_MIN,
       vmax=L_MAX
    )
    occ_fig.colorbar(occ_img, ax=occ_ax)  # ← add this right after imshow
    occ_ax.set_title("Occupancy Grid")
    plt.show(block=False)

    dt = 0.01 
    print("Figures:", plt.get_fignums())
    # -------------------------------------------------
    # Main simulation loop
    # -------------------------------------------------
    while plt.fignum_exists(fig.number) and plt.fignum_exists(occ_fig.number):

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
            for i in range(len(lidar_ranges)):
                
                hit_distance = lidar_ranges[i]
            
                # check if obstacle detected
                hit = hit_distance < lidar.max_range - 1e-3
            
                # start and end of ray
                (x0, y0), (x1, y1) = lidar_rays[i]
                
                start = world_to_grid(x0, y0)
                end   = world_to_grid(x1, y1)
                
                cells = RAYTRACE(start, end)
            
                # distance-based confidence
                weight = math.exp(-hit_distance / DIST_SCALE)
            
            
                # mark free cells
                for (cx, cy) in cells["before_hit"]:
                    if 0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT:
                        grid[cy][cx] = clamp(grid[cy][cx] + weight * (-L_FREE), L_MIN, L_MAX)
            
                # mark obstacle cell
                if hit and cells["hit_cell"] is not None:
                    hx, hy = cells["hit_cell"]
                    if 0 <= hx < GRID_WIDTH and 0 <= hy < GRID_HEIGHT:
                        grid[hy][hx] = clamp(grid[hy][hx] + weight * L_OCC, L_MIN, L_MAX)
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
        
        occ_img.set_data(np.array(grid))
        occ_img.set_clim(vmin=L_MIN, vmax=L_MAX)  
        occ_fig.canvas.draw_idle()
        plt.pause(dt)
