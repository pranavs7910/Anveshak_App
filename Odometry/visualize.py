import math
from matplotlib.patches import Polygon, Circle
from world.obstacles import get_world_obstacles

ARENA_SIZE = 20.0
LIDAR_RANGE = 4.0


class Visualizer:
    def __init__(self, fig):
        # store the figure we want to draw on
        self.fig = fig
        self.ax = fig.add_subplot(111)

        # history of robot path
        self.real_xs = []
        self.real_ys = []
        self.idea_xs = []
        self.idea_ys = []

    def update(
        self,
        odom,
        lidar_points,
        lidar_rays,
        lidar_hits,
        obstacles,
        show_lidar=True,
        show_odom=True
    ):
        # ---------- IMPORTANT ----------
        # Clear ONLY this figure, not all figures
        self.ax.clear()

        self.ax.set_aspect("equal")
        self.ax.set_xlim(0, ARENA_SIZE)
        self.ax.set_ylim(0, ARENA_SIZE)
        self.ax.set_title("Autonomy Debug View")

        # -------------------------------
        # Store robot trajectory
        # -------------------------------
        self.real_xs.append(odom.gt_x)
        self.real_ys.append(odom.gt_y)
        self.idea_xs.append(odom.x)
        self.idea_ys.append(odom.y)

        # -------------------------------
        # Draw obstacles
        # -------------------------------
        for obs in get_world_obstacles():
            poly = Polygon(
                obs.corners(),
                closed=True,
                facecolor="lightgray",
                edgecolor="black",
                linewidth=2
            )
            self.ax.add_patch(poly)

        # -------------------------------
        # Draw LiDAR
        # -------------------------------
        if show_lidar:

            # lidar circle
            circle = Circle(
                (odom.gt_x, odom.gt_y),
                LIDAR_RANGE,
                edgecolor="gray",
                facecolor="none",
                linewidth=1
            )
            self.ax.add_patch(circle)

            # lidar rays
            for (x0, y0), (x1, y1) in lidar_rays:
                self.ax.plot([x0, x1], [y0, y1], color="gray", linewidth=0.5)

            # hit points
            if lidar_hits:
                hx, hy = zip(*lidar_hits)
                self.ax.scatter(hx, hy, s=5)

        # -------------------------------
        # Draw odometry path
        # -------------------------------
        if show_odom:
            self.ax.plot(
                self.idea_xs,
                self.idea_ys,
                "--r",
                label="Odometry"
            )
            self.ax.scatter([odom.x], [odom.y], color="red")

        # -------------------------------
        # Draw ground truth
        # -------------------------------
        self.ax.plot(self.real_xs, self.real_ys, color="green", label="Ground Truth")
        self.ax.scatter([odom.gt_x], [odom.gt_y], color="green")

        # heading arrow
        self.ax.arrow(
            odom.gt_x,
            odom.gt_y,
            0.8 * math.cos(odom.gt_theta),
            0.8 * math.sin(odom.gt_theta),
            head_width=0.15,
            color="green"
        )

        self.ax.legend()

        # redraw THIS figure only
        self.fig.canvas.draw_idle()