import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import pygame
import math
import sys

WIDTH, HEIGHT = 600, 600
MAP_SIZE = 2.0  # meters
SCALE = WIDTH / MAP_SIZE  # pixels per meter
CRATE_POS = [1.7, 0.85]  # hardcoded crate position

def get_yaw_from_quaternion(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class ForkliftDisplay(Node):
    def __init__(self):
        super().__init__('forklift_display')

        self.forklift_pos = None
        self.forklift_yaw = 0.0
        self.goal_pos = None
        self.crate_goal_pos = None
        self.path_history = []

        self.state = "IDLE"  # States: IDLE, TO_CRATE, TO_GOAL
        self.robot_pos = pygame.Vector2(0, 0)  # Initialize to bottom-left (0,0)

        self.landmarks = [
            [0.5, 0.5, "Pillar A"],
            [1.2, 0.4, "Box"],
            [1.6, 1.5, "Ramp"],
            [1.8, 0.7, "Dock"]
        ]

        self.create_subscription(PoseStamped, '/fused_pose', self.robot_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Forklift Navigation Display")
        self.font = pygame.font.SysFont(None, 24)

        self.crate_button = pygame.Rect(10, HEIGHT - 80, 100, 30)
        self.goal_button = pygame.Rect(120, HEIGHT - 80, 100, 30)
        self.next_crate_button = pygame.Rect(10, HEIGHT - 40, 180, 30)
        self.show_next_crate = False


        # New flag
        self.calibrated = False
        self.create_subscription(Bool, '/calibration_status', self.calib_callback, 10)

    def calib_callback(self, msg):
        self.calibrated = msg.data
        if msg.data:
            self.get_logger().info("Calibration successful.")


    def robot_callback(self, msg):
        self.forklift_pos = [msg.pose.position.x, msg.pose.position.y]
        self.robot_pos = pygame.Vector2(self.forklift_pos[0], self.forklift_pos[1])


        if not self.path_history or self.forklift_pos != list(self.path_history[-1]):
            self.path_history.append(tuple(self.forklift_pos))

        # qz = msg.pose.orientation.z
        # qw = msg.pose.orientation.w
        # self.forklift_yaw = -math.atan2(2.0 * qz * qw, 1.0 - 2.0 * qz * qz)

        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        self.forklift_yaw = -get_yaw_from_quaternion(qx, qy, qz, qw)

        # print(f"[Display] Pose from /fused_pose: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}")

        if self.goal_pos and self.forklift_pos:
            dist = math.hypot(self.forklift_pos[0] - self.goal_pos[0], self.forklift_pos[1] - self.goal_pos[1])
            if dist < 0.01 and self.state in ["TO_CRATE", "TO_GOAL"]:
                self.get_logger().info("Reached target.")
                self.goal_pos = None
                self.state = "IDLE"

    def goal_callback(self, msg):
        if self.state == "TO_CRATE":
            self.crate_goal_pos = [msg.pose.position.x, msg.pose.position.y]
            self.get_logger().info(f"Crate goal detected: {self.crate_goal_pos}")

    def handle_click(self, pos):
        if not self.calibrated:
            return  # Block control until calibration
        if self.crate_button.collidepoint(pos):
            self.goal_pos = CRATE_POS.copy()
            self.state = "TO_CRATE"
        elif self.goal_button.collidepoint(pos):
            if self.crate_goal_pos:
                self.goal_pos = self.crate_goal_pos.copy()
                self.state = "TO_GOAL"
        if self.show_next_crate and self.next_crate_button.collidepoint(pos):
            self.get_logger().info("Next crate requested.")
            self.state = "TO_CRATE"
            self.goal_pos = CRATE_POS.copy()
            self.show_next_crate = False


    def world_to_screen(self, x, y):
        return int(x * SCALE), int(HEIGHT - y * SCALE)

    def draw_grid(self, spacing=0.1):
        for x in range(int(MAP_SIZE / spacing) + 1):
            screen_x = int(x * spacing * SCALE)
            pygame.draw.line(self.screen, (50, 50, 50), (screen_x, 0), (screen_x, HEIGHT))
        for y in range(int(MAP_SIZE / spacing) + 1):
            screen_y = HEIGHT - int(y * spacing * SCALE)
            pygame.draw.line(self.screen, (50, 50, 50), (0, screen_y), (WIDTH, screen_y))

    def draw_landmarks(self):
        for x, y, label in self.landmarks:
            pos = self.world_to_screen(x, y)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 6)
            self.screen.blit(self.font.render(label, True, (50, 50, 50)), (pos[0] + 8, pos[1] - 8))

    def draw_arrow(self, start, angle, color=(255, 0, 0), size=20):
        end = (
            start[0] + size * math.cos(angle),
            start[1] + size * math.sin(angle)
        )
        pygame.draw.line(self.screen, color, start, end, 3)
        for offset in [math.pi / 6, -math.pi / 6]:
            tip = (
                end[0] - 10 * math.cos(angle - offset),
                end[1] - 10 * math.sin(angle - offset)
            )
            pygame.draw.line(self.screen, color, end, tip, 3)

    def draw_arrow_between(self, start, end):
        pygame.draw.line(self.screen, (0, 255, 0), start, end, 4)
        dx, dy = end[0] - start[0], end[1] - start[1]
        angle = math.atan2(dy, dx)
        for offset in [math.pi / 6, -math.pi / 6]:
            tip_x = end[0] - 10 * math.cos(angle - offset)
            tip_y = end[1] - 10 * math.sin(angle - offset)
            pygame.draw.line(self.screen, (0, 255, 0), end, (tip_x, tip_y), 4)

    def draw_fov_cone(self, origin, angle, fov=math.pi / 3, length=40):
        left_angle = angle - fov / 2
        right_angle = angle + fov / 2

        left = (
            origin[0] + length * math.cos(left_angle),
            origin[1] + length * math.sin(left_angle)
        )
        right = (
            origin[0] + length * math.cos(right_angle),
            origin[1] + length * math.sin(right_angle)
        )

        pygame.draw.polygon(self.screen, (100, 100, 255, 100), [origin, left, right], 1)

    def draw_compass(self):
        compass_center = (50, 50)
        compass_radius = 30
        pygame.draw.circle(self.screen, (80, 80, 80), compass_center, compass_radius, 2)
        self.draw_arrow(compass_center, self.forklift_yaw, color=(255, 255, 0), size=compass_radius - 5)
        label = self.font.render("N", True, (255, 255, 255))
        self.screen.blit(label, (compass_center[0] - 8, compass_center[1] - compass_radius - 12))
    
    def draw_walls(self):
        wall_color = (0, 0, 0)
        wall_thickness = 10  # in pixels

        # Define wall positions (in meters)
        wall1_start = (0.0, 1.4)
        wall1_end = (1.5, 1.4)

        wall2_start = (0.5, 0.6)
        wall2_end = (2.0, 0.6)

        pygame.draw.line(self.screen, wall_color,
                        self.world_to_screen(*wall1_end),
                        self.world_to_screen(*wall1_start),
                        wall_thickness)

        pygame.draw.line(self.screen, wall_color,
                        self.world_to_screen(*wall2_start),
                        self.world_to_screen(*wall2_end),
                        wall_thickness)


    def draw_axes(self):
        pygame.draw.line(self.screen, (0, 0, 255), (0, HEIGHT - 1), (WIDTH, HEIGHT - 1), 2)  # X-axis (bottom)
        pygame.draw.line(self.screen, (0, 255, 0), (0, 0), (0, HEIGHT), 2)           # Y-axis (left)

        self.screen.blit(self.font.render("X", True, (0, 0, 255)), (WIDTH - 15, HEIGHT - 25))
        self.screen.blit(self.font.render("Y", True, (0, 255, 0)), (5, 5))

    def draw_display(self):
        self.screen.fill((220, 230, 255))  # Light gray background
        self.draw_axes()
        self.draw_walls()
        self.draw_landmarks()

        if self.forklift_pos:
            forklift_screen = self.world_to_screen(*self.forklift_pos)
            pygame.draw.circle(self.screen, (255, 0, 0), forklift_screen, 8)
            self.draw_arrow(forklift_screen, self.forklift_yaw)
            self.draw_fov_cone(forklift_screen, self.forklift_yaw)

            for pos in self.path_history:
                trail = self.world_to_screen(*pos)
                pygame.draw.circle(self.screen, (100, 100, 100), trail, 2)

        if self.goal_pos:
            goal_screen = self.world_to_screen(*self.goal_pos)
            pygame.draw.circle(self.screen, (0, 150, 255), goal_screen, 8)

            if self.forklift_pos:
                forklift_screen = self.world_to_screen(*self.forklift_pos)
                self.draw_arrow_between(forklift_screen, goal_screen)

        self.draw_compass()

        pygame.draw.rect(self.screen, (200, 200, 200), pygame.Rect(0, HEIGHT - 40, WIDTH, 40))
        pos_text = "Forklift: N/A" if not self.forklift_pos else f"Forklift: ({self.forklift_pos[0]:.2f}, {self.forklift_pos[1]:.2f})"
        goal_text = "Goal: N/A" if not self.goal_pos else f"Goal: ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f})"
        status = self.font.render(f"{pos_text} | {goal_text}", True, (0, 0, 0))
        self.screen.blit(status, (10, HEIGHT - 30))

        # Draw buttons
        pygame.draw.rect(self.screen, (80, 80, 80), self.crate_button)
        pygame.draw.rect(self.screen, (80, 80, 80), self.goal_button)
        self.screen.blit(self.font.render("Go to Crate", True, (255, 255, 255)), (self.crate_button.x + 5, self.crate_button.y + 5))
        self.screen.blit(self.font.render("Go to Goal", True, (255, 255, 255)), (self.goal_button.x + 5, self.goal_button.y + 5))

        if self.state == "TO_CRATE" and self.robot_pos.distance_to(CRATE_POS) < 0.05:
            self.state = "IDLE"
            self.goal_pos = None
            self.reached_crate = True
            self.show_next_crate = False
            crate_text = self.font.render("Crate reached, scan to get goal position", True, (255, 255, 0))
            self.screen.blit(crate_text, (10, HEIGHT - 60))

        elif self.state == "TO_GOAL" and self.goal_pos and self.robot_pos.distance_to(self.goal_pos) < 0.05:
            self.state = "IDLE"
            self.goal_pos = None
            self.reached_crate = False
            self.show_next_crate = True
            goal_text = self.font.render("Goal reached", True, (0, 255, 0))
            self.screen.blit(goal_text, (10, HEIGHT - 60))

        # Draw next crate button if allowed
        if self.show_next_crate:
            pygame.draw.rect(self.screen, (0, 150, 255), self.next_crate_button)
            next_text = self.font.render("Go to next crate", True, (255, 255, 255))
            self.screen.blit(next_text, (self.next_crate_button.x + 10, self.next_crate_button.y + 5))

        if not self.calibrated:
            calib_text = self.font.render("Waiting for calibration...", True, (255, 0, 0))
        else:
            calib_text = self.font.render("Calibration successful. You may proceed.", True, (0, 128, 0))
        self.screen.blit(calib_text, (WIDTH - 300, HEIGHT - 30))

        pygame.display.flip()

def main(args=None):
    rclpy.init(args=args)
    node = ForkliftDisplay()
    clock = pygame.time.Clock()

    try:
        while rclpy.ok():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    node.handle_click(event.pos)

            rclpy.spin_once(node, timeout_sec=0.01)
            node.draw_display()
            clock.tick(30)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        pygame.quit()

if __name__ == '__main__':
    main()
