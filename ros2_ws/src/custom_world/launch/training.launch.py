import os
import sys

from ament_index_python.packages import get_package_prefix
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# sim_guard.py sits next to this file; realpath follows the
# --symlink-install link back to it.
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from sim_guard import prepend_path, refuse_if_gazebo_running  # noqa: E402


def generate_training_launch(context, *args, **kwargs):
    stage = LaunchConfiguration("stage").perform(context)

    if stage not in {"2", "3", "4"}:
        raise RuntimeError(
            f"Invalid training stage '{stage}'. "
            "Expected one of: 2, 3, 4."
        )

    pkg_share = get_package_share_directory("custom_world")
    pkg_tb3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")

    world_file = os.path.join(
        pkg_share,
        "worlds",
        f"training_stage{stage}.world",
    )

    prepend_path(
        "GZ_SIM_RESOURCE_PATH",
        os.path.join(pkg_share, "models"),
    )

    # The stage 3/4 obstacle-motion systems (libobstacles.so and
    # friends) ship in lib/turtlebot3_gazebo, which Gazebo does not
    # search by default. Without this they fail to load and the
    # "moving" obstacles stay put.
    prepend_path(
        "GZ_SIM_SYSTEM_PLUGIN_PATH",
        os.path.join(
            get_package_prefix("turtlebot3_gazebo"),
            "lib",
            "turtlebot3_gazebo",
        ),
    )

    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    use_sim_time = LaunchConfiguration("use_sim_time")

    gui = LaunchConfiguration("gui").perform(context).lower()

    if gui not in {"true", "false"}:
        raise RuntimeError(
            f"Invalid gui value '{gui}'. Expected true or false."
        )

    # Without the GUI, run the server only and render the GPU LiDAR
    # offscreen. This frees the CPU/GPU the GUI uses so the
    # simulation can run faster than real time.
    gz_mode_args = "" if gui == "true" else "-s --headless-rendering "

    refuse_if_gazebo_running()

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_ros_gz_sim,
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={
            "gz_args": f"-r {gz_mode_args}-v1 {world_file}",
            "on_exit_shutdown": "true",
        }.items(),
    )

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_tb3_gazebo,
                "launch",
                "robot_state_publisher.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )

    spawn_turtlebot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_tb3_gazebo,
                "launch",
                "spawn_turtlebot3.launch.py",
            )
        ),
        launch_arguments={
            "x_pose": x_pose,
            "y_pose": y_pose,
        }.items(),
    )

    set_pose_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/world/default/set_pose"
            "@ros_gz_interfaces/srv/SetEntityPose"
        ],
        output="screen",
    )

    return [
        gazebo,
        robot_state_publisher,
        spawn_turtlebot,
        set_pose_bridge,
    ]


def generate_launch_description():
    os.environ.setdefault("TURTLEBOT3_MODEL", "burger")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "stage",
                default_value="2",
                description="Training world stage: 2, 3, or 4",
            ),
            DeclareLaunchArgument(
                "x_pose",
                default_value="0.0",
                description="Initial TurtleBot X position",
            ),
            DeclareLaunchArgument(
                "y_pose",
                default_value="0.0",
                description="Initial TurtleBot Y position",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use Gazebo simulation time",
            ),
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Show the Gazebo GUI (false = headless server)",
            ),
            OpaqueFunction(
                function=generate_training_launch,
            ),
        ]
    )