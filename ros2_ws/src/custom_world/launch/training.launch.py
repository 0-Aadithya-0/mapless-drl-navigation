import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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

    models_path = os.path.join(pkg_share, "models")

    existing_resource_path = os.environ.get(
        "GZ_SIM_RESOURCE_PATH",
        "",
    )

    if existing_resource_path:
        os.environ["GZ_SIM_RESOURCE_PATH"] = (
            models_path + os.pathsep + existing_resource_path
        )
    else:
        os.environ["GZ_SIM_RESOURCE_PATH"] = models_path

    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    use_sim_time = LaunchConfiguration("use_sim_time")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_ros_gz_sim,
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={
            "gz_args": f"-r -v1 {world_file}",
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
            OpaqueFunction(
                function=generate_training_launch,
            ),
        ]
    )