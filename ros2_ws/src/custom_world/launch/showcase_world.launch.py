import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# sim_guard.py sits next to this file; realpath follows the
# --symlink-install link back to it.
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from sim_guard import prepend_path, refuse_if_gazebo_running  # noqa: E402


def generate_launch_description():
    os.environ.setdefault('TURTLEBOT3_MODEL', 'burger')

    refuse_if_gazebo_running()

    pkg_share = get_package_share_directory('custom_world')
    world_file = os.path.join(pkg_share, 'worlds', 'showcase.world')

    # Gazebo Harmonic searches GZ_SIM_RESOURCE_PATH for model:// resources.
    prepend_path('GZ_SIM_RESOURCE_PATH', os.path.join(pkg_share, 'models'))

    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    # 1. Gazebo Harmonic server
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r -s -v2 {world_file}',
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # 2. Gazebo Harmonic GUI
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': '-g -v2',
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # 3. Robot TF / state publisher
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_tb3_gazebo,
                'launch',
                'robot_state_publisher.launch.py',
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items(),
    )

    # 4. Spawn Burger + start ROS <-> Gazebo bridges
    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_tb3_gazebo,
                'launch',
                'spawn_turtlebot3.launch.py',
            )
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose,
        }.items(),
    )

    # 5. Pose-reset service used by the RL environment's reset()
    set_pose_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/default/set_pose'
            '@ros_gz_interfaces/srv/SetEntityPose'
        ],
        output='screen',
    )

    ld = LaunchDescription()
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(set_pose_bridge_cmd)

    return ld