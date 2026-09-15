import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    os.environ.setdefault('TURTLEBOT3_MODEL', 'burger')

    pkg_share = get_package_share_directory('custom_world')
    world_file = os.path.join(pkg_share, 'worlds', 'showcase.world')

    # Gazebo Harmonic searches GZ_SIM_RESOURCE_PATH for model:// resources.
    models_path = os.path.join(pkg_share, 'models')
    existing_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')

    if existing_resource_path:
        os.environ['GZ_SIM_RESOURCE_PATH'] = (
            models_path + os.pathsep + existing_resource_path
        )
    else:
        os.environ['GZ_SIM_RESOURCE_PATH'] = models_path

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

    ld = LaunchDescription()
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)

    return ld