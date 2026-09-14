import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Where colcon installed OUR package's data (never hardcode install paths)
    os.environ.setdefault('TURTLEBOT3_MODEL', 'burger')
    pkg_share = get_package_share_directory('custom_world')
    world_file = os.path.join(pkg_share, 'worlds', 'showcase.world')

    # Let Gazebo find the vendored warehouse models.
    # The world file references models by NAME; GAZEBO_MODEL_PATH is the
    # search list Gazebo consults. We append ours, keeping existing entries
    # (which include turtlebot3_gazebo's models, set by its apt env hook).
    models_path = os.path.join(pkg_share, 'models')
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] += ':' + models_path
    else:
        os.environ['GAZEBO_MODEL_PATH'] = models_path

    # Spawn-pose knobs — overridable from the command line
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    # 1. Physics server, loading OUR world
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file}.items(),
    )

    # 2. The GUI window
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # 3. Publishes the robot's TF frames (odom, base_link, lidar frame).
    #    Silent now — but your Week 2 state vector reads /odom and /scan,
    #    and this node is what makes those coordinates meaningful.
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    # 4. Spawn the burger at (x_pose, y_pose)
    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, 'launch', 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={'x_pose': x_pose, 'y_pose': y_pose}.items(),
    )

    ld = LaunchDescription()
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    return ld