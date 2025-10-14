# obstacle_stop/launch/obstacle_stop.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('pointcloud_topic', default_value='/camera/depth/color/points'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument('min_distance', default_value='0.5'),
        Node(
            package='obstacle_stop',
            executable='obstacle_stop_node',
            name='obstacle_stop_node',
            parameters=[{
                'pointcloud_topic': LaunchConfiguration('pointcloud_topic'),
                'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
                'min_distance': LaunchConfiguration('min_distance'),
                # 必要なら他のパラメータもここに追加
            }],
            output='screen'
        )
    ])
