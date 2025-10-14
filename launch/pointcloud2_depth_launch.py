from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='your_package_name',
            executable='height_slice_laserscan',
            name='height_slice_laserscan',
            output='screen'
        )
    ])
