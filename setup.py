from setuptools import setup

package_name = 'obstacle_stop'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],  # サブフォルダ名と一致
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kadowaki',
    maintainer_email='kadowaki@todo.todo',
    description='Obstacle stop node',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'obstacle_stop_node = obstacle_stop.obstacle_stop_node:main',
            'pointcloud2_depth = obstacle_stop.pointcloud2_depth:main',
            'scan_stop_node = obstacle_stop.scan_stop_node:main',
            'cmd_vel_subscriber = obstacle_stop.cmd_vel_subscriber:main',
        ],
    },
)
