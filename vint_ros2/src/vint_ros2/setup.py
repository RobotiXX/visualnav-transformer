from glob import glob
from setuptools import find_packages, setup


package_name = 'vint_ros2'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nle',
    maintainer_email='nhatleminh1997@gmail.com',
    description='ROS 2 data, inference, and route tools for ViNT navigation',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'convert_bag = vint_ros2.convert_bag:main',
            'compose_nav2_profile = vint_ros2.compose_nav2_profile:main',
            'image_goal_inference = vint_ros2.image_goal_inference:main',
            'map_pose_sampler = vint_ros2.map_pose_sampler:main',
            'pointgoal_inference = vint_ros2.pointgoal_inference:main',
            'prepare_dataset = vint_ros2.prepare_dataset:main',
            'validate_route = vint_ros2.validate_route:main',
            'write_experiment_context = '
            'vint_ros2.write_experiment_context:main',
        ],
    },
)
