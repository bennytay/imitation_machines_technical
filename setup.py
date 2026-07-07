from setuptools import setup

package_name = 'so101_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'mujoco', 'numpy'],
    zip_safe=True,
    maintainer='bennytay',
    maintainer_email='you@example.com',
    description='MuJoCo-ROS2 bridge for the SO-101 arm',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mujoco_bridge = so101_bridge.mujoco_bridge_node:main',
            'wave_motion = so101_bridge.wave_motion_node:main',
        ],
    },
)
