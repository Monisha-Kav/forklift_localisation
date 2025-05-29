from setuptools import find_packages, setup

package_name = 'my_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/sim_launch.launch.py']),
        ('share/' + package_name + '/launch', ['launch/localisation_bayes_prefuckup.launch.py']),
        ('share/' + package_name + '/launch', ['launch/localization_bayes_fused.launch.py']),
        ('share/' + package_name + '/launch', ['launch/localization_kf_fused.launch.py']),
        ('share/' + package_name + '/launch', ['my_gazebo.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='monisha',
    maintainer_email='monisha@todo.todo',
    description='Camera follower and ArUco detector node for simulation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_follower_node = my_simulation.camera_follower_node:main',
            'aruco_live_detector = my_simulation.aruco_live_detector:main',
            'sensor_fusion_node_sim = my_simulation.sensor_fusion_node_sim:main',
            'forklift_display_sim = my_simulation.forklift_display_sim:main',
            'new_forklift_display_sim = my_simulation.new_forklift_display_sim:main',
            'full_sensor_fusion_fixed = my_simulation.full_sensor_fusion_fixed:main',
            'full_sensor_fusion_node = my_simulation.full_sensor_fusion_node:main',
            'kf_full_sensor_fusion_m = my_simulation.kf_full_sensor_fusion_m:main',
        ],
    },
)

