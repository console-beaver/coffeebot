from launch import LaunchDescription
from launch_ros.actions import PushRosNamespace
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    stretch_core_path = get_package_share_directory("stretch_core")

    return LaunchDescription([
        GroupAction([
            PushRosNamespace("robot1"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    stretch_core_path + "/launch/stretch_driver.launch.py"
                )
            )
        ]),
        GroupAction([
            PushRosNamespace("robot2"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    stretch_core_path + "/launch/stretch_driver.launch.py"
                )
            )
        ])
    ])
