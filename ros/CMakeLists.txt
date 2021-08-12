cmake_minimum_required(VERSION 3.5)
project(jetson_voice_ros)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Audio.msg"
  "msg/AudioInfo.msg"
  "msg/IntentSlot.msg"
  "msg/QuestionAnswerQuery.msg"
  "msg/QuestionAnswerResult.msg"
  "msg/Slot.msg"
  DEPENDENCIES std_msgs
)

# install Python modules
ament_python_install_package(${PROJECT_NAME})

# install Python executables
file(GLOB python_nodes ${PROJECT_NAME}/*.py)

install(PROGRAMS
  ${python_nodes}
  DESTINATION lib/${PROJECT_NAME}
)

# install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)
 
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
