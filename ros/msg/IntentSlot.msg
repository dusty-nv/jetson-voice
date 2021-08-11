# the original query text
std_msgs/String query

# the classified intent label
std_msgs/String intent

# the intent probability between [0,1]
float32 score

# list of slots
jetson_voice_ros/Slot[] slots