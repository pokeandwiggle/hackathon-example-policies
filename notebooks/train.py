# [markdown]
# **⚠️ RESTART KERNEL** before running cells below - code has been updated to fix action dimension mismatch.

# [markdown]
# # 💿 Dataset Conversion
# 
# This notebook converts raw robot recordings (`.mcap` files) into the LeRobot format required for training. 
# 
# The process involves:
# 1.  **Exploring** the available raw data.
# 2.  **Configuring** the dataset parameters (e.g., observations, actions).
# 3.  **Running** the conversion script.

# [markdown]
# --- 
# ## 1. Explore Raw Data
# 
# First, let's list the available raw data directories. Each directory contains a set of `.mcap` files from different teleoperation sessions.

# [markdown]
# --- 
# ## 2. Configure Conversion
# 
# Now, specify the input and output paths and define the dataset's structure. 
# 
# > **Action Required:** Update `RAW_DATA_DIR` and `OUTPUT_DIR` below.

#
import pathlib
from example_policies.data_ops.config.pipeline_config import PipelineConfig, ActionMode

# --- Paths ---
# TODO: Set the input directory containing your .mcap files.
RAW_DATA_DIR = pathlib.Path("./data/stack_daytime/")  # Using the local data directory with sample file

# TODO: Set your desired output directory name.
OUTPUT_DIR = pathlib.Path("./data/stack_daytime_lerobot")

# --- Configuration ---
# TODO: A descriptive label for the task, used for VLA-style text conditioning.
TASK_LABEL = "test"

cfg = PipelineConfig(
    task_name=TASK_LABEL,
    # Observation features to include in the dataset.
    include_tcp_poses=True,
    include_rgb_images=True,
    include_depth_images=False,
    # Action representation. DELTA_TCP is a good default.
    action_level=ActionMode.DELTA_TCP,
    # Subsampling and filtering. These are task-dependent.
    target_fps=10,
    max_pause_seconds=1000.2,
    min_episode_seconds=1,
)

print(f"Input path:  {RAW_DATA_DIR}")
print(f"Output path: {OUTPUT_DIR}")

# [markdown]
# ### Troubleshooting: Enable pause saving or adjust pause detection
# 
# If you're getting "No episodes were successfully converted", it might be because:
# 1. The robot is mostly stationary in the recording, so frames are classified as "pauses"
# 2. `save_pauses=False` by default, so paused frames are discarded
# 
# **Solutions:**
# - Set `save_pauses=True` to save paused frames (good for debugging)
# - Increase `max_pause_seconds` to be more lenient about what's considered a pause
# - Set `max_pause_seconds=999` to effectively disable pause filtering

#
# DEBUG: Inspect MCAP file contents
from mcap.reader import make_reader
import pathlib

mcap_file = list(RAW_DATA_DIR.rglob("*.mcap"))[0]
print(f"Inspecting: {mcap_file}\n")

with open(mcap_file, 'rb') as f:
    reader = make_reader(f)
    
    # Get summary info
    summary = reader.get_summary()
    if summary and summary.statistics:
        duration_ns = summary.statistics.message_end_time - summary.statistics.message_start_time
        duration_s = duration_ns / 1e9
        print(f"Recording duration: {duration_s:.2f} seconds")
        print(f"Total messages: {summary.statistics.message_count}\n")
    
    # Get all channels/topics
    print("Available topics in MCAP:")
    topics = {}
    topic_times = {}
    
    for schema, channel, message in reader.iter_messages():
        if channel.topic not in topics:
            topics[channel.topic] = {
                'schema': schema.name,
                'count': 0
            }
            topic_times[channel.topic] = {'first': None, 'last': None}
        
        topics[channel.topic]['count'] += 1
        
        # Track first and last timestamps
        if topic_times[channel.topic]['first'] is None:
            topic_times[channel.topic]['first'] = message.log_time
        topic_times[channel.topic]['last'] = message.log_time
    
    for topic, info in sorted(topics.items()):
        # Calculate Hz
        if topic in topic_times:
            first_time = topic_times[topic]['first']
            last_time = topic_times[topic]['last']
            if first_time and last_time and last_time > first_time:
                duration_ns = last_time - first_time
                duration_s = duration_ns / 1e9
                hz = info['count'] / duration_s if duration_s > 0 else 0
            else:
                hz = 0
        else:
            hz = 0
            
        print(f"  {topic}")
        print(f"    Schema: {info['schema']}")
        print(f"    Messages: {info['count']}")
        print(f"    Hz: {hz:.2f}")
    
print(f"\n\n{'='*80}")
print("TOPICS REQUIRED BY CONFIG:")
print(f"{'='*80}")
from example_policies.data_ops.pipeline.frame_buffer import FrameBuffer
fb = FrameBuffer(cfg)
required_topics = fb.get_topic_names()

missing_count = 0
for topic in required_topics:
    if topic in topics:
        print(f"  ✓ FOUND: {topic}")
    else:
        print(f"  ✗ MISSING: {topic}")
        missing_count += 1

print(f"\n{'='*80}")
if missing_count > 0:
    print(f"⚠️  WARNING: {missing_count} required topics are MISSING from the MCAP file!")
    print("The conversion will fail because the MCAP file doesn't have the expected ROS topics.")
    print("\nYou need to either:")
    print("  1. Use a different MCAP file with the correct topics")
    print("  2. Modify the topic mappings in rosbag_topics.py to match your MCAP file")
else:
    print("✓ All required topics are present!")

# [markdown]
# --- 
# ## 3. Run Conversion
# 
# This cell executes the conversion process. It may take a while depending on the size of your data. You will see progress updates printed below.

#
from example_policies.data_ops.dataset_conversion import convert_episodes
from example_policies.data_ops.utils.conversion_utils import get_selected_episodes, get_sorted_episodes

episode_paths = get_sorted_episodes(RAW_DATA_DIR)
convert_episodes(episode_paths, OUTPUT_DIR, cfg)

# [markdown]
# --- 
# ## ✅ Done!
# 
# Your new dataset is ready at the output path you specified. You can now proceed to the next notebook to train a policy.


