"""
Analyze TCP topic data for repeated values.

Hypothesis: sensor timestamps are at 20Hz, log timestamps at 100Hz,
so each unique value is repeated ~5 times.
"""

import os
from collections import Counter, defaultdict

from mcap.reader import make_reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_msg, register_types

# Register custom ROS message types (same as the project does)
_CUSTOM_MSG_DEFS = {
    "teleop_controller_msgs/msg/PoseTwist": (
        "std_msgs/Header header\n"
        "geometry_msgs/Pose pose\n"
        "geometry_msgs/Twist twist\n"
    ),
    "teleop_controller_msgs/msg/GripperValues": (
        "std_msgs/Header header\n"
        "float64 width\n"
        "float64 speed\n"
        "float64 force\n"
    ),
}
for _name, _def in _CUSTOM_MSG_DEFS.items():
    try:
        register_types(get_types_from_msg(_def, _name))
    except Exception:
        pass

MCAP_DIR = "data/raw/v2"

# TCP-related topics to analyze
TCP_TOPICS = [
    "/arm_left/tcp_pose",
    "/arm_right/tcp_pose",
    "/cartesian_target_left",
    "/cartesian_target_right",
    "/cartesian_waypoint_left",
    "/cartesian_waypoint_right",
]


def extract_pose_values(msg, schema_name):
    """Extract numeric values from a decoded ROS2 message as a tuple."""
    # TransformStamped (actual TCP poses)
    if "TransformStamped" in schema_name:
        t = msg.transform
        return (
            t.translation.x, t.translation.y, t.translation.z,
            t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w,
        )
    # PoseStamped (waypoints)
    if "PoseStamped" in schema_name:
        p = msg.pose
        return (
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
        )
    # PoseTwist (cartesian targets)
    if "PoseTwist" in schema_name:
        p = msg.pose
        tw = msg.twist
        return (
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
            tw.linear.x, tw.linear.y, tw.linear.z,
            tw.angular.x, tw.angular.y, tw.angular.z,
        )
    return None


def get_sensor_timestamp(msg, schema_name):
    """Extract sensor timestamp in seconds from header/stamp."""
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        s = msg.header.stamp
        return s.sec + s.nanosec * 1e-9
    if hasattr(msg, "stamp"):
        s = msg.stamp
        return s.sec + s.nanosec * 1e-9
    return None


def analyze_file(filepath):
    print(f"\n{'='*80}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"{'='*80}")

    # Collect: topic -> [(log_time_s, sensor_time_s, values_tuple), ...]
    topic_data = defaultdict(list)
    topic_schemas = {}

    with open(filepath, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        schema_map = {}
        if summary:
            for sid, s in summary.schemas.items():
                schema_map[sid] = s.name
            channel_map = {}
            for cid, ch in summary.channels.items():
                channel_map[cid] = (ch.topic, schema_map.get(ch.schema_id, "unknown"))

    with open(filepath, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=TCP_TOPICS):
            schema_name = schema.name
            topic_schemas[channel.topic] = schema_name
            try:
                decoded = deserialize_cdr(message.data, schema_name)
            except Exception as e:
                print(f"  WARN: Could not decode {channel.topic}: {e}")
                continue

            sensor_ts = get_sensor_timestamp(decoded, schema_name)
            values = extract_pose_values(decoded, schema_name)

            topic_data[channel.topic].append((
                message.log_time / 1e9,  # log time in seconds
                sensor_ts,               # sensor time in seconds (or None)
                values,
            ))

    for topic in sorted(topic_data.keys()):
        entries = topic_data[topic]
        n = len(entries)
        if n == 0:
            continue

        print(f"\n--- {topic} ({n} messages) [{topic_schemas.get(topic, '?')}] ---")

        # Log-time frequency
        if n > 1:
            dt_log = [entries[i+1][0] - entries[i][0] for i in range(n-1)]
            avg_dt_log = sum(dt_log) / len(dt_log)
            log_hz = 1.0 / avg_dt_log if avg_dt_log > 0 else 0
            print(f"  Log-time frequency:    {log_hz:.1f} Hz (avg dt={avg_dt_log*1000:.2f} ms)")

        # Sensor-time frequency (unique stamps only)
        sensor_times = [e[1] for e in entries if e[1] is not None]
        if len(sensor_times) > 1:
            unique_sensor_times = sorted(set(sensor_times))
            dt_sensor = [unique_sensor_times[i+1] - unique_sensor_times[i]
                         for i in range(len(unique_sensor_times)-1)]
            avg_dt_sensor = sum(dt_sensor) / len(dt_sensor)
            sensor_hz = 1.0 / avg_dt_sensor if avg_dt_sensor > 0 else 0
            print(f"  Sensor-time frequency: {sensor_hz:.1f} Hz (avg dt={avg_dt_sensor*1000:.2f} ms)")
            print(f"  Unique sensor timestamps: {len(unique_sensor_times)} / {len(sensor_times)} total")

            # Are sensor timestamps also repeating?
            sensor_repeat_counts = []
            cur_st = sensor_times[0]
            cur_cnt = 1
            for i in range(1, len(sensor_times)):
                if sensor_times[i] == cur_st:
                    cur_cnt += 1
                else:
                    sensor_repeat_counts.append(cur_cnt)
                    cur_st = sensor_times[i]
                    cur_cnt = 1
            sensor_repeat_counts.append(cur_cnt)
            st_dist = Counter(sensor_repeat_counts)
            print(f"  Sensor timestamp repeat distribution: {dict(sorted(st_dist.items()))}")

        # Count consecutive repeats of pose values
        if entries[0][2] is not None:
            repeat_counts = []
            current_val = entries[0][2]
            current_count = 1
            for i in range(1, n):
                if entries[i][2] == current_val:
                    current_count += 1
                else:
                    repeat_counts.append(current_count)
                    current_val = entries[i][2]
                    current_count = 1
            repeat_counts.append(current_count)

            unique_values = len(repeat_counts)
            avg_repeat = sum(repeat_counts) / len(repeat_counts)
            min_repeat = min(repeat_counts)
            max_repeat = max(repeat_counts)

            repeat_dist = Counter(repeat_counts)

            print(f"  Unique consecutive value groups: {unique_values}")
            print(f"  Value repeat count: avg={avg_repeat:.2f}, min={min_repeat}, max={max_repeat}")
            print(f"  Value repeat distribution: {dict(sorted(repeat_dist.items()))}")

            # Show first 25 entries
            print(f"\n  First 25 messages:")
            print(f"  {'idx':>5} {'log_off_ms':>10} {'sensor_off_ms':>13} {'chg':>3}  xyz (first 3 values)")
            t0_log = entries[0][0]
            t0_sensor = entries[0][1] if entries[0][1] is not None else 0
            prev_val = None
            for i, (lt, st, val) in enumerate(entries[:25]):
                lt_off = (lt - t0_log) * 1000
                st_off = ((st - t0_sensor) * 1000) if st is not None else None
                changed = "NEW" if val != prev_val else "  ="
                val_short = f"({val[0]:.6f}, {val[1]:.6f}, {val[2]:.6f})" if val else "None"
                st_str = f"{st_off:10.2f}" if st_off is not None else "       None"
                print(f"  {i:5d} {lt_off:10.2f} {st_str:>13} {changed}  {val_short}")
                prev_val = val
        else:
            print("  (Could not extract pose values)")


if __name__ == "__main__":
    for fname in sorted(os.listdir(MCAP_DIR)):
        if fname.endswith(".mcap"):
            analyze_file(os.path.join(MCAP_DIR, fname))
