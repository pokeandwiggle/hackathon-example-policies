"""
Visualize TCP topic repetition: sensor timestamps at 20Hz vs log timestamps at 100Hz.
"""

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from mcap.reader import make_reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_msg, register_types

# Register custom ROS message types
_CUSTOM_MSG_DEFS = {
    "teleop_controller_msgs/msg/PoseTwist": (
        "std_msgs/Header header\n"
        "geometry_msgs/Pose pose\n"
        "geometry_msgs/Twist twist\n"
    ),
}
for _name, _def in _CUSTOM_MSG_DEFS.items():
    try:
        register_types(get_types_from_msg(_def, _name))
    except Exception:
        pass

MCAP_DIR = "data/raw/v2"

# Possible topic name variants for each role
TCP_RIGHT_CANDIDATES = ["/arm_right/tcp_pose", "/panda_right/tcp"]
TCP_LEFT_CANDIDATES = ["/arm_left/tcp_pose", "/panda_left/tcp"]
DESIRED_RIGHT_CANDIDATES = ["/cartesian_target_right"]


def get_sensor_timestamp(msg):
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        s = msg.header.stamp
        return s.sec + s.nanosec * 1e-9
    return None


def extract_xyz(msg, schema_name):
    if "TransformStamped" in schema_name:
        t = msg.transform
        return np.array([t.translation.x, t.translation.y, t.translation.z])
    if schema_name.endswith("Transform") or "Transform" in schema_name and "Stamped" not in schema_name:
        return np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    if "PoseTwist" in schema_name:
        p = msg.pose
        return np.array([p.position.x, p.position.y, p.position.z])
    if "PoseStamped" in schema_name:
        p = msg.pose
        return np.array([p.position.x, p.position.y, p.position.z])
    return None


def extract_full_pose(msg, schema_name):
    """Extract full 7-dim pose: [x, y, z, qx, qy, qz, qw]."""
    if "TransformStamped" in schema_name:
        t = msg.transform
        return np.array([t.translation.x, t.translation.y, t.translation.z,
                         t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])
    if schema_name.endswith("Transform") or "Transform" in schema_name and "Stamped" not in schema_name:
        return np.array([msg.translation.x, msg.translation.y, msg.translation.z,
                         msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
    if "PoseTwist" in schema_name:
        p = msg.pose
        return np.array([p.position.x, p.position.y, p.position.z,
                         p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
    return None


def detect_topics(mcap_path):
    """Detect which topic names are present in this MCAP file."""
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        available = {ch.topic for ch in summary.channels.values()}

    tcp_right = next((t for t in TCP_RIGHT_CANDIDATES if t in available), None)
    tcp_left = next((t for t in TCP_LEFT_CANDIDATES if t in available), None)
    desired_right = next((t for t in DESIRED_RIGHT_CANDIDATES if t in available), None)

    topics = {}
    if tcp_right:
        topics[tcp_right] = "Actual TCP Right"
    if tcp_left:
        topics[tcp_left] = "Actual TCP Left"
    if desired_right:
        topics[desired_right] = "Desired TCP Right"
    return topics, tcp_right, tcp_left, desired_right


def load_data(mcap_path, topics):
    topic_data = defaultdict(lambda: {"log_t": [], "sensor_t": [], "xyz": [], "full_pose": []})
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=list(topics.keys())):
            decoded = deserialize_cdr(message.data, schema.name)
            st = get_sensor_timestamp(decoded)
            xyz = extract_xyz(decoded, schema.name)
            full = extract_full_pose(decoded, schema.name)
            if xyz is not None:
                topic_data[channel.topic]["log_t"].append(message.log_time / 1e9)
                topic_data[channel.topic]["sensor_t"].append(st)
                topic_data[channel.topic]["xyz"].append(xyz)
                if full is not None:
                    topic_data[channel.topic]["full_pose"].append(full)

    for topic in topic_data:
        topic_data[topic]["log_t"] = np.array(topic_data[topic]["log_t"])
        # sensor_t may contain None values (e.g. Transform msgs without header)
        st_list = topic_data[topic]["sensor_t"]
        if st_list and st_list[0] is not None:
            topic_data[topic]["sensor_t"] = np.array(st_list, dtype=float)
        else:
            topic_data[topic]["sensor_t"] = None
        topic_data[topic]["xyz"] = np.stack(topic_data[topic]["xyz"])
        if topic_data[topic]["full_pose"]:
            topic_data[topic]["full_pose"] = np.stack(topic_data[topic]["full_pose"])
    return topic_data


def plot_episode(mcap_path, mcap_file):
    topics, tcp_right, tcp_left, desired_right = detect_topics(mcap_path)
    if not tcp_right:
        print(f"  SKIP: no right TCP topic found")
        return

    data = load_data(mcap_path, topics)

    # Derive a readable episode name from filename
    episode_name = os.path.splitext(mcap_file)[0]

    # ── Limit to first 2 seconds for readability ──
    WINDOW_S = 2.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2.2, 3.5, 2]})
    fig.suptitle(
        f"TCP Pose Repetitions: 20 Hz Sensor vs 100 Hz Log\n{episode_name}",
        fontsize=15, fontweight="bold", y=0.97,
    )

    colors_actual = {"x": "#e74c3c", "y": "#2ecc71", "z": "#3498db"}

    # ── Prepare actual TCP data for delta panel ──
    d = data[tcp_right]
    t0 = d["log_t"][0]
    t_rel = d["log_t"] - t0
    mask = t_rel <= WINDOW_S
    t_plot = t_rel[mask] * 1000  # ms
    xyz = d["xyz"][mask]

    # ── Panel 1: Frame-to-frame delta of actual TCP (line plot) ──
    ax_delta = axes[0]
    # Compute per-step change magnitude
    deltas = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    t_delta = t_plot[1:]  # one fewer point than xyz

    # Line plot with markers — zero stretches are clearly flat at 0
    ax_delta.plot(t_delta, deltas, "-o", color="#2c3e50", markersize=3,
                  linewidth=1.0, zorder=3, label="‖Δxyz‖ per step")

    # Shade zero-delta (repeated) regions red
    is_repeat = deltas == 0
    i = 0
    while i < len(is_repeat):
        if is_repeat[i]:
            start = i
            while i < len(is_repeat) and is_repeat[i]:
                i += 1
            ax_delta.axvspan(t_delta[start] - 5, t_delta[min(i - 1, len(t_delta) - 1)] + 5,
                             color="#e74c3c", alpha=0.12, zorder=0)
        else:
            i += 1

    # Color the zero-value dots red, non-zero dots dark
    for idx in range(len(deltas)):
        if deltas[idx] == 0:
            ax_delta.plot(t_delta[idx], deltas[idx], "o", color="#e74c3c",
                          markersize=5, zorder=4)

    # Draw a prominent zero line
    ax_delta.axhline(0, color="#e74c3c", linewidth=0.8, linestyle="-", alpha=0.4, zorder=1)

    # Legend
    update_line = plt.Line2D([], [], color="#2c3e50", marker="o", markersize=4,
                             linewidth=1.0, label="Sensor updated (Δ > 0)")
    stale_dot = plt.Line2D([], [], color="#e74c3c", marker="o", markersize=5,
                           linewidth=0, label="Stale / repeated (Δ = 0)")
    ax_delta.legend(handles=[update_line, stale_dot], loc="upper right", fontsize=9)

    ax_delta.set_title(
        "Frame-to-Frame Position Change (‖Δxyz‖) — flat at zero for 4 consecutive stale steps",
        fontsize=11,
    )
    ax_delta.set_ylabel("|Δ position| (m)")
    ax_delta.set_xlim(0, WINDOW_S * 1000)
    ax_delta.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))

    # Count stats
    n_zero = int(np.sum(deltas == 0))
    n_total = len(deltas)
    pct = n_zero / n_total * 100
    ax_delta.annotate(
        f"{n_zero}/{n_total} steps ({pct:.0f}%) have Δ = 0 (stale)",
        xy=(0.5, 0.85), xycoords="axes fraction",
        fontsize=10, ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffeaa7", ec="#f39c12", alpha=0.9),
    )

    # ── Panel 2: Full state vector (both arms) — staircase view ──
    ax_state = axes[1]
    ZOOM_MS = 500  # tighter window to see steps clearly

    # Build full state: left arm (7) + right arm (7) = 14 dims, or just right arm (7) if left is missing
    d_right = data[tcp_right]
    has_left = tcp_left and tcp_left in data and len(data[tcp_left]["full_pose"]) > 0
    if has_left:
        d_left = data[tcp_left]
    t0_state = d_right["log_t"][0]
    t_rel_state = (d_right["log_t"] - t0_state) * 1000
    mask_state = t_rel_state <= ZOOM_MS
    t_state = t_rel_state[mask_state]

    pose_right = d_right["full_pose"][:len(t_state)]
    if has_left:
        pose_left = d_left["full_pose"][:len(t_state)]
        state_labels = [
            "L: x", "L: y", "L: z", "L: qx", "L: qy", "L: qz", "L: qw",
            "R: x", "R: y", "R: z", "R: qx", "R: qy", "R: qz", "R: qw",
        ]
        full_state = np.concatenate([pose_left, pose_right], axis=1)  # (N, 14)
    else:
        state_labels = ["R: x", "R: y", "R: z", "R: qx", "R: qy", "R: qz", "R: qw"]
        full_state = pose_right  # (N, 7)

    n_dims = full_state.shape[1]

    # Offset each dimension vertically for clarity
    cmap = plt.cm.tab20(np.linspace(0, 1, max(n_dims, 2)))
    for dim_i in range(n_dims):
        vals = full_state[:, dim_i]
        # Normalize to [0, 1] within this window
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin > 1e-10:
            normed = (vals - vmin) / (vmax - vmin)
        else:
            normed = np.zeros_like(vals)
        # Offset each dim so they don't overlap
        offset = dim_i * 1.1
        ax_state.plot(t_state, normed + offset, "-", color=cmap[dim_i],
                      linewidth=1.2, zorder=2)
        ax_state.plot(t_state, normed + offset, ".", color=cmap[dim_i],
                      markersize=3, zorder=3)
        # Label on right
        ax_state.text(ZOOM_MS + 5, normed[-1] + offset, state_labels[dim_i],
                      fontsize=6.5, ha="left", va="center", color=cmap[dim_i],
                      fontweight="bold")

    # Vertical dashed lines at sensor updates (from right arm)
    changed_state = np.zeros(len(t_state), dtype=bool)
    changed_state[0] = True
    for ii in range(1, len(t_state)):
        if not np.array_equal(full_state[ii], full_state[ii - 1]):
            changed_state[ii] = True
    for idx in np.where(changed_state)[0]:
        ax_state.axvline(t_state[idx], color="#333", linewidth=0.6,
                         linestyle="--", alpha=0.4, zorder=1)

    # Shade repeat groups
    grp = np.where(changed_state)[0]
    for gi, start in enumerate(grp):
        end = grp[gi + 1] if gi + 1 < len(grp) else len(t_state)
        ax_state.axvspan(t_state[start], t_state[min(end - 1, len(t_state) - 1)],
                         color=["#f9e9e9", "#e9f0f9"][gi % 2], alpha=0.45, zorder=0)

    ax_state.set_title(
        f"Full Observation State ({n_dims} dims) — first {ZOOM_MS} ms",
        fontsize=11,
    )
    ax_state.set_ylabel("Normalized value\n(offset per dim)")
    ax_state.set_xlabel("Time (ms)")
    ax_state.set_xlim(0, ZOOM_MS)
    ax_state.set_yticks([])

    # Annotate one plateau
    if len(t_state) > 2:
        ax_state.annotate(
            f"All {n_dims} state dims\nconstant for stale steps",
            xy=(t_state[2], n_dims * 1.1 / 2), xytext=(t_state[2] + 100, n_dims * 1.1 * 0.7),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa", alpha=0.9),
        )

    # ── Panel 3: Timing diagram ──
    ax3 = axes[2]
    ax3.set_title("Message Timing Comparison", fontsize=11)

    # Actual TCP: sensor ticks and log ticks
    d = data[tcp_right]
    t0 = d["log_t"][0]
    t_rel = d["log_t"] - t0
    mask = t_rel <= WINDOW_S
    log_times = t_rel[mask] * 1000

    # Log ticks (100Hz) — small markers
    ax3.scatter(log_times, np.full_like(log_times, 2.0), marker="|", s=40,
                color="#e74c3c", alpha=0.6, linewidths=0.8, zorder=2)
    ax3.text(-30, 2.0, "Actual TCP\nlog msgs\n(100 Hz)", fontsize=8, ha="right", va="center",
             color="#e74c3c", fontweight="bold")

    # Sensor ticks (20Hz) — larger markers
    sensor_ts = d["sensor_t"]
    has_sensor_ts = sensor_ts is not None
    if has_sensor_ts:
        sensor_times_raw = sensor_ts[mask]
        sensor_t0 = sensor_times_raw[0]
        sensor_rel = (sensor_times_raw - sensor_t0) * 1000
        unique_sensor = np.unique(sensor_rel)
        ax3.scatter(unique_sensor, np.full_like(unique_sensor, 1.5), marker="|", s=100,
                    color="#2c3e50", linewidths=1.5, zorder=3)
        ax3.text(-30, 1.5, "Actual TCP\nsensor updates", fontsize=8, ha="right", va="center",
                 color="#2c3e50", fontweight="bold")

    # Desired TCP: log ticks (30Hz, all unique)
    if desired_right and desired_right in data:
        d_des = data[desired_right]
        t0d = d_des["log_t"][0]
        t_rel_d = d_des["log_t"] - t0d
        mask_d = t_rel_d <= WINDOW_S
        log_times_d = t_rel_d[mask_d] * 1000

        ax3.scatter(log_times_d, np.full_like(log_times_d, 0.8), marker="|", s=60,
                    color="#2980b9", linewidths=1.0, zorder=2)
        ax3.text(-30, 0.8, "Desired TCP\nlog msgs", fontsize=8, ha="right", va="center",
                 color="#2980b9", fontweight="bold")

    ax3.set_ylim(0.3, 2.5)
    ax3.set_xlim(0, WINDOW_S * 1000)
    ax3.set_yticks([])
    ax3.set_xlabel("Time (ms)")

    # Highlight a 50ms window to show the 5-repeat pattern
    ax3.axvspan(0, 50, color="#ffeaa7", alpha=0.4, zorder=0)
    ax3.annotate(
        "5 log msgs per\n1 sensor update",
        xy=(25, 1.75), xytext=(150, 2.3),
        fontsize=9, ha="center",
        arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffeaa7", ec="#f39c12", alpha=0.9),
    )

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0.08, 0, 1, 0.95])

    out_dir = "scripts/tcp_repetition_plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{episode_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def main():
    mcap_files = sorted(f for f in os.listdir(MCAP_DIR) if f.endswith(".mcap"))
    print(f"Found {len(mcap_files)} MCAP files in {MCAP_DIR}")
    for mcap_file in mcap_files:
        mcap_path = os.path.join(MCAP_DIR, mcap_file)
        print(f"\nProcessing: {mcap_file}")
        try:
            plot_episode(mcap_path, mcap_file)
        except Exception as e:
            print(f"  ERROR processing {mcap_file}: {e}")


if __name__ == "__main__":
    main()
