import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
})

COLORS = {
    "baseline": "#1D9E75",
    "dr": "#534AB7",
}

def load_run(log_dir, label, color):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = (steps, values)
    return {"label": label, "color": color, "data": data}


def smooth(values, weight=0.85):
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


def plot_tag(ax, runs, tag, title, ylabel, smoothing=0.85):
    for run in runs:
        if tag not in run["data"]:
            continue
        steps, values = run["data"][tag]
        ax.plot(steps, smooth(values, smoothing), color=run["color"],
                linewidth=2, label=run["label"])
        ax.fill_between(steps,
                        smooth(values, smoothing),
                        alpha=0.08, color=run["color"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Iteration")
    ax.legend(fontsize=9)


def save_single(runs, tag, title, ylabel, filename, out_dir, smoothing=0.85):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_tag(ax, runs, tag, title, ylabel, smoothing)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def make_overview_figure(runs, out_dir):
    individual = [
        ("Train/mean_reward",                   "Mean Reward",               "Reward",        "mean_reward.png"),
        ("Train/mean_episode_length",            "Episode Length",            "Steps",         "episode_length.png"),
        ("Curriculum/terrain_levels",            "Terrain Level (Curriculum)","Level",         "terrain_level.png"),
        ("Episode_Reward/track_lin_vel_xy_exp",  "Linear Velocity Tracking",  "Reward",        "vel_tracking.png"),
        ("Metrics/base_velocity/error_vel_xy",   "Velocity Tracking Error",   "Error (m/s)",   "vel_error.png"),
        ("Episode_Termination/base_contact",     "Fall Rate",                 "Fraction",      "fall_rate.png"),
    ]
    for tag, title, ylabel, fname in individual:
        save_single(runs, tag, title, ylabel, fname, out_dir)

    # also save combined overview
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Anymal C Rough Terrain — PPO Training", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    positions = [gs[0,0], gs[0,1], gs[0,2], gs[1,0], gs[1,1], gs[1,2]]
    for (tag, title, ylabel, _), pos in zip(individual, positions):
        ax = fig.add_subplot(pos)
        plot_tag(ax, runs, tag, title, ylabel)
    plt.savefig(os.path.join(out_dir, "overview.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved overview.png")


def make_reward_breakdown(runs, out_dir):
    reward_tags = [
        ("Episode_Reward/track_lin_vel_xy_exp", "Lin Vel Tracking (XY)",  "reward_lin_vel_xy.png"),
        ("Episode_Reward/track_ang_vel_z_exp",  "Ang Vel Tracking (Z)",   "reward_ang_vel_z.png"),
        ("Episode_Reward/lin_vel_z_l2",         "Vertical Vel Penalty",   "reward_lin_vel_z.png"),
        ("Episode_Reward/ang_vel_xy_l2",        "Ang Vel XY Penalty",     "reward_ang_vel_xy.png"),
        ("Episode_Reward/dof_torques_l2",       "Torque Penalty",         "reward_torques.png"),
        ("Episode_Reward/dof_acc_l2",           "Acceleration Penalty",   "reward_acc.png"),
        ("Episode_Reward/action_rate_l2",       "Action Rate Penalty",    "reward_action_rate.png"),
        ("Episode_Reward/feet_air_time",        "Feet Air Time",          "reward_feet_air.png"),
        ("Episode_Reward/undesired_contacts",   "Undesired Contacts",     "reward_contacts.png"),
    ]
    for tag, title, fname in reward_tags:
        save_single(runs, tag, title, "Reward", fname, out_dir, smoothing=0.9)

    # also save combined grid
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle("Reward Term Breakdown", fontsize=14, fontweight="bold", y=0.99)
    for i, (tag, title, _) in enumerate(reward_tags):
        plot_tag(axes.flatten()[i], runs, tag, title, "Reward", smoothing=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved reward_breakdown.png")


def make_loss_figure(runs, out_dir):
    loss_tags = [
        ("Loss/value_function",   "Value Function Loss", "loss_value.png"),
        ("Loss/surrogate",        "Surrogate Loss (PPO)","loss_surrogate.png"),
        ("Loss/entropy",          "Entropy Loss",        "loss_entropy.png"),
        ("Loss/learning_rate",    "Learning Rate",       "learning_rate.png"),
        ("Policy/mean_noise_std", "Action Noise Std",    "noise_std.png"),
        ("Perf/total_fps",        "Simulation FPS",      "fps.png"),
    ]
    for tag, title, fname in loss_tags:
        save_single(runs, tag, title, "", fname, out_dir, smoothing=0.8)

    # also save combined grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Training Diagnostics", fontsize=14, fontweight="bold", y=1.01)
    for i, (tag, title, _) in enumerate(loss_tags):
        plot_tag(axes.flatten()[i], runs, tag, title, "", smoothing=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved diagnostics.png")


def make_summary_table(runs, out_dir):
    metrics = {
        "Final Mean Reward":        "Train/mean_reward",
        "Final Episode Length":     "Train/mean_episode_length",
        "Final Terrain Level":      "Curriculum/terrain_levels",
        "Final Vel Error XY":       "Metrics/base_velocity/error_vel_xy",
        "Final Fall Rate":          "Episode_Termination/base_contact",
        "Final Velocity Tracking":  "Episode_Reward/track_lin_vel_xy_exp",
    }

    fig, ax = plt.subplots(figsize=(8, len(metrics) * 0.7 + 1.5))
    ax.axis("off")

    col_labels = ["Metric"] + [r["label"] for r in runs]
    table_data = []

    for metric_name, tag in metrics.items():
        row = [metric_name]
        for run in runs:
            if tag in run["data"]:
                last_val = run["data"][tag][1][-1]
                row.append(f"{last_val:.4f}")
            else:
                row.append("N/A")
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C2C2A")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_facecolor("#F1EFE8")
        elif col == 1:
            cell.set_facecolor("#E1F5EE")
        elif col == 2:
            cell.set_facecolor("#EEEDFE")
        cell.set_edgecolor("white")

    fig.suptitle("Final Metrics Summary", fontsize=13, fontweight="bold")
    plt.savefig(os.path.join(out_dir, "summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved summary_table.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to baseline log dir")
    parser.add_argument("--dr", default=None, help="Path to DR experiment log dir (optional)")
    parser.add_argument("--out", default="plots", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    runs = [load_run(args.baseline, "Baseline", COLORS["baseline"])]
    if args.dr:
        runs.append(load_run(args.dr, "Extended DR", COLORS["dr"]))

    make_overview_figure(runs, args.out)
    make_reward_breakdown(runs, args.out)
    make_loss_figure(runs, args.out)
    make_summary_table(runs, args.out)

    print(f"\nAll plots saved to: {args.out}/")


if __name__ == "__main__":
    main()
