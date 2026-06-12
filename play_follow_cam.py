# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import omni.appwindow
import omni.usd
from pxr import Gf, UsdGeom
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    try: env_cfg.commands.base_velocity.debug_vis = False
    except: pass
    try: env_cfg.terminations.time_out = None
    except: pass
    try: env_cfg.terminations.base_contact = None
    except: pass

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )


    # Read velocity commands from file written by controller script
    import json as _json
    _cmd_file = "/tmp/robot_cmd.json"
    with open(_cmd_file, 'w') as _f:
        _json.dump({"vx": 0.0, "vy": 0.0, "wz": 0.0}, _f)
    print("[INFO] Run controller: python3 scripts/reinforcement_learning/rsl_rl/robot_controller.py")

    stage = omni.usd.get_context().get_stage()
    cam_path = "/World/FollowCam"
    UsdGeom.Camera.Define(stage, cam_path).GetFocalLengthAttr().Set(18.0)
    try:
        import omni.kit.viewport.utility as vp_util
        vp = vp_util.get_active_viewport()
        if vp: vp.camera_path = cam_path
        print("[INFO] Follow camera active")
    except Exception as e:
        print(f"[WARNING] {e}")
    # --- Camera Control UI ---
    import omni.ui as ui
    cam_settings = {"dist": 2.5, "height": 0.9, "target_z": 0.3}
    _win = ui.Window("Camera Controls", width=300, height=160)
    with _win.frame:
        with ui.VStack():
            ui.Label("Camera Distance")
            dist_slider = ui.FloatSlider(min=0.5, max=6.0, step=0.1)
            dist_slider.model.set_value(cam_settings["dist"])
            def on_dist(m): cam_settings["dist"] = m.get_value_as_float()
            dist_slider.model.add_value_changed_fn(on_dist)
            ui.Label("Camera Height")
            height_slider = ui.FloatSlider(min=0.0, max=3.0, step=0.1)
            height_slider.model.set_value(cam_settings["height"])
            def on_height(m): cam_settings["height"] = m.get_value_as_float()
            height_slider.model.add_value_changed_fn(on_height)
            ui.Label("Target Height")
            tgt_slider = ui.FloatSlider(min=0.0, max=2.0, step=0.1)
            tgt_slider.model.set_value(cam_settings["target_z"])
            def on_tgt(m): cam_settings["target_z"] = m.get_value_as_float()
            tgt_slider.model.add_value_changed_fn(on_tgt)
    # --- End Camera Control UI ---
    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        # read velocity command from controller file
        try:
            with open("/tmp/robot_cmd.json") as _f:
                _cmd = _json.load(_f)
            vx, vy, wz = _cmd["vx"], _cmd["vy"], _cmd["wz"]
            if _cmd.get("reset", False):
                print("[INFO] Resetting environment")
                obs, _ = env.reset()
                with open("/tmp/robot_cmd.json", "w") as _f:
                    _json.dump({"vx": 0.0, "vy": 0.0, "wz": 0.0, "reset": False}, _f)
                continue
        except Exception:
            vx, vy, wz = 0.0, 0.0, 0.0
        try:
            cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
            cmd_term.vel_command_b[0, 0] = vx
            cmd_term.vel_command_b[0, 1] = vy
            # heading command mode: increment heading target
            import math as _math
            cmd_term.is_heading_env[0] = True
            cmd_term.heading_target[0] += wz * 0.02
            if abs(wz) > 0.1: print(f"[DEBUG2] is_heading_env={cmd_term.is_heading_env[0]} heading_target={cmd_term.heading_target[0]:.2f} robot_heading={cmd_term.robot.data.heading_w[0]:.2f} vel_cmd={cmd_term.vel_command_b[0]}")
            cmd_term.heading_target[0] = float(cmd_term.heading_target[0]) % (2 * _math.pi)
            cmd_term.time_left[0] = 1000.0
            if abs(wz) > 0.1: print(f"[DEBUG] heading={cmd_term.heading_target[0]:.2f} wz={wz}")
        except Exception as e:
            print(f"[DEBUG ERROR] {e}")
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        try:
            robot = env.unwrapped.scene["robot"]
            pos_t = robot.data.root_pos_w[0]
            quat_t = robot.data.root_quat_w[0]
            px, py, pz = float(pos_t[0]), float(pos_t[1]), float(pos_t[2])
            w,x,y,z = float(quat_t[0]),float(quat_t[1]),float(quat_t[2]),float(quat_t[3])
            fx = 1-2*(y*y+z*z); fy = 2*(x*y+w*z); fz = 2*(x*z-w*y)
            norm = (fx*fx+fy*fy+fz*fz)**0.5 + 1e-8
            fx,fy,fz = fx/norm, fy/norm, fz/norm
            cp = Gf.Vec3d(px-fx*cam_settings["dist"], py-fy*cam_settings["dist"], pz+cam_settings["height"])
            tgt = Gf.Vec3d(px, py, pz+cam_settings["target_z"])
            cxf = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path))
            cxf.ClearXformOpOrder(); cxf.AddTranslateOp().Set(cp)
            lk=(tgt-cp).GetNormalized(); up=Gf.Vec3d(0,0,1)
            ri=Gf.Cross(lk,up).GetNormalized(); u2=Gf.Cross(ri,lk).GetNormalized()
            m4=Gf.Matrix4d(ri[0],ri[1],ri[2],0,u2[0],u2[1],u2[2],0,-lk[0],-lk[1],-lk[2],0,0,0,0,1)
            q=m4.ExtractRotationQuat()
            cxf.AddOrientOp().Set(Gf.Quatf(float(q.GetReal()),float(q.GetImaginary()[0]),float(q.GetImaginary()[1]),float(q.GetImaginary()[2])))
        except Exception:
            pass

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
