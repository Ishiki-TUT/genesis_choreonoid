#!/usr/bin/env python3
"""HRP2 walking control example using vnoid_python.

This script builds a small alternating footstep plan, feeds it to
vnoid_python.SteppingController, and converts the resulting foot references
into Choreonoid IK targets when a robot model is available.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VNOID_ROOT = os.path.join(PROJECT_ROOT, "userdir", "vnoid")
if VNOID_ROOT not in sys.path:
    sys.path.insert(0, VNOID_ROOT)

try:
    exec(open("/choreonoid_ws/install/share/irsl_choreonoid/sample/irsl_import.py").read())
except Exception:
    pass

from vnoid_python.footstep_planner import Footstep, FootstepPlanner, Param, Step
from vnoid_python.stepping_controller import Base, Centroid, Foot, SteppingController, Timer


ANKLE_HEIGHT_OFFSET = 0.105


def build_footstep_plan(
    stride: float,
    spacing: float,
    com_height: float,
    step_duration: float,
    time_constant: float,
    num_steps: int = 5,
) -> tuple[Footstep, Param]:
    """Build a simple alternating left/right walking plan."""

    param = Param(com_height=com_height, T=time_constant)
    steps: list[Step] = []

    left_x = 0.0
    right_x = 0.0
    left_y = spacing * 0.5
    right_y = -spacing * 0.5

    for index in range(num_steps):
        step = Step()
        step.side = 0 if index % 2 == 0 else 1
        step.duration = step_duration
        step.stride = stride
        step.spacing = spacing
        step.turn = 0.0
        step.sway = 0.0
        step.climb = 0.0
        step.stepping = index != 0

        if index == 0:
            left_x = 0.0
            right_x = 0.0
        elif index % 2 == 1:
            right_x += stride
        else:
            left_x += stride

        step.foot_pos[0] = np.array([left_x, left_y, 0.0], dtype=np.float64)
        step.foot_pos[1] = np.array([right_x, right_y, 0.0], dtype=np.float64)
        step.foot_angle[0] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        step.foot_angle[1] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        step.foot_ori[0] = R.from_euler("xyz", step.foot_angle[0])
        step.foot_ori[1] = R.from_euler("xyz", step.foot_angle[1])

        center = (step.foot_pos[0] + step.foot_pos[1]) * 0.5
        step.zmp = center.copy()
        step.dcm = center + np.array([0.0, 0.0, com_height], dtype=np.float64)
        steps.append(step)

    footstep = Footstep(steps=steps)
    planner = FootstepPlanner()
    planner.generate_dcm(param, footstep)

    # Keep the initial stance as a double-support phase.
    footstep.steps[0].stepping = False

    return footstep, param


@dataclass
class WalkingContext:
    footstep: Footstep
    footstep_buffer: Footstep
    param: Param
    controller: SteppingController
    timer: Timer
    centroid: Centroid
    base: Base
    feet: list[Foot]
    com_pos: np.ndarray


def create_context(
    stride: float,
    spacing: float,
    com_height: float,
    step_duration: float,
    time_constant: float,
) -> WalkingContext:
    footstep, param = build_footstep_plan(
        stride=stride,
        spacing=spacing,
        com_height=com_height,
        step_duration=step_duration,
        time_constant=time_constant,
    )

    controller = SteppingController()
    timer = Timer(dt=0.0)
    centroid = Centroid()
    base = Base()
    feet = [Foot(), Foot()]
    footstep_buffer = Footstep(steps=[Step(), Step()])

    initial_step = footstep.steps[0]
    feet[0].pos_ref = initial_step.foot_pos[0].copy()
    feet[1].pos_ref = initial_step.foot_pos[1].copy()
    feet[0].angle_ref = initial_step.foot_angle[0].copy()
    feet[1].angle_ref = initial_step.foot_angle[1].copy()
    feet[0].ori_ref = R.from_euler("xyz", feet[0].angle_ref)
    feet[1].ori_ref = R.from_euler("xyz", feet[1].angle_ref)
    feet[0].contact_ref = True
    feet[1].contact_ref = True

    centroid.zmp_ref = initial_step.zmp.copy()
    centroid.zmp_target = initial_step.zmp.copy()
    centroid.dcm_ref = initial_step.dcm.copy()
    centroid.dcm_target = initial_step.dcm.copy()
    centroid.com_pos_ref = initial_step.dcm.copy()
    com_pos = initial_step.dcm.copy()

    return WalkingContext(
        footstep=footstep,
        footstep_buffer=footstep_buffer,
        param=param,
        controller=controller,
        timer=timer,
        centroid=centroid,
        base=base,
        feet=feet,
        com_pos=com_pos,
    )


def foot_to_coords(foot: Foot):
    """Convert a foot reference to a Choreonoid coordinates object."""

    target = coordinates(fv(foot.pos_ref[0], foot.pos_ref[1], foot.pos_ref[2] + ANKLE_HEIGHT_OFFSET))
    target.rotation = foot.ori_ref.as_matrix()
    return target


def update_robot_from_feet(srobot, context: WalkingContext) -> None:
    """Apply the current foot references to the robot model via IK."""

    if srobot is None:
        return

    try:
        srobot.rootLink.p = fv(context.com_pos[0], context.com_pos[1], context.param.com_height)
    except Exception:
        pass

    try:
        srobot.lleg.inverseKinematics(foot_to_coords(context.feet[0]))
    except Exception:
        pass

    try:
        srobot.rleg.inverseKinematics(foot_to_coords(context.feet[1]))
    except Exception:
        pass


def run_walking_demo(
    srobot=None,
    duration: float = 10.0,
    dt: float = 0.01,
    stride: float = 0.10,
    spacing: float = 0.20,
    com_height: float = 0.71,
    step_duration: float = 0.80,
    time_constant: float = 0.40,
):
    """Run the walking controller loop."""

    context = create_context(
        stride=stride,
        spacing=spacing,
        com_height=com_height,
        step_duration=step_duration,
        time_constant=time_constant,
    )

    context.timer.dt = dt
    current_time = 0.0
    total_time = min(duration, len(context.footstep.steps) * step_duration)

    print("Walking simulation with vnoid started...")
    print(f"  stride={stride:.3f} m, spacing={spacing:.3f} m, com_height={com_height:.3f} m")
    print(f"  step_duration={step_duration:.3f} s, T={time_constant:.3f} s, dt={dt:.4f} s")

    while current_time < total_time:
        context.timer.time = current_time

        # Simple CoM integration using the current DCM target.
        com_dot = (context.centroid.dcm_target - context.com_pos) / context.param.T
        context.com_pos = context.com_pos + com_dot * dt
        context.centroid.dcm_ref = context.com_pos.copy()
        context.centroid.zmp_ref = context.centroid.zmp_target.copy()
        context.centroid.com_pos_ref = context.com_pos.copy()

        active = context.controller.update(
            context.timer,
            context.param,
            context.footstep,
            context.footstep_buffer,
            context.centroid,
            context.base,
            context.feet,
        )

        if not active:
            print(f"Controller finished at t={current_time:.3f} s")
            break

        update_robot_from_feet(srobot, context)

        current_time += dt

    print("Walking simulation completed!")
    return context


def load_cnoid_robot(robot_urdf_path: str, robot_name: str = "CnoidRobot"):
    """Load an HRP2 robot into Choreonoid when the API is available."""

    if "RobotModel" not in globals():
        return None

    try:
        if "cutil" in globals():
            try:
                ib.loadRobotItem(cutil.getShareDirectory() + "/model/misc/floor.body")
            except Exception:
                pass
        return RobotModel.loadModelItem(robot_urdf_path, world=True, name=robot_name)
    except Exception as exc:
        print(f"Warning: could not load robot model: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="HRP2 walking control example using vnoid_python")
    parser.add_argument("--urdf", type=str, default=os.path.join(os.path.dirname(__file__), "hrp2_description", "HRP2_genesis.urdf"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--stride", type=float, default=0.10)
    parser.add_argument("--spacing", type=float, default=0.20)
    parser.add_argument("--com-height", type=float, default=0.71)
    parser.add_argument("--step-duration", type=float, default=0.80)
    parser.add_argument("--T", type=float, default=0.40)
    args = parser.parse_args()

    srobot = load_cnoid_robot(args.urdf)
    if srobot is not None:
        try:
            srobot.setAngleMap(
                {
                    "RLEG_JOINT0": 0.0,
                    "RLEG_JOINT1": 0.0,
                    "RLEG_JOINT2": -0.4,
                    "RLEG_JOINT3": 0.8,
                    "RLEG_JOINT4": -0.4,
                    "RLEG_JOINT5": 0.0,
                    "LLEG_JOINT0": 0.0,
                    "LLEG_JOINT1": 0.0,
                    "LLEG_JOINT2": -0.4,
                    "LLEG_JOINT3": 0.8,
                    "LLEG_JOINT4": -0.4,
                    "LLEG_JOINT5": 0.0,
                }
            )
            if hasattr(srobot, "moveCentroidOnFoot"):
                srobot.moveCentroidOnFoot()
        except Exception:
            pass

    run_walking_demo(
        srobot=srobot,
        duration=args.duration,
        dt=args.dt,
        stride=args.stride,
        spacing=args.spacing,
        com_height=args.com_height,
        step_duration=args.step_duration,
        time_constant=args.T,
    )


if __name__ == "__main__":
    main()