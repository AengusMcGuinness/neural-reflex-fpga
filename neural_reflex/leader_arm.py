"""Inspect and cautiously jog a single Hiwonder SO-ARM101 leader arm.

The black leader arm is normally a passive teleoperator. This module uses its
existing six serial bus servos as an experimental output device. It deliberately
does not claim Cartesian end-effector control or use the follower-arm gripper API.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from glob import glob
import select
import sys
import time


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
POSITION_MODE = 0
RAW_MAX = 4095


@dataclass(frozen=True)
class JointState:
    position: int
    minimum: int
    maximum: int
    torque_enabled: bool
    operating_mode: int


def make_bus(port: str):
    """Use Hiwonder's own LeRobot serial driver; import it only when needed."""
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.hiwonder import HiwonderMotorsBus
    except ImportError as error:
        raise RuntimeError("Install the local Hiwonder driver as shown in ARM_SETUP.md") from error

    # Hiwonder's fork uses the hx30hm protocol profile for *both* SO101 roles.
    motors = {
        name: Motor(index, "hx30hm", MotorNormMode.DEGREES)
        for index, name in enumerate(JOINTS, 1)
    }
    return HiwonderMotorsBus(port=port, motors=motors)


def inspect(bus) -> dict[str, JointState]:
    """Read the six joints without changing torque, IDs, calibration, or goals."""
    states = {}
    for name in JOINTS:
        raw = lambda register: int(bus.read(register, name, normalize=False))
        states[name] = JointState(
            position=raw("Present_Position"),
            minimum=raw("Min_Position_Limit"),
            maximum=raw("Max_Position_Limit"),
            torque_enabled=bool(raw("Torque_Enable")),
            operating_mode=raw("Operating_Mode"),
        )
    return states


def validate_motion_state(states: dict[str, JointState]) -> None:
    if set(states) != set(JOINTS):
        raise ValueError("expected all six leader-arm joints")
    for name, state in states.items():
        if state.operating_mode != POSITION_MODE:
            raise ValueError(f"{name} is not in position mode")
        if state.torque_enabled:
            raise ValueError(f"{name} already has torque enabled; inspect the arm first")
        if not (0 <= state.minimum < state.maximum <= RAW_MAX):
            raise ValueError(f"{name} has invalid position limits")
        if not (state.minimum <= state.position <= state.maximum):
            raise ValueError(f"{name} is outside its configured position limits")


def disable_all_torque(bus) -> dict[str, str]:
    """Attempt every motor even if one reports a fault during torque-off."""
    errors = {}
    for name in JOINTS:
        try:
            bus.disable_torque(name)
        except Exception as error:
            errors[name] = str(error).splitlines()[0]
    return errors


class JogSession:
    """Software-bounded, one-step-at-a-time joint motion."""

    def __init__(self, bus, states: dict[str, JointState], max_step: int = 10,
                 max_travel: int = 80):
        validate_motion_state(states)
        if not (1 <= max_step <= 20):
            raise ValueError("max_step must be 1..20 raw ticks")
        if not (1 <= max_travel <= 200):
            raise ValueError("max_travel must be 1..200 raw ticks")
        self.bus = bus
        self.states = states
        self.max_step = max_step
        self.max_travel = max_travel
        self.goals = {name: state.position for name, state in states.items()}
        self.armed = False

    def arm(self) -> None:
        if self.armed:
            raise RuntimeError("already armed")
        # Match all goals to measured positions before enabling torque.
        for name, goal in self.goals.items():
            self.bus.write("Goal_Position", name, goal, normalize=False)
        try:
            self.bus.enable_torque()
        except BaseException:
            disable_all_torque(self.bus)
            raise
        self.armed = True

    def jog(self, name: str, delta: int) -> int:
        if not self.armed:
            raise RuntimeError("motion is not armed")
        if name not in self.states:
            raise ValueError(f"unknown joint: {name}")
        if not isinstance(delta, int) or delta == 0 or abs(delta) > self.max_step:
            raise ValueError(f"step must be a nonzero integer within ±{self.max_step} ticks")
        state = self.states[name]
        target = self.goals[name] + delta
        lower = max(state.minimum, state.position - self.max_travel)
        upper = min(state.maximum, state.position + self.max_travel)
        if not lower <= target <= upper:
            raise ValueError(f"{name} target {target} exceeds the session bounds {lower}..{upper}")
        self.bus.write("Goal_Position", name, target, normalize=False)
        self.goals[name] = target
        return target

    def stop(self) -> None:
        if self.armed:
            errors = disable_all_torque(self.bus)
            if errors:
                raise RuntimeError("Torque-off could not be verified for: "
                                   + ", ".join(errors))
            self.armed = False


def print_state(states: dict[str, JointState]) -> None:
    print("joint             tick  limits       torque  mode")
    for name, state in states.items():
        print(f"{name:17} {state.position:4}  {state.minimum:4}..{state.maximum:4}  "
              f"{'on ' if state.torque_enabled else 'off'}     {state.operating_mode}")


def run_probe(bus) -> None:
    bus.connect()
    try:
        print_state(inspect(bus))
    finally:
        # A read-only probe must leave the existing torque state alone.
        bus.disconnect(disable_torque=False)


def run_jog(bus, max_step: int, max_travel: int, timeout: float) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("jog requires an interactive terminal")
    bus.connect()
    session = None
    try:
        states = inspect(bus)
        print_state(states)
        session = JogSession(bus, states, max_step=max_step, max_travel=max_travel)
        print("Type 'arm' to enable torque, then '<joint> <signed ticks>' to move one step.")
        print(f"Each step is at most {max_step} ticks; each joint stays within ±{max_travel} "
              "ticks of its starting position.")
        print(f"'status' reads positions; 'stop' or inactivity for {timeout:g}s disables torque.")
        while True:
            sys.stdout.write("leader> ")
            sys.stdout.flush()
            ready, _, _ = select.select([sys.stdin], [], [], timeout if session.armed else None)
            if not ready:
                print("\nInactivity timeout; stopping.")
                break
            parts = sys.stdin.readline().strip().split()
            if not parts or parts[0] in ("stop", "quit", "exit"):
                break
            if parts == ["status"]:
                print_state(inspect(bus))
            elif parts == ["arm"]:
                session.arm()
                print("Torque enabled at the measured pose.")
            elif len(parts) == 2 and parts[0] in JOINTS:
                try:
                    target = session.jog(parts[0], int(parts[1]))
                    time.sleep(0.15)
                    measured = int(bus.read("Present_Position", parts[0], normalize=False))
                    print(f"{parts[0]}: target={target}, measured={measured}")
                    if abs(target - measured) > 60:
                        raise RuntimeError("joint is not tracking its small target; stopping")
                except ValueError as error:
                    print(error)
            else:
                print("Use 'arm', 'status', '<joint> <signed ticks>', or 'stop'.")
    finally:
        try:
            if session is not None:
                session.stop()
        finally:
            # stop() attempted every motor; always close the serial port.
            bus.disconnect(disable_torque=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ports", help="list local serial device paths")
    probe = commands.add_parser("probe", help="read joint state without moving the arm")
    probe.add_argument("--port", required=True)
    jog = commands.add_parser("jog", help="interactive, bounded joint movement")
    jog.add_argument("--port", required=True)
    jog.add_argument("--enable-motion", action="store_true", required=True)
    jog.add_argument("--max-step", type=int, default=10)
    jog.add_argument("--max-travel", type=int, default=80)
    jog.add_argument("--timeout", type=float, default=10.0)
    web = commands.add_parser("web", help="local browser controls for all six joints")
    web.add_argument("--port", required=True, help="arm USB serial port")
    web.add_argument("--http-port", type=int, default=8767)
    web.add_argument("--enable-motion", action="store_true", required=True)
    web.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.command == "ports":
        print("\n".join(glob("/dev/cu.*")) or "No serial devices found")
        return 0
    bus = make_bus(args.port)
    if args.command == "probe":
        run_probe(bus)
    elif args.command == "jog":
        if not 1 <= args.timeout <= 60:
            parser.error("--timeout must be 1..60 seconds")
        run_jog(bus, args.max_step, args.max_travel, args.timeout)
    else:
        from .arm_web import serve
        serve(bus, http_port=args.http_port, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
