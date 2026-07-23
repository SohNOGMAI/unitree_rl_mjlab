#!/usr/bin/env python3
"""Short, ultra-slow, stand-alone hoist proof test for the G1 wire module.

RobStride06 is the main 51 mm effective-radius winch.  GL40_2 applies only
the small back tension used by the existing wire_main.py.  This program does
not communicate with G1 and must be run separately from g1_ctrl.

The RS06 manual rates the motor at 11 Nm.  A 480 N cable-force limit at the
current pulley is 24.48 Nm and is therefore a short-duration peak-region
limit, not a continuous operating point.  This script deliberately permits
it only for a time-limited proof test and never commands 480 N continuously.
"""

from __future__ import annotations

import argparse
import csv
import os
import select
import signal
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path

CONTROL_HZ = 200.0
GRAVITY_M_S2 = 9.81

ROBSTRIDE_PULLEY_RADIUS_M = 0.051
GL40_PULLEY_RADIUS_M = 0.012
ROBSTRIDE_RATED_TORQUE_NM = 11.0
ROBSTRIDE_PROTOCOL_PEAK_TORQUE_NM = 36.0
ABSOLUTE_MAX_TENSION_N = 480.0

DEFAULT_ROBOT_MASS_KG = 33.341142  # Sum of inertials in the repository G1 XML.
DEFAULT_MODULE_MASS_KG = 2.5


class FeedbackError(RuntimeError):
    pass


class TerminalKeys:
    def __enter__(self):
        self.enabled = sys.stdin.isatty()
        self.old = None
        if self.enabled:
            self.old = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        return self

    def read(self) -> str:
        if not self.enabled:
            return ""
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        return ""

    def __exit__(self, exc_type, exc, tb):
        if self.old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old)


@dataclass
class Sample:
    can_id: int
    position_rad: float
    velocity_rad_s: float
    torque_nm: float
    temperature_c: float
    status: int


def parse_sample(result, name: str) -> Sample:
    if result is None or len(result) != 6 or result[0] is None or result[1] is None:
        raise FeedbackError(f"{name}: no fresh CAN feedback")
    return Sample(
        can_id=int(result[0]),
        position_rad=float(result[1]),
        velocity_rad_s=float(result[2]),
        torque_nm=float(result[3]),
        temperature_c=float(result[4]),
        status=int(result[5]) if result[5] is not None else 0,
    )


def send_fresh(motor, p_ref: float, v_ref: float, kp: float, kd: float, tau_ff: float,
               name: str) -> Sample:
    # MotorHandle intentionally returns its previous sample on timeout.  A
    # hoist watchdog must instead inspect the backend's fresh return value.
    if motor.backend == "xiaomi":
        result = motor.controller.send_control_command(
            p_ref=p_ref, v_ref=v_ref, kp=kp, kd=kd, tau_ff=tau_ff
        )
    elif motor.backend == "tmotor":
        result = motor.controller.send_rad_command(p_ref, v_ref, kp, kd, tau_ff)
    else:
        raise FeedbackError(f"{name}: unsupported backend {motor.backend}")
    return parse_sample(result, name)


class HoistTest:
    def __init__(self, args):
        self.a = args
        self.rob = None
        self.gl = None
        self.shutdown_bus = None
        self.stop_requested = False
        self.immediate_stop_requested = False
        self.rows = []

        total_mass = args.robot_mass_kg + args.module_mass_kg
        self.weight_n = total_mass * GRAVITY_M_S2
        self.support_tension_n = self.weight_n / args.vertical_component
        self.rated_tension_n = ROBSTRIDE_RATED_TORQUE_NM / ROBSTRIDE_PULLEY_RADIUS_M
        self.max_torque_nm = args.max_tension_n * ROBSTRIDE_PULLEY_RADIUS_M

    def print_plan(self):
        min_uz = self.weight_n / self.a.max_tension_n
        print("\n--- Slow hoist proof-test plan ---")
        print(f"total suspended mass : {self.a.robot_mass_kg + self.a.module_mass_kg:.3f} kg")
        print(f"weight               : {self.weight_n:.1f} N")
        print(f"wire vertical factor : {self.a.vertical_component:.3f}")
        print(f"support feed-forward : {self.support_tension_n:.1f} N")
        print(f"cable speed          : {self.a.speed_m_s * 1000.0:.1f} mm/s")
        print(f"test travel          : {self.a.travel_m * 1000.0:.1f} mm")
        print(f"hard tension limit   : {self.a.max_tension_n:.1f} N ({self.max_torque_nm:.2f} Nm)")
        print(f"RS06 rated force     : {self.rated_tension_n:.1f} N (11 Nm)")
        print(f"minimum usable u_z   : {min_uz:.3f}")
        print(f"peak-region timeout  : {self.a.max_peak_seconds:.1f} s")
        print("SPACE = controlled tension ramp-down; Ctrl-C = immediate motor disable")

    def validate(self):
        if self.a.max_tension_n <= 0.0 or self.a.max_tension_n > ABSOLUTE_MAX_TENSION_N:
            raise ValueError(f"--max-tension-n must be in (0, {ABSOLUTE_MAX_TENSION_N}]")
        if self.max_torque_nm > ROBSTRIDE_PROTOCOL_PEAK_TORQUE_NM:
            raise ValueError("requested tension exceeds the RS06 protocol torque range")
        if not 0.0 < self.a.vertical_component <= 1.0:
            raise ValueError("--vertical-component must be in (0, 1]")
        if self.support_tension_n >= self.a.max_tension_n:
            raise ValueError(
                "480 N cannot support this mass/anchor geometry. Move the anchor closer "
                "to vertical; do not increase the limit."
            )
        if not 0.0 < self.a.speed_m_s <= 0.01:
            raise ValueError("proof-test --speed-m-s must be in (0, 0.01]")
        if not 0.0 < self.a.travel_m <= 0.02:
            raise ValueError("proof-test --travel-m must be in (0, 0.02]")
        if self.a.max_peak_seconds <= 0.0:
            raise ValueError("--max-peak-seconds must be positive")

    def setup(self):
        try:
            from evarl_motors_lib import create_motor, shutdown_bus
        except ImportError as exc:
            raise RuntimeError(
                "evarl_motors_lib/python-can is not installed in this Python environment. "
                "Install evarl_motors_lib as described in README_JA.md."
            ) from exc
        self.shutdown_bus = shutdown_bus
        self.rob = create_motor(
            self.a.bus, self.a.robstride_id, "RobStride06",
            motor_dir=self.a.robstride_dir, socket_timeout=0.02,
        )
        self.gl = create_motor(
            self.a.bus, self.a.gl40_id, "GL40_2",
            motor_dir=self.a.gl40_dir, socket_timeout=0.02,
        )
        rob0 = parse_sample(self.rob.enable_motor(), "RobStride enable")
        gl0 = parse_sample(self.gl.enable_motor(), "GL40 enable")
        self.rob.set_control_mode()
        print(f"RobStride enabled: pos={rob0.position_rad:+.4f} rad temp={rob0.temperature_c:.1f} C")
        print(f"GL40 enabled:      pos={gl0.position_rad:+.4f} rad temp={gl0.temperature_c:.1f} C")
        return rob0.position_rad

    def disable(self):
        for name, motor in (("RobStride", self.rob), ("GL40", self.gl)):
            if motor is None:
                continue
            try:
                motor.disable_motor()
                print(f"{name} disabled")
            except Exception as exc:
                print(f"WARNING: failed to disable {name}: {exc}", file=sys.stderr)
        if self.rob is not None and self.shutdown_bus is not None:
            try:
                self.shutdown_bus(self.rob.controller.bus_name)
            except Exception as exc:
                print(f"WARNING: failed to close CAN bus: {exc}", file=sys.stderr)

    def _signal_stop(self, signum, frame):
        del signum, frame
        self.immediate_stop_requested = True

    def run(self):
        try:
            start_position = self.setup()
        except Exception:
            # setup() can fail after only one of the two motors was enabled.
            # Never leave that motor energized on a partial initialization.
            self.disable()
            raise
        old_sigint = signal.signal(signal.SIGINT, self._signal_stop)
        period = 1.0 / CONTROL_HZ
        state = "RAMP_UP"
        state_start = time.monotonic()
        test_start = state_start
        last = state_start
        last_print = 0.0
        base_tension = 0.0
        velocity_integral_n = 0.0
        peak_started = None

        try:
            with TerminalKeys() as keys:
                while True:
                    loop_start = time.monotonic()
                    dt = min(max(loop_start - last, 0.0), 0.05)
                    last = loop_start

                    key = keys.read()
                    if key == " ":
                        self.stop_requested = True
                        print("\nSPACE: controlled ramp-down requested")
                    if self.immediate_stop_requested:
                        raise KeyboardInterrupt

                    reel_in_m = self.a.reel_sign * (
                        getattr(self, "last_rob_position", start_position) - start_position
                    ) * ROBSTRIDE_PULLEY_RADIUS_M
                    cable_velocity_m_s = self.a.reel_sign * getattr(
                        self, "last_rob_velocity", 0.0
                    ) * ROBSTRIDE_PULLEY_RADIUS_M

                    if self.stop_requested and state != "RAMP_DOWN":
                        state = "RAMP_DOWN"
                        state_start = loop_start

                    if state == "RAMP_UP":
                        base_tension = min(
                            self.support_tension_n,
                            base_tension + self.a.tension_ramp_n_s * dt,
                        )
                        tension_cmd = base_tension
                        velocity_ref = 0.0
                        if base_tension >= self.support_tension_n - 1e-6:
                            state = "HOIST"
                            state_start = loop_start
                            print("[WIRE] support tension reached -> HOIST")
                    elif state == "HOIST":
                        velocity_ref = self.a.speed_m_s
                        velocity_error = velocity_ref - cable_velocity_m_s
                        velocity_integral_n += self.a.velocity_ki * velocity_error * dt
                        velocity_integral_n = max(
                            -self.a.integral_limit_n,
                            min(self.a.integral_limit_n, velocity_integral_n),
                        )
                        tension_cmd = (
                            self.support_tension_n
                            + self.a.velocity_kp * velocity_error
                            + velocity_integral_n
                        )
                        if reel_in_m >= self.a.travel_m:
                            state = "HOLD"
                            state_start = loop_start
                            print("[WIRE] target travel reached -> HOLD")
                    elif state == "HOLD":
                        velocity_ref = 0.0
                        velocity_error = -cable_velocity_m_s
                        tension_cmd = self.support_tension_n + self.a.velocity_kp * velocity_error
                        if loop_start - state_start >= self.a.hold_seconds:
                            state = "RAMP_DOWN"
                            state_start = loop_start
                            print("[WIRE] hold complete -> RAMP_DOWN")
                    elif state == "RAMP_DOWN":
                        velocity_ref = 0.0
                        base_tension = max(0.0, base_tension - self.a.release_ramp_n_s * dt)
                        tension_cmd = base_tension
                        if base_tension <= 0.0:
                            print("[WIRE] tension command is zero -> DISABLE")
                            break
                    else:
                        raise RuntimeError(f"unknown state {state}")

                    tension_cmd = max(0.0, min(self.a.max_tension_n, tension_cmd))
                    if tension_cmd > self.rated_tension_n:
                        if peak_started is None:
                            peak_started = loop_start
                        elif loop_start - peak_started >= self.a.max_peak_seconds:
                            print("[WIRE] peak-region time limit reached -> RAMP_DOWN")
                            state = "RAMP_DOWN"
                            state_start = loop_start
                    else:
                        peak_started = None

                    rob_tau_cmd = self.a.reel_sign * tension_cmd * ROBSTRIDE_PULLEY_RADIUS_M
                    gl_tau_cmd = self.a.gl40_dir_command * (
                        self.a.gl40_back_tension_nm if tension_cmd > 0.0 else 0.0
                    )
                    rob_sample = send_fresh(self.rob, 0.0, 0.0, 0.0, 0.0, rob_tau_cmd, "RobStride")
                    gl_sample = send_fresh(self.gl, 0.0, 0.0, 0.0, 0.0, gl_tau_cmd, "GL40")
                    self.last_rob_position = rob_sample.position_rad
                    self.last_rob_velocity = rob_sample.velocity_rad_s

                    reel_in_m = self.a.reel_sign * (
                        rob_sample.position_rad - start_position
                    ) * ROBSTRIDE_PULLEY_RADIUS_M
                    cable_velocity_m_s = (
                        self.a.reel_sign * rob_sample.velocity_rad_s
                        * ROBSTRIDE_PULLEY_RADIUS_M
                    )
                    estimated_tension_n = abs(rob_sample.torque_nm) / ROBSTRIDE_PULLEY_RADIUS_M

                    if abs(cable_velocity_m_s) > self.a.max_cable_speed_m_s:
                        print("[WIRE] overspeed detected -> RAMP_DOWN")
                        state = "RAMP_DOWN"
                        state_start = loop_start
                    if estimated_tension_n > self.a.max_tension_n * 1.05:
                        print("[WIRE] estimated over-tension -> RAMP_DOWN")
                        state = "RAMP_DOWN"
                        state_start = loop_start
                    if rob_sample.temperature_c > self.a.max_temperature_c or gl_sample.temperature_c > self.a.max_temperature_c:
                        raise RuntimeError("motor over-temperature")
                    if rob_sample.status != 0:
                        raise RuntimeError(f"RobStride fault code {rob_sample.status}")
                    # GL MIT status 1 means enabled.  Some firmware reports 0.
                    if gl_sample.status not in (0, 1):
                        raise RuntimeError(f"GL40 fault/status code {gl_sample.status}")
                    if loop_start - test_start > self.a.max_total_seconds:
                        print("[WIRE] total test timeout -> RAMP_DOWN")
                        state = "RAMP_DOWN"
                        state_start = loop_start

                    self.rows.append({
                        "time_s": loop_start - test_start,
                        "state": state,
                        "reel_in_m": reel_in_m,
                        "cable_velocity_m_s": cable_velocity_m_s,
                        "velocity_ref_m_s": velocity_ref,
                        "tension_cmd_n": tension_cmd,
                        "estimated_tension_n": estimated_tension_n,
                        "rob_position_rad": rob_sample.position_rad,
                        "rob_velocity_rad_s": rob_sample.velocity_rad_s,
                        "rob_torque_nm": rob_sample.torque_nm,
                        "rob_temperature_c": rob_sample.temperature_c,
                        "rob_status": rob_sample.status,
                        "gl_velocity_rad_s": gl_sample.velocity_rad_s,
                        "gl_torque_nm": gl_sample.torque_nm,
                        "gl_temperature_c": gl_sample.temperature_c,
                        "gl_status": gl_sample.status,
                    })

                    if loop_start - last_print >= 0.2:
                        print(
                            f"[WIRE] state={state:9s} reel={reel_in_m:+.4f}m "
                            f"v={cable_velocity_m_s:+.4f}m/s "
                            f"Tcmd={tension_cmd:6.1f}N Test={estimated_tension_n:6.1f}N "
                            f"temp={rob_sample.temperature_c:4.1f}C"
                        )
                        last_print = loop_start

                    sleep_time = period - (time.monotonic() - loop_start)
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nCtrl-C: immediate motor disable requested", file=sys.stderr)
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            # Best effort zero command before disabling.  Do not hide the
            # original fault if CAN has already failed.
            try:
                send_fresh(self.rob, 0.0, 0.0, 0.0, 0.0, 0.0, "RobStride")
                send_fresh(self.gl, 0.0, 0.0, 0.0, 0.0, 0.0, "GL40")
            except Exception as exc:
                print(f"WARNING: zero command failed: {exc}", file=sys.stderr)
            self.disable()
            self.write_csv()

    def write_csv(self):
        if not self.rows:
            return
        path = Path(self.a.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"CSV saved: {path}")


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true", help="enable real motors; omitted means plan-only")
    p.add_argument("--bus", default="can0")
    p.add_argument("--robstride-id", type=int, default=127)
    p.add_argument("--gl40-id", type=int, default=1)
    p.add_argument("--robstride-dir", type=int, choices=(-1, 1), default=1)
    p.add_argument("--gl40-dir", type=int, choices=(-1, 1), default=1)
    p.add_argument("--reel-sign", type=int, choices=(-1, 1), default=1,
                   help="command sign that winds the main cable")
    p.add_argument("--gl40-dir-command", type=int, choices=(-1, 1), default=1)
    p.add_argument("--robot-mass-kg", type=float, default=DEFAULT_ROBOT_MASS_KG)
    p.add_argument("--module-mass-kg", type=float, default=DEFAULT_MODULE_MASS_KG)
    p.add_argument("--vertical-component", type=float, default=1.0,
                   help="vertical component u_z of the unit cable direction")
    p.add_argument("--speed-m-s", type=float, default=0.005)
    p.add_argument("--travel-m", type=float, default=0.010)
    p.add_argument("--max-tension-n", type=float, default=ABSOLUTE_MAX_TENSION_N)
    p.add_argument("--tension-ramp-n-s", type=float, default=50.0)
    p.add_argument("--release-ramp-n-s", type=float, default=80.0)
    p.add_argument("--velocity-kp", type=float, default=2000.0,
                   help="outer-loop cable velocity gain [N/(m/s)]")
    p.add_argument("--velocity-ki", type=float, default=500.0,
                   help="outer-loop cable velocity integral gain [N/m]")
    p.add_argument("--integral-limit-n", type=float, default=40.0)
    p.add_argument("--hold-seconds", type=float, default=0.5)
    p.add_argument("--max-peak-seconds", type=float, default=8.0)
    p.add_argument("--max-total-seconds", type=float, default=20.0)
    p.add_argument("--max-cable-speed-m-s", type=float, default=0.03)
    p.add_argument("--max-temperature-c", type=float, default=70.0)
    p.add_argument("--gl40-back-tension-nm", type=float, default=0.05)
    p.add_argument("--csv", default="results/wire_hoist/slow_hoist.csv")
    return p


def main():
    args = make_parser().parse_args()
    test = HoistTest(args)
    test.validate()
    test.print_plan()
    if not args.execute:
        print("\nPlan only: motors were not opened. Add --execute only after the unloaded sign test.")
        return

    print("\nREQUIRED: independent secondary support, physical E-stop, clear exclusion zone.")
    print("The secondary support must catch the robot after no more than the commanded travel.")
    confirmation = input('Type exactly "SECONDARY SUPPORT READY" to enable the motors: ')
    if confirmation != "SECONDARY SUPPORT READY":
        print("Confirmation failed; motors were not opened.")
        return
    test.run()


if __name__ == "__main__":
    main()
