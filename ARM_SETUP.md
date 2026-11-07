# One black Hiwonder SO-ARM101 leader arm

The single arm available for Neo is the **black leader arm**. Hiwonder documents
it as a manually moved input device with six HX-10HM servos and a handle. The
white follower arm, stronger HX-30HM servos, and grasping claw are unavailable.
The leader's servos expose position and torque registers, so this repo includes
an **experimental, bounded joint jog** to move its tip. Cartesian tip targeting
and grasping have not been verified on this hardware.

**Live status (November 7, 2026):** The black arm was detected on this Mac and
all six motors answered the read-only probe at 12.1–12.3 V. A 120-tick
shoulder-pan sweep visibly moved the arm and returned it near its start. A
60-tick positive shoulder-lift movement visibly moved the tip down, then
returned it. The final probe showed torque off on all six motors.

## Hardware connection

Clamp the base to a stable table. Connect its controller's USB-C data port to
the Mac. Use only the supplied 12 V adapter for servo power; USB alone is not
the motor supply. Clear the arm's sweep area and keep the power plug within
reach before enabling motion.

## Install the driver

From the Neo repo:

```bash
./scripts/setup_leader_arm.sh
```

This downloads Hiwonder's LeRobot fork at a tested revision into ignored
`.local/`, then creates an isolated Python environment with only the serial
driver dependencies. It does not install the large model-training stack or
change system Python. The source is Hiwonder's
[SO-ARM101 fork](https://github.com/Hiwonder/hiwonder-SoArm-101).

## Find and inspect the arm

```bash
./scripts/leader_arm ports
./scripts/leader_arm probe --port /dev/cu.YOUR_ARM_PORT
```

Run `ports` before and after plugging in the USB-C data cable. The newly added
`/dev/cu.*` path is the arm. `probe` reads all six positions, configured motor
limits, torque state, and operating mode. It never commands motion or changes
torque. The first motion test requires all joints to report position mode `0`,
torque `off`, and valid limits.

## Local web controls

From the Neo folder on the Mac connected to the arm, run:

```bash
./scripts/leader_arm web --port /dev/cu.usbmodem5CE51146561 --enable-motion
```

Open [http://127.0.0.1:8767/](http://127.0.0.1:8767/) on that Mac. The page
shows each joint's measured position, configured limits, operating mode,
voltage, temperature, and torque state. Select **Position control**, turn on
torque for **one joint**, then use its `−` and `+` buttons, target slider, or
exact tick input. The slider sends its target when released. Start with a
5-tick step and keep the arm's movement area clear. **Free / manual** and
**Stop all motors** turn off torque so the arm can be moved by hand. Torque
also turns off after 15 seconds without a movement command or when the
server stops with Ctrl+C.

The server binds only to `127.0.0.1`. It starts with all torque off, checks
joint limits, allows at most 20 ticks per step and 160 ticks of travel per
joint from the pose where that joint's torque was enabled. Slider targets are
ramped in 20-tick increments and checked against measured position. Set the
arm in a pose where it can rest safely when torque turns off. The
shoulder-pan and shoulder-lift controls have been checked on the physical
arm; the other four controls are exposed through the same motor bus but have
not yet been physically jogged.

If a servo reports overload or stops tracking, the page shows a motor fault
and attempts to turn every motor off. If torque cannot be verified off,
disconnect the 12 V adapter while supporting the arm. Check for a joint at
its mechanical stop or a load that the small leader servos cannot hold;
after clearing the condition and restoring power, use **Retry after checking
arm**. The page marks old readings as stale during a fault. A power cycle
cleared a previous overload on shoulder lift and elbow; its cause was not
confirmed. The page's arm modes are **Position control** and **Free / manual**.
Raw servo velocity and PWM modes are shown only as a numeric operating mode
because they can command continuous motion.

## First motion test

The black leader is not Hiwonder's intended output arm. The terminal jog is
also available if you prefer typed commands:

```bash
./scripts/leader_arm jog --port /dev/cu.YOUR_ARM_PORT --enable-motion
```

On the tested Mac, the current command is:

```bash
./scripts/leader_arm jog --port /dev/cu.usbmodem5CE51146561 --enable-motion
```

At the prompt:

```text
arm
shoulder_pan 5
shoulder_pan -5
stop
```

The web page uses larger 20-tick steps and permits up to 160 ticks of travel
from its starting pose. The arm port on the tested Mac was
`/dev/cu.usbmodem5CE51146561`, but USB port names can change after
reconnection, so run `ports` if it disappears.

`arm` sets each goal to its measured position before enabling torque. Each
command moves one joint by at most 10 raw ticks, and no joint can travel more
than 80 ticks from its starting position in one session. The controller reads
back the moved joint, stops if it fails to track, disables torque on `stop`,
Ctrl+C, or an error, and times out after 10 seconds without a command.

These are **joint** movements; they are not millimetre-accurate X/Y/Z end
effector commands. The leader's handle cannot grasp an object like the absent
follower's claw. Once the physical movement is checked, Neo can map a decoded
EMG state to bounded movement of selected joints or use a measured kinematic
model for Cartesian tip movement.

Hiwonder's [SO-ARM101 manual](https://docs.hiwonder.com/projects/LeRobot/en/latest/docs/SO_ARM101_Open_Source_6_Axis_Robotic_Arm_User_Manual.html)
describes the leader's normal role and servo ratings. The source driver uses
Hiwonder's `hx30hm` serial profile for the leader role too; this setup uses
that same profile without rewriting motor IDs or calibration.
