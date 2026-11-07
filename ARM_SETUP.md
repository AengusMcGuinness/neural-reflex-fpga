# One black Hiwonder SO-ARM101 leader arm

The single arm available for Neo is the **black leader arm**. Hiwonder documents
it as a manually moved input device with six HX-10HM servos and a handle. The
white follower arm, stronger HX-30HM servos, and grasping claw are unavailable.
The leader's servos expose position and torque registers, so this repo includes
an **experimental, bounded joint jog** to move its tip. Cartesian tip targeting
and grasping have not been verified on this hardware.

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

## First motion test

The black leader is not Hiwonder's intended output arm. Start with a small
shoulder-pan movement while supporting the arm and watching the tip:

```bash
./scripts/leader_arm jog --port /dev/cu.YOUR_ARM_PORT --enable-motion
```

At the prompt:

```text
arm
shoulder_pan 5
shoulder_pan -5
stop
```

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
