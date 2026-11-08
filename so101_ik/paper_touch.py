"""Read the arm with the pen on the paper and record the MODEL tip x, y, z under the current calibration.

Run once per touch (torque off, arm held still). so101_ik.write_plan then takes the fitted plane through the
touches via --plane_z_m / --plane_slope / --distance_m / --lateral_m, so any calibration bias at the writing
spot cancels. Usage: PYTHONPATH=.local/hiwonder-SoArm-101/src .venv-ik/bin/python -m so101_ik.paper_touch
"""
import json, time, sys, math
from neural_reflex.leader_arm import make_bus, inspect
from so101_ik.fk import JOINTS, build_so101_model
from so101_ik.touch_calibrate import tip_xyz, PAN_AXIS_X_M
from so101_ik.write_arm import load_calibration

OUT = "data/so101_ik/paper_points.json"
bus = make_bus('/dev/ttyACM0'); bus.connect()
try:
    s = inspect(bus); ticks = {j: int(s[j].position) for j in JOINTS}
    time.sleep(0.3); s2 = inspect(bus); moved = any(abs(int(s2[j].position) - ticks[j]) > 3 for j in JOINTS)
finally:
    bus.disconnect(disable_torque=False)
cal = load_calibration(); m = build_so101_model((0, 0, 0.05))
xyz = tip_xyz(ticks, cal, m)
try:
    pts = json.load(open(OUT))
except Exception:
    pts = []
print("ticks", ticks, "| moving" if moved else "| still")
print(f"model tip: x {xyz[0]*1000:.1f}  y {xyz[1]*1000:.1f}  z {xyz[2]*1000:.1f} mm   radial from pan axis {math.hypot(xyz[0]-PAN_AXIS_X_M, xyz[1])*1000:.0f} mm")
if moved:
    print("NOT saved: arm was moving")
else:
    pts.append({"ticks": ticks, "xyz_m": [float(v) for v in xyz], "t": time.time()})
    json.dump(pts, open(OUT, "w"), indent=1)
    print(f"SAVED as paper point {len(pts)}")
    if len(pts) >= 2:
        zs = [p["xyz_m"][2] for p in pts]; xs = [p["xyz_m"][0] for p in pts]; ys = [p["xyz_m"][1] for p in pts]
        print(f"so far: z mean {sum(zs)/len(zs)*1000:.1f} mm (spread {(max(zs)-min(zs))*1000:.1f} mm), "
              f"x {min(xs)*1000:.0f}..{max(xs)*1000:.0f}, y {min(ys)*1000:.0f}..{max(ys)*1000:.0f} mm")
