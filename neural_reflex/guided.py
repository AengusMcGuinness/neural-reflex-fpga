"""On-screen target movement for guided recording: the target IS the label.

A target arm, drawn beside the camera image, slowly picks an item off the table at a few spots.
You copy it with your band arm. Every recorded frame stores the target's shoulder lift, elbow,
hand position and grip, so labels are exact and chosen to stay inside what the robot can do.
The camera still runs, to measure how closely you followed.

Human side view, in arm lengths with the shoulder at the origin: forward +x, up +y.
shoulder_lift 0 = arm hanging down, 90 = level forward. elbow_flex 0 = straight.
"""

from __future__ import annotations

import math

import numpy as np

UPPER = 0.47           # shoulder -> elbow, fraction of shoulder-to-hand length
LOWER = 0.53           # elbow -> hand point
TABLE_UP = -0.55       # seated: table surface below the shoulder, in arm lengths
TARGETS = (0.45, 0.60, 0.75)    # forward distances of the spots, in arm lengths
ABOVE, LIFT = 0.25, 0.20        # hover height and lift height above the table
REST = (0.30, TABLE_UP + 0.02)  # hand resting on the table near your body
ROUNDS = 2

# (key, instruction, seconds, hand goal or None to stay, grip goal)
_SETUP = (
    ("ready", "Sit at the table, band arm toward the camera side. Rest your hand on the table near you", 5, REST, 0.0),
    ("sync", "SYNC: make 4 sharp fist clenches", 6, REST, 0.0),
    ("rest", "Relax. Next: copy the white arm on the right, slowly", 3, REST, 0.0),
)


def _pick(spot: int, f: float):
    table = TABLE_UP
    return (
        ("reach", f"Spot {spot}: reach over it", 4, (f, table + ABOVE), 0.0),
        ("lower", f"Spot {spot}: lower your hand onto the table", 2, (f, table + 0.02), 0.0),
        ("grip", "Close your hand into a firm FIST", 2, None, 1.0),
        ("hold", "Hold the fist", 1, None, 1.0),
        ("lift", "Lift, keeping the fist", 3, (f, table + LIFT), 1.0),
        ("place", "Put it back down", 3, (f, table + 0.02), 1.0),
        ("open", "Open your hand", 1, None, 0.0),
        ("return", "Bring your hand back near you", 4, REST, 0.0),
    )


def build_timeline(rounds: int = ROUNDS, spots=TARGETS):
    """List of (key, text, seconds, round, spot, start_hand, end_hand, start_grip, end_grip)."""
    steps = []
    hand, grip = REST, 0.0
    for key, text, seconds, goal, g in _SETUP:
        goal = goal or hand
        steps.append((key, text, seconds, None, None, hand, goal, grip, g))
        hand, grip = goal, g
    for r in range(1, rounds + 1):
        for spot, f in enumerate(spots, 1):
            for key, text, seconds, goal, g in _pick(spot, f):
                goal = goal or hand
                steps.append((key, f"Round {r}/{rounds}. {text}", seconds, r, spot, hand, goal, grip, g))
                hand, grip = goal, g
    steps.append(("rest_end", "Done. Rest. Recording stops automatically", 3, None, None, hand, REST, grip, 0.0))
    return tuple(steps)


TIMELINE = build_timeline()
PROTOCOL = tuple((s[0], s[1], s[2], s[3]) for s in TIMELINE)   # same shape arm_pose's guide uses
TOTAL_S = sum(s[2] for s in TIMELINE)


def _ease(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (s * (6 * s - 15) + 10)


def arm_ik(forward: float, up: float):
    """Shoulder lift and elbow flex (degrees) putting the hand at (forward, up), elbow below/behind."""
    d = min(math.hypot(forward, up), UPPER + LOWER - 1e-6)
    cos_elbow = (d * d - UPPER ** 2 - LOWER ** 2) / (2 * UPPER * LOWER)
    elbow = math.acos(max(-1.0, min(1.0, cos_elbow)))
    # Angle of the shoulder->hand line measured from straight down, forward positive.
    line = math.atan2(forward, -up)
    inner = math.atan2(LOWER * math.sin(elbow), UPPER + LOWER * math.cos(elbow))
    return math.degrees(line - inner), math.degrees(elbow)


def target_at(elapsed: float) -> dict | None:
    """The target at this moment: step, round, spot, hand position, angles and grip. None when done."""
    start = 0.0
    for key, text, seconds, rnd, spot, h0, h1, g0, g1 in TIMELINE:
        if elapsed < start + seconds:
            s = _ease((elapsed - start) / seconds)
            forward = h0[0] + (h1[0] - h0[0]) * s
            up = h0[1] + (h1[1] - h0[1]) * s
            lift, elbow = arm_ik(forward, up)
            return {"step": key, "text": text, "round": rnd, "spot": spot, "forward": forward, "up": up,
                    "shoulder_lift": lift, "elbow_flex": elbow, "grip": g0 + (g1 - g0) * s,
                    "step_left_s": start + seconds - elapsed, "total_left_s": TOTAL_S - elapsed}
        start += seconds
    return None


def trajectory(dt: float = 0.02) -> dict:
    """The whole target as arrays, for checking it against the robot offline."""
    t = np.arange(0, TOTAL_S, dt)
    rows = [target_at(x) for x in t]
    return {"time": t, **{k: np.array([r[k] for r in rows]) for k in ("forward", "up", "shoulder_lift", "elbow_flex", "grip")},
            "step": [r["step"] for r in rows]}


def measured_angles(world_landmarks, visibility, side: str) -> dict | None:
    """Your signed shoulder lift and elbow flex in the same side view as the target, using the
    camera's vertical when the table hides your hips. None if the arm isn't visible."""
    if world_landmarks is None:
        return None
    from .arm_pose import SIDES
    from .hand_map import body_frame
    index = SIDES[side]
    if min(visibility[index[k]] for k in ("shoulder", "elbow", "wrist")) < 0.5 or min(visibility[11], visibility[12]) < 0.5:
        return None
    p = np.asarray(world_landmarks, dtype=float)
    up, forward, _ = body_frame(p, visibility)
    shoulder, elbow = p[index["shoulder"]], p[index["elbow"]]
    hand = (p[index["index"]] + p[index["pinky"]]) / 2
    upper, lower = elbow - shoulder, hand - elbow
    lift = math.degrees(math.atan2(float(np.dot(upper, forward)), float(-np.dot(upper, up))))
    cos_e = np.dot(upper, lower) / (np.linalg.norm(upper) * np.linalg.norm(lower) + 1e-9)
    reach = np.linalg.norm(upper) + np.linalg.norm(lower)
    return {"shoulder_lift": lift, "elbow_flex": math.degrees(math.acos(float(np.clip(cos_e, -1, 1)))),
            "forward": float(np.dot(hand - shoulder, forward) / reach), "up": float(np.dot(hand - shoulder, up) / reach)}


def draw_panel(frame, target: dict, measured: dict | None = None, width: int = 300):
    """Target arm (white) and, if the camera sees it, your arm (green) on a panel at the right."""
    import cv2
    h, w = frame.shape[:2]
    x0 = w - width
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h - 118), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    scale = width * 0.62
    sx, sy = x0 + int(width * 0.22), int((h - 118) * 0.30)
    P = lambda f, u: (int(sx + f * scale), int(sy - u * scale))
    table_y = P(0, TABLE_UP)[1]
    cv2.line(frame, (x0 + 8, table_y), (w - 8, table_y), (120, 120, 120), 2)
    for i, f in enumerate(TARGETS, 1):
        cx = P(f, TABLE_UP)[0]
        color = (0, 200, 255) if target.get("spot") == i else (90, 90, 90)
        cv2.line(frame, (cx - 14, table_y + 6), (cx + 14, table_y + 6), color, 4)
        cv2.putText(frame, str(i), (cx - 5, table_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (sx, sy), 6, (200, 200, 200), -1)

    def arm(lift, elbow, color, thick):
        a = math.radians(lift)
        ex, ey = UPPER * math.sin(a), -UPPER * math.cos(a)
        b = a + math.radians(elbow)
        hx, hy = ex + LOWER * math.sin(b), ey - LOWER * math.cos(b)
        cv2.line(frame, (sx, sy), P(ex, ey), color, thick, cv2.LINE_AA)
        cv2.line(frame, P(ex, ey), P(hx, hy), color, thick, cv2.LINE_AA)
        return P(hx, hy)

    if measured and measured.get("shoulder_lift") is not None:
        arm(measured["shoulder_lift"], measured["elbow_flex"], (90, 220, 90), 4)
    hand = arm(target["shoulder_lift"], target["elbow_flex"], (255, 255, 255), 9)
    radius = int(9 + 7 * target["grip"])
    cv2.circle(frame, hand, radius, (255, 255, 255), -1 if target["grip"] > 0.5 else 2, cv2.LINE_AA)
    cv2.putText(frame, "copy the white arm", (x0 + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, "FIST" if target["grip"] > 0.5 else "open hand", (x0 + 12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255) if target["grip"] > 0.5 else (170, 170, 170), 1, cv2.LINE_AA)
    if measured is not None:
        cv2.putText(frame, "green: you", (x0 + 12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 90), 1, cv2.LINE_AA)
