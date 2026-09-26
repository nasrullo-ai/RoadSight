"""Each rule fires on a hand-built scenario and stays quiet on a benign twin."""

import numpy as np
import pytest
from synth import CAR, H, W, car, make_tracks, piecewise

from roadsight.config import load_config
from roadsight.events import rules as _rules  # noqa: F401
from roadsight.events.base import RULES
from roadsight.io.video import VideoMeta
from roadsight.perception.signal import SignalTimeline
from roadsight.scene.auto import AutoScene
from roadsight.scene.scene import Lane, Scene, StopLine, Zone

CFG = load_config()["events"]
BL = float(np.sqrt(CAR[0] * CAR[1]))  # 1 body length in px


def run(label, tracks, scene=None, signal=None, duration=60.0):
    scene = scene or road_scene()
    scene.auto = AutoScene.build(tracks.df, W, H, load_config()["scene"])
    meta = VideoMeta("x.mp4", "x.mp4", 12.5, W, H, int(duration * 12.5))
    segs = RULES[label](CFG[label], scene).detect(tracks, signal or SignalTimeline(), meta)
    return [s for s in segs if s.confidence >= CFG[label].get("min_confidence", 0.5)]


def road_scene(**kw) -> Scene:
    s = Scene(W, H)
    s.carriageway = [np.array([[0, 250], [W, 250], [W, 600], [0, 600]], float)]
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_stopped_vehicle_with_passing_traffic():
    a = car(piecewise([(0, 100, 420), (3, 500, 420), (23, 500, 420), (26, 900, 420)]))
    b = car(piecewise([(8, 0, 440), (16, 1200, 440)]))
    segs = run("stopped_vehicle", make_tracks([(1, 2, a, 0, 26), (2, 2, b, 8, 16)], 30), duration=30)
    assert len(segs) == 1
    assert abs(segs[0].start - 3.0) < 1.0 and abs(segs[0].end - 23.0) < 1.5


def test_briefly_stopped_vehicle_is_ignored():
    a = car(piecewise([(0, 100, 420), (3, 500, 420), (8, 500, 420), (11, 900, 420)]))
    assert run("stopped_vehicle", make_tracks([(1, 2, a, 0, 11)], 15), duration=15) == []


def test_wrong_way_against_drawn_lane():
    lane = Lane("L1", np.array([[0, 250], [W, 250], [W, 600], [0, 600]], float), np.array([1.0, 0.0]))
    wrong = car(piecewise([(2, 1100, 420), (8, 300, 420)]))
    right = car(piecewise([(0, 100, 380), (8, 1100, 380)]))
    segs = run("wrong_way", make_tracks([(1, 2, wrong, 2, 8), (2, 2, right, 0, 8)], 10), road_scene(lanes=[lane]), duration=10)
    assert len(segs) == 1 and segs[0].info["tracks"] == [1]
    assert segs[0].start < 3.0 and segs[0].end > 7.0


def test_jaywalking_outside_crosswalk():
    person = lambda t: (*piecewise([(1, 640, 200), (7, 640, 650)])(t), 30.0, 80.0)  # noqa: E731
    segs = run("jaywalking", make_tracks([(1, 0, person, 1, 7)], 10), duration=10)
    assert len(segs) == 1
    assert 1.5 < segs[0].start < 3.5 and 4.5 < segs[0].end < 6.5


def test_crossing_at_crosswalk_is_legal():
    person = lambda t: (*piecewise([(1, 640, 200), (7, 640, 650)])(t), 30.0, 80.0)  # noqa: E731
    cw = [np.array([[590, 240], [690, 240], [690, 610], [590, 610]], float)]
    assert run("jaywalking", make_tracks([(1, 0, person, 1, 7)], 10), road_scene(crosswalks=cw), duration=10) == []


def test_congestion_long_queue():
    objs = []
    for i in range(10):
        x0 = 100 + i * 110
        objs.append((i + 1, 2, car(piecewise([(0, x0, 420), (80, x0 + 60, 420)])), 0, 80))
    segs = run("congestion", make_tracks(objs, 80), duration=80)
    assert len(segs) == 1 and segs[0].end - segs[0].start > 60


def test_accident_contact_and_stop():
    a = car(piecewise([(0, 200, 420), (5, 600, 420), (5.3, 640, 420), (20, 640, 420)]))
    b = car(piecewise([(0, 1100, 425), (5, 760, 425), (5.3, 700, 425), (20, 700, 425)]))
    segs = run("accident", make_tracks([(1, 2, a, 0, 20), (2, 2, b, 0, 20)], 20), duration=20)
    assert len(segs) >= 1
    assert abs(segs[0].start - 5.0) < 1.0


def test_smooth_stop_is_not_accident():
    a = car(piecewise([(0, 200, 420), (6, 600, 420), (9, 640, 420), (20, 640, 420)]))
    assert run("accident", make_tracks([(1, 2, a, 0, 20)], 20), duration=20) == []


def test_red_light_runner():
    line = StopLine("S1", np.array([[600, 250], [600, 600]], float), "sig", np.array([1.0, 0.0]))
    sig = SignalTimeline({"sig": [(0.0, 3.0, "green"), (3.0, 20.0, "red")]})
    a = car(piecewise([(0, 100, 420), (10, 1100, 420)]))  # crosses x=600 at ~t=5
    b = car(piecewise([(0, 100, 480), (2.5, 1100, 480)]))  # crosses on green
    segs = run("red_light", make_tracks([(1, 2, a, 0, 10), (2, 2, b, 0, 2.5)], 12), road_scene(stop_lines=[line]), sig, 12)
    assert len(segs) == 1 and 4.0 < segs[0].start < 6.0


def test_illegal_u_turn_in_zone():
    zone = [np.array([[400, 250], [900, 250], [900, 600], [400, 600]], float)]
    ts = np.linspace(0, np.pi, 30)
    keys = [(0, 100, 350)] + [(4 + 3 * k / np.pi, 650 + 100 * np.sin(k), 450 - 100 * np.cos(k)) for k in ts] + [(11, 100, 550)]
    a = car(piecewise(keys))
    segs = run("illegal_u_turn", make_tracks([(1, 2, a, 0, 11)], 12), road_scene(no_u_turn_zones=zone), duration=12)
    assert len(segs) == 1 and 3.0 < segs[0].start < 5.5


def test_solid_line_crossing():
    line = [np.array([[0, 400], [W, 400]], float)]
    a = car(piecewise([(0, 100, 370), (3, 500, 370), (5, 750, 460), (9, 1200, 460)]))
    segs = run("solid_line_crossing", make_tracks([(1, 2, a, 0, 9)], 10), road_scene(solid_lines=line), duration=10)
    assert len(segs) == 1 and 2.5 < segs[0].start < 5.0


def test_failure_to_yield():
    cw = [np.array([[590, 250], [690, 250], [690, 600], [590, 600]], float)]
    person = lambda t: (640.0, 300.0 + 25.0 * t, 30.0, 80.0)  # noqa: E731  walking across the zebra
    a = car(piecewise([(0, 100, 440), (8, 1200, 440)]))  # reaches the zebra as the walker is in its lane
    segs = run("failure_to_yield", make_tracks([(1, 2, a, 0, 8), (2, 0, person, 0, 8)], 10), road_scene(crosswalks=cw), duration=10)
    assert len(segs) == 1


def test_waiting_pedestrian_is_not_failure_to_yield():
    cw = [np.array([[590, 250], [690, 250], [690, 600], [590, 600]], float)]
    person = lambda t: (640.0, 380.0, 30.0, 80.0)  # noqa: E731  standing still
    a = car(piecewise([(0, 100, 440), (8, 1200, 440)]))
    assert run("failure_to_yield", make_tracks([(1, 2, a, 0, 8), (2, 0, person, 0, 8)], 10), road_scene(crosswalks=cw), duration=10) == []


def test_road_obstacle_animal():
    dog = lambda t: (500.0, 450.0, 40.0, 30.0)  # noqa: E731
    segs = run("road_obstacle", make_tracks([(1, 16, dog, 2, 9)], 10), duration=10)
    assert len(segs) == 1 and segs[0].end - segs[0].start > 5


def test_illegal_turn_movement_table():
    zones = [
        Zone("W", np.array([[0, 250], [200, 250], [200, 600], [0, 600]], float)),
        Zone("N", np.array([[500, 0], [800, 0], [800, 250], [500, 250]], float)),
    ]
    keys = (
        [(0, 50, 450), (3, 500, 450)]
        + [(3 + k, 500 + 150 * np.sin(k * np.pi / 6), 300 + 150 * np.cos(k * np.pi / 6)) for k in np.linspace(0, 3, 12)]
        + [(8, 650, 100)]
    )
    a = car(piecewise(keys))
    scene = road_scene(exit_zones=zones, allowed_movements={("N", "W")})
    segs = run("illegal_turn", make_tracks([(1, 2, a, 0, 8)], 10), scene, duration=10)
    assert len(segs) == 1


@pytest.mark.parametrize("label", sorted(RULES))
def test_rules_quiet_on_empty_tracks(label):
    tracks = make_tracks([(1, 2, car(piecewise([(0, 100, 420), (10, 1100, 420)])), 0, 10)], 12)
    if label in ("road_obstacle", "fire_smoke"):
        assert run(label, tracks, duration=12) == []
    else:
        assert run(label, tracks, duration=12) == [], label


def test_near_miss_hard_brake_without_contact():
    # Follower A at 3 BL/s closes on slow leader B and brakes hard at TTC ~0.3 s, stopping ~25 px short.
    a = car(piecewise([(0, 40, 420), (3.9, 847.3, 420), (4.2, 881.5, 420), (12, 1045.3, 420)]))
    b = car(piecewise([(0, 900, 420), (12, 1152, 420)]))
    segs = run("near_miss", make_tracks([(1, 2, a, 0, 12), (2, 2, b, 0, 12)], 12), duration=12)
    assert len(segs) == 1 and 2.5 < segs[0].start < 4.5


def test_steady_following_is_not_near_miss():
    a = car(piecewise([(0, 40, 420), (12, 1000, 420)]))
    b = car(piecewise([(0, 300, 420), (12, 1260, 420)]))
    assert run("near_miss", make_tracks([(1, 2, a, 0, 12), (2, 2, b, 0, 12)], 12), duration=12) == []
