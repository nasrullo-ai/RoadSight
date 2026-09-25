"""Importing this package registers every rule in ``roadsight.events.base.RULES``."""

from roadsight.events.rules import (  # noqa: F401
    accident,
    congestion,
    failure_to_yield,
    fire_smoke,
    jaywalking,
    near_miss,
    red_light,
    road_obstacle,
    solid_line_crossing,
    stop_line,
    stopped_vehicle,
    turns,
    wrong_way,
)
