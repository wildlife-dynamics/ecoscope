import random
from dataclasses import dataclass

from shapely.geometry import Point, Polygon


@dataclass
class BoundingBox:
    min_x: float
    max_x: float
    min_y: float
    max_y: float


# EPSG 3857 as it allows us to define ranges in metres
EPSG_3857_BOUNDS = BoundingBox(
    min_x=-20037508.34,
    max_x=20037508.34,
    min_y=-20037508,
    max_y=20048966.1,
)

# In places like create_meshgrid where there's a conversion to UTM,
# we want to restrict to ranges valid in that CRS
EPSG_3857_BOUNDS_UTM_SAFE = BoundingBox(
    min_x=-20037508.34,
    max_x=20037508.34,
    min_y=-15538710,
    max_y=18807213,
)


def random_point_in_bounds(bounds: BoundingBox | Polygon):
    if isinstance(bounds, Polygon):
        bounds = BoundingBox(
            min_x=bounds.bounds[0],
            max_x=bounds.bounds[2],
            min_y=bounds.bounds[1],
            max_y=bounds.bounds[3],
        )
    return Point(
        random.uniform(bounds.min_x, bounds.max_x),
        random.uniform(bounds.min_y, bounds.max_y),
    )


def random_3857_rectangle(
    width_min: float,
    width_max: float,
    height_min: float,
    height_max: float,
    utm_safe: bool,
):
    bounds = EPSG_3857_BOUNDS_UTM_SAFE if utm_safe else EPSG_3857_BOUNDS
    south_west = random_point_in_bounds(bounds)
    south_east = Point(
        min(south_west.x + random.uniform(width_min, width_max), bounds.max_x),
        south_west.y,
    )
    north_east = Point(
        south_east.x,
        max(south_west.y - random.uniform(height_min, height_max), -bounds.max_y),
    )
    north_west = Point(
        south_west.x,
        north_east.y,
    )

    return Polygon([south_west, south_east, north_east, north_west])


def random_points_in_bounds(bounds: BoundingBox | Polygon | None, num_points: int):
    points = []
    for i in range(num_points):
        points.append(random_point_in_bounds(bounds))  # type: ignore[arg-type]
    return points
