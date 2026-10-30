from dataclasses import dataclass
from typing import Annotated

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register
from wt_task import task

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope.platform.tasks.analysis._create_meshgrid import (
    AoiAnnotation,
    IntersectingOnlyAnnotation,
    create_meshgrid,
)
from ecoscope.platform.tasks.analysis._patrol_summary import (
    _EVENT_AGGREGATION_EXTRA,
    CountAggregation,
    SumOfColumnAggregation,
)
from ecoscope.platform.tasks.analysis._time_density import (
    AutoScaleGridCellSize,
    AutoScaleOrCustomAnnotation,
    CrsAnnotation,
    CustomGridCellSize,
)
from ecoscope.platform.tasks.results._map_utils import (
    DEFAULT_TILE_LAYER_PRESETS,
    TileLayer,
    make_preset_or_custom_json_schema_extra,
)

_base_maps_preset_extra = make_preset_or_custom_json_schema_extra(TileLayer, DEFAULT_TILE_LAYER_PRESETS)


def _advanced_base_maps_json_schema_extra(schema: dict) -> None:
    """Preset/custom tile-layer variants, rendered inside the Advanced accordion."""
    _base_maps_preset_extra(schema)
    schema["ecoscope:advanced"] = True


@dataclass
class EncounterRateMapOptions:
    aggregation: CountAggregation | SumOfColumnAggregation
    auto_scale_or_custom_cell_size: AutoScaleGridCellSize | CustomGridCellSize | None
    crs: str
    intersecting_only: bool
    threshold_km: float
    base_maps: list[TileLayer]

    def get_meshgrid_params(self):
        return {
            "auto_scale_or_custom_cell_size": self.auto_scale_or_custom_cell_size,
            "crs": self.crs,
            "intersecting_only": self.intersecting_only,
        }

    def get_agg_column(self) -> str:
        if isinstance(self.aggregation, SumOfColumnAggregation):
            return self.aggregation.column
        return ""


_OptionsParam = Annotated[EncounterRateMapOptions, Field(title="")]


@register()
def set_encounter_rate_map_options(
    aggregation: Annotated[
        CountAggregation | SumOfColumnAggregation | SkipJsonSchema[None],
        Field(default=None, json_schema_extra=_EVENT_AGGREGATION_EXTRA),
    ] = None,
    # Advanced-accordion field order follows signature order: threshold first.
    threshold_km: Annotated[
        float,
        AdvancedField(
            default=0.2,
            title="Minimum Patrol Effort Threshold (km)",
            description="Grid cells with less patrol effort than this render grey.",
            ge=0,
        ),
    ] = 0.2,
    auto_scale_or_custom_cell_size: AutoScaleOrCustomAnnotation = None,
    crs: CrsAnnotation = "EPSG:3857",
    intersecting_only: IntersectingOnlyAnnotation = False,
    base_maps: Annotated[
        list[TileLayer] | SkipJsonSchema[None],
        Field(
            default=None,
            title="Map Base Layers",
            description=(
                "Select tile layers to use as base layers in map outputs."
                " The first layer in the list will be the bottommost layer displayed."
            ),
            json_schema_extra=_advanced_base_maps_json_schema_extra,
        ),
    ] = None,
) -> EncounterRateMapOptions:
    """
    Combined options for the encounter rate map card: aggregation mode
    (count events vs. sum an event column), meshgrid options, low-effort
    mask threshold, and map base layers — one card, one Advanced accordion.
    """
    if aggregation is None:
        aggregation = CountAggregation()
    if base_maps is None:
        base_maps = [
            TileLayer(layer_name="TERRAIN"),
            TileLayer(layer_name="SATELLITE", opacity=0.5),
        ]
    return EncounterRateMapOptions(
        aggregation=aggregation,
        auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
        crs=crs,
        intersecting_only=intersecting_only,
        threshold_km=threshold_km,
        base_maps=base_maps,
    )


@register()
def get_agg_column_from_map_options(options: _OptionsParam) -> str:
    """Chosen sum column, or "" in Count mode."""
    return options.get_agg_column()


@register()
def call_meshgrid_from_map_options(
    aoi: AoiAnnotation,
    options: _OptionsParam,
) -> AnyGeoDataFrame:
    return task(create_meshgrid).validate().call(aoi=aoi, **options.get_meshgrid_params())


@register()
def get_mask_threshold_from_map_options(options: _OptionsParam) -> float:
    return options.threshold_km


@register()
def get_base_maps_from_map_options(options: _OptionsParam) -> list[TileLayer]:
    return options.base_maps
