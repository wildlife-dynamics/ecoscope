import json
import logging
from typing import Annotated, Literal, cast

import matplotlib.pyplot as plt
import pandas as pd
from ecoscope.platform.annotations import AdvancedField, AnyDataFrame
from pydantic import Field
from wt_registry import register

logger = logging.getLogger(__name__)


@register()
def assign_subject_colors(
    df: AnyDataFrame,
    subject_id_column: Annotated[
        str,
        Field(description="Column containing subject identifiers (e.g., 'groupby_col', 'subject__id')"),
    ] = "subject__id",
    additional_column: Annotated[
        str,
        Field(description="Column containing subject additional data as JSON (e.g., 'subject__additional')"),
    ] = "subject__additional",
    output_column: Annotated[
        str,
        Field(description="Name of the output column for assigned colors"),
    ] = "subject_color",
    fallback_strategy: Annotated[
        Literal["default_color", "palette"],
        Field(
            description="Strategy for subjects with missing or duplicate rgb values: "
            "'default_color' keeps original rgb (even duplicates) and uses default_color for missing; "
            "'palette' assigns palette colors to both duplicates and missing"
        ),
    ] = "default_color",
    default_color: Annotated[
        str,
        AdvancedField(
            description="Hex color for subjects without rgb (used when fallback_strategy='default_color')",
            default="#FFFF00",
        ),
    ] = "#FFFF00",
    default_palette: Annotated[
        str,
        AdvancedField(
            description="Color palette for fallback colors (used when fallback_strategy='palette')",
            default="tab20",
        ),
    ] = "tab20",
) -> AnyDataFrame:
    """
    Assign colors to subjects based on rgb field from subject__additional JSON.

    Strategy:
    1. Parse rgb from subject__additional JSON field
    2. Identify subjects with unique vs duplicate rgb values
    3. Based on fallback_strategy:
       - 'default_color': Keep original rgb (even duplicates), use default_color for missing
       - 'palette': Only unique rgb kept, duplicates and missing get palette colors
    4. Return dataframe with new color column

    Args:
        df: Input dataframe with subject observations
        subject_id_column: Column name containing subject identifiers
        additional_column: Column name containing JSON with rgb data
        output_column: Name for the output color column
        fallback_strategy: Strategy for handling duplicates and missing rgb values
        default_color: Hex color string for missing rgb (when fallback_strategy='default_color')
        default_palette: Matplotlib palette name for fallback colors (when fallback_strategy='palette')

    Returns:
        DataFrame with added color column
    """
    if subject_id_column not in df.columns:
        raise ValueError(f"Subject ID column '{subject_id_column}' not found in dataframe")

    # Define NAN_COLOR to match apply_color_map behavior
    NAN_COLOR = (0, 0, 0, 0)

    # Step 1: Parse rgb values from subject__additional
    subject_rgb_map = {}

    if additional_column in df.columns:
        for subject_id in df[subject_id_column].unique():
            subject_rows = df[df[subject_id_column] == subject_id]
            # Get the first non-null additional data for this subject
            for additional_data in subject_rows[additional_column]:
                if pd.notna(additional_data):
                    try:
                        if isinstance(additional_data, str):
                            additional_dict = json.loads(additional_data)
                        elif isinstance(additional_data, dict):
                            additional_dict = additional_data
                        else:
                            continue

                        rgb_value = additional_dict.get("rgb")
                        if rgb_value:
                            subject_rgb_map[subject_id] = rgb_value
                            break
                    except (json.JSONDecodeError, AttributeError) as e:
                        logger.warning(f"Failed to parse additional data for subject {subject_id}: {e}")
                        continue
    else:
        logger.warning(f"Column '{additional_column}' not found in dataframe. All subjects will use palette colors.")

    # Step 2: Identify duplicate rgb values
    rgb_counts = pd.Series(subject_rgb_map).value_counts()
    duplicate_rgb_values = set(rgb_counts[rgb_counts > 1].index)

    # Helper functions
    def hex_to_rgba(hex_color: str) -> tuple[float, float, float, float]:
        hex_color = hex_color.lstrip("#")
        r, g, b = (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
        return (r / 255.0, g / 255.0, b / 255.0, 1.0)

    def parse_rgb_str(rgb_str: str) -> tuple[float, float, float, float] | None:
        try:
            r, g, b = [int(x.strip()) for x in rgb_str.split(",")]
            return (r / 255.0, g / 255.0, b / 255.0, 1.0)
        except (ValueError, AttributeError):
            return None

    default_color_rgba = hex_to_rgba(default_color)

    # Step 3: Get palette colors (only if needed)
    palette_colors = None
    if fallback_strategy == "palette":
        try:
            cmap = plt.get_cmap(default_palette)
            palette_colors = [cmap(i) for i in range(cmap.N)]
        except ValueError:
            logger.warning(f"Palette '{default_palette}' not found, falling back to 'tab20'")
            cmap = plt.get_cmap("tab20")
            palette_colors = [cmap(i) for i in range(cmap.N)]

    # Step 4: Build final color mapping
    final_color_map = {}
    subjects_needing_fallback = []

    for subject_id in df[subject_id_column].unique():
        rgb_str = subject_rgb_map.get(subject_id)

        if rgb_str is None:
            subjects_needing_fallback.append(subject_id)
            continue

        is_duplicate = rgb_str in duplicate_rgb_values
        if is_duplicate and fallback_strategy == "palette":
            subjects_needing_fallback.append(subject_id)
            continue

        parsed = parse_rgb_str(rgb_str)
        if parsed:
            final_color_map[subject_id] = parsed
        else:
            logger.warning(f"Failed to parse rgb '{rgb_str}' for subject {subject_id}.")
            subjects_needing_fallback.append(subject_id)

    # Assign fallback colors
    for idx, subject in enumerate(subjects_needing_fallback):
        if fallback_strategy == "palette" and palette_colors:
            final_color_map[subject] = palette_colors[idx % len(palette_colors)]
        else:
            final_color_map[subject] = default_color_rgba

    # Step 5: Apply color mapping to dataframe, using NAN_COLOR for null subjects
    def apply_color(subject_id):
        if pd.isna(subject_id):
            return NAN_COLOR
        color = final_color_map.get(subject_id)
        if color is None:
            return NAN_COLOR
        # Convert from 0-1 range to 0-255 range, keeping as tuple
        return tuple(round(chan * 255) for chan in color)

    df[output_column] = df[subject_id_column].apply(apply_color)
    return cast(AnyDataFrame, df)
