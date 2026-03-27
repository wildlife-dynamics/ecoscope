# Custom Task Patterns

In [Your First Custom Task](./first-custom-task.md) you wrote a basic task with simple types. Now let's use the Platform SDK's type system and testing tools for richer integration with Desktop.

---

## Ecoscope-specific type annotations

The Platform SDK provides specialized types that give Desktop more context about your parameters. Import them from `ecoscope.platform`:

### DataFrame types

Instead of plain `pandas.DataFrame`, use the Platform SDK's annotated DataFrame types:

```python
from ecoscope.platform.annotations import AnyGeoDataFrame, AnyDataFrame

@register()
def filter_by_distance(
    geodataframe: Annotated[AnyGeoDataFrame, Field(description="Input geodataframe", exclude=True)],
    max_distance_km: Annotated[float, Field(description="Maximum distance in km")] = 100.0,
) -> Annotated[AnyGeoDataFrame, Field()]:
    ...
```

`AnyGeoDataFrame` tells the compiler that this parameter expects geospatial data with a geometry column. `AnyDataFrame` is for non-spatial DataFrames.

For stricter validation, use **Pandera schemas**:

```python
from ecoscope.platform.schemas import EventGDFSchema
from pandera.typing import DataFrame

@register()
def count_event_types(
    df: Annotated[DataFrame[EventGDFSchema], Field(exclude=True)],
) -> Annotated[int, Field()]:
    return df["event_type"].nunique()
```

The schema validates at runtime that the DataFrame has the expected columns and types. See the [schemas reference](../reference/schemas.md).

### AdvancedField

Use `AdvancedField` to hide a parameter behind the "Advanced" toggle in Desktop's form. This is useful for parameters that most users should not need to change:

```python
from ecoscope.platform.annotations import AdvancedField

@register()
def draw_custom_chart(
    dataframe: Annotated[AnyDataFrame, Field(exclude=True)],
    title: Annotated[str, Field(description="Chart title")],
    color_scheme: Annotated[str, AdvancedField(description="Color scheme")] = "default",
) -> Annotated[str, Field()]:
    ...
```

See the [annotations reference](../reference/annotations.md).

### Connection protocol types

If your task needs to call an external API, type the parameter with a connection protocol:

```python
from ecoscope.platform.connections import EarthRangerClient

@register()
def get_custom_data(
    client: Annotated[EarthRangerClient, Field(exclude=True)],
    category: Annotated[str, Field(description="Event category to fetch")],
) -> Annotated[AnyGeoDataFrame, Field()]:
    ...
```

Desktop will render a data-source picker for `EarthRangerClient` parameters. Available protocols include `EarthRangerClient`, `SmartClient`, and `EarthEngineClient`. See the [connections reference](../reference/connections.md).

---

## Testing with mock_loaders

The `test-cases.yaml` file lets you run workflows locally without a live data source. Each test case provides mock return values for tasks that would normally call an external API.

### `test-cases.yaml` format

```yaml
- test_name: "basic"
  description: "Basic test with sample events"
  params:
    set_time_range:
      start: "2024-01-01"
      end: "2024-01-31"
  mock_tasks:
    get_events:
      return:
        loader: "parquet"
        path: "tests/sample_events.parquet"
```

The `loader: "parquet"` entry uses the Platform SDK's built-in parquet mock loader, which reads `.parquet` or `.geoparquet` files and returns a DataFrame or GeoDataFrame as appropriate. See the [mock_loaders reference](../reference/mock_loaders.md).

### Running tests

After compiling, run:

```console
$ cd ecoscope-workflows-events-example-workflow
$ pixi run test
```

This executes each test case defined in `test-cases.yaml`. Results snapshots are saved to `__results_snapshots__/` for comparison.

---

## Next steps

- **[Data Sources](./data-sources.md)** — Learn the connection and environment variable patterns.
- **[Form Customization](./form-customization.md)** — Control how your task's parameters appear in the Desktop form.
