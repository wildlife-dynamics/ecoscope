import { BaseModel } from './model';
import { GeoJsonLayer } from '@deck.gl/layers';
import { WidgetModel } from "@jupyter-widgets/base";
import { parse } from '@loaders.gl/core';
import { JSONLoader } from '@loaders.gl/json';
import { GeoJSON } from 'geojson';

export class GeoJsonLayerModel extends BaseModel{
  
    render() {
      return new GeoJsonLayer({
        id: this.widgetModel.model_id,
        data: parse(this.widgetModel.get("data"), JSONLoader),
        visible: this.widgetModel.get("visible"),
        opacity: this.widgetModel.get("opacity"),
        pointType: this.widgetModel.get("point_type"),
      });
    }
  }