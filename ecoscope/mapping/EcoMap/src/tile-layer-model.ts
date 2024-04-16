import { BaseModel } from './model';
import { TileLayer } from '@deck.gl/geo-layers';
import { BitmapLayer } from '@deck.gl/layers';

export class TileLayerModel extends BaseModel{

    render() {
      return new TileLayer({
        id: this.widgetModel.model_id,
        data: this.widgetModel.get("data"),
        visible: this.widgetModel.get("visible"),
        opacity: this.widgetModel.get("opacity"),
        minZoom: this.widgetModel.get("min_zoom"),
        maxZoom: this.widgetModel.get("max_zoom"),
        tileSize: this.widgetModel.get("tile_size"),
  
        renderSubLayers: props => {
          const [min, max] = props.tile.boundingBox 
  
          return new BitmapLayer(props, {
            data: undefined,
            image: props.data,
            bounds:  [min[0], min[1], max[0], max[1]]
          });
        } 
      });
    }
  }