import { BaseModel } from "./model";
import {TitleWidget} from "./title-widget"

export class TitleWidgetModel extends BaseModel{

    render() {
      return new TitleWidget({
        id:  this.widgetModel.get("id"), 
        title: this.widgetModel.get("title"), 

        style: {
          'fontSize': this.widgetModel.get("font_size"),
          'fontStyle': this.widgetModel.get("font_style"),
          'fontFamily': this.widgetModel.get("font_family"),
          'color': this.widgetModel.get("font_color"),
          'backgroundColor': this.widgetModel.get("background_color"),
          'outline': this.widgetModel.get("outline"),
          'borderRadius': this.widgetModel.get("border_radius"),
          'border': this.widgetModel.get("border"),
          'padding': this.widgetModel.get("padding"),
        }
      })
    }
  }