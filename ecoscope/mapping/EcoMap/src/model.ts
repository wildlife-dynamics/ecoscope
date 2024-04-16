import { Layer, Widget } from "@deck.gl/core";
import { WidgetModel } from "@jupyter-widgets/base";

export abstract class BaseModel {

    widgetModel: WidgetModel;
    constructor(widgetModel: WidgetModel) {
        this.widgetModel = widgetModel;
    }

    abstract render();
  }