import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import DeckGL from '@deck.gl/react';
import { Widget, Layer, MapViewState } from '@deck.gl/core'
import { WidgetModel} from "@jupyter-widgets/base";
import {TileLayerModel} from './tile-layer-model.js'
import {GeoJsonLayerModel} from './geojson-layer-model.js'
import {TitleWidgetModel} from './widget-model.js'
import { BaseModel } from "./model.js";
import type { IWidgetManager } from "@jupyter-widgets/base";

async function loadModels(widgetManager: IWidgetManager, layerIds: string[]){
  const resolvedModels: Promise<WidgetModel>[] = [];
  layerIds.forEach(layerId => {

    resolvedModels.push(widgetManager.get_model(layerId.slice("IPY_MODEL_".length)));

  });
  //so that the models are returned and not the promises
  return await Promise.all(resolvedModels);
}

async function loadControls(widgetManager: IWidgetManager, controlIds: string[]){
  const resolvedControls: Promise<WidgetModel>[] = [];
  controlIds.forEach(controlId => {

    resolvedControls.push(widgetManager.get_model(controlId.slice("IPY_MODEL_".length)));

  });
  //so that the models are returned and not the promises
  return await Promise.all(resolvedControls);
}

function App() {
  let [getWidth] = useModelState<number>("width");
  let [getHeight] = useModelState<number>("height");
  let [getController] = useModelState<boolean>("controller");
  let [getInitialViewState] = useModelState<MapViewState>("initial_view_state");

  let [layerIds] = useModelState<string[]>("layers");

  let [controlIds] = useModelState<string[]>("controls");

  const manager: IWidgetManager = useModel().widget_manager;

  let [layers, setLayers] = React.useState<Record<string, Layer>>({});
  let [mapControls, setMapControls] = React.useState<Record<string, Widget>>({});

  React.useEffect(() => {
    const callback = async () => {
      const resolvedModels: WidgetModel[] =  await loadModels(manager, layerIds);
      const resolveControlModels: WidgetModel[] = await loadControls(manager, controlIds);
      //Init / render layers
      const newLayers: Record<string, Layer<any>> = { ...layers };

      resolvedModels.forEach((model: WidgetModel) =>{
        let newModel: BaseModel;
        const layerType = model.get("_layer_type");
        const layerId = model.model_id;

        if (!(layerId in layers)){
          if ("TileLayer" == layerType){
            newModel = new TileLayerModel(model);
            //layers.push(new TileLayerModel(model).render());
          }
          else if ("GeoJsonLayer" == layerType){
            newModel = new GeoJsonLayerModel(model);
            //layers.push(new GeoJsonLayerModel(model).render());
          }
          else {throw new Error(`${layerType} not supported`);}

          newLayers[layerId] = newModel.render();
        }
      });
    
      setLayers(newLayers);

      //Init / render map controls
      const newControls: Record<string, Widget<any>> = { ...mapControls };
      resolveControlModels.forEach((model: WidgetModel) =>{

        let newControl: BaseModel;
        const controlType = model.get("_control_type");
        const controlId = model.model_id;
        if (!(controlId in mapControls)){
          if ("TitleControl" == controlType){
            newControl = new TitleWidgetModel(model);
          }
          else {throw new Error(`${controlType} not supported`);}

          newControls[controlId] = newControl.render();
        }
      });
      setMapControls(newControls);
    };
    callback().catch(console.error);
  }, [layerIds]);


  let drawLayers: Layer[] = [];
  for (const layer in layers){
    drawLayers.push(layers[layer]);
  }

  let drawControls: Widget[] = [];
  for (const control in mapControls){
    drawControls.push(mapControls[control]);
  }
  
  //Change id 
  return (
    <div id="EcoMap" style={{ height: getHeight, width: getWidth }}>
      <DeckGL
          initialViewState={getInitialViewState}
          controller={getController}
          width={getWidth}
          height={getHeight}
          layers={drawLayers} 
          widgets={drawControls}/>
      </div>
  );
}

export let render = createRender(App);
