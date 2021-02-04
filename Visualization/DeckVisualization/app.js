import React, { useEffect } from 'react';
import styles from './App.module.css';
import {render} from 'react-dom';
import {StaticMap} from 'react-map-gl';
import DeckGL, {TripsLayer} from 'deck.gl';
import athens from './athens.json';

const MAPBOX_ACCESS_TOKEN = "pk.eyJ1IjoicmljaGFyZHNvd2VycyIsImEiOiJjams5dnZtcWYya2RxM3FxcW03YzF0a2pzIn0._YTRBqzmI1sZnBONyEBr3A";

const INITIAL_VIEW_STATE = {
  latitude: 37.977482,
  longitude: 23.732405,
  zoom: 16,
  bearing: 0,
  pitch: 0,
};

const MAP_STYLE = 'mapbox://styles/richardsowers/cjohs10mm1f9m2sltnluju4fm';

for (var feat of athens['features']) {
  feat['times'] = [...Array(feat['geometry']['coordinates'].length).keys()];
}

console.log(athens);

function Root() {
  const [viewState, setViewState] = React.useState(INITIAL_VIEW_STATE)
  const handleChangeViewState = ({ viewState }) => setViewState(viewState);
  
  const onClick = info => {
    if (info.object) {
      // eslint-disable-next-line
      alert(`${info.object.properties.name} (${info.object.properties.abbrev})`);
    }
  };
  const [currentTime,setCurrentTime] = React.useState(0);
  useEffect(() => {
    var timerID = setInterval( () => tick(), 10);

    return function cleanup() {
      clearInterval(timerID);
    }
  });
  function tick() {
    setCurrentTime(currentTime => (currentTime + 1) % 1000);
    setLayers(layers => layers.map(layer => layer.clone({currentTime:currentTime})));
  }
  
  const [layers, setLayers] = React.useState([
    new TripsLayer({
      id: 'trips-layer' ,
      data: athens['features'],
      // Styles
      getColor: d => d.properties.color,//[253,128,93],
      rounded: true,
      trailLength: 200,
      widthMinPixels: 5,
      getPath: d => d.geometry.coordinates,
      getTimestamps: d => d.times,
      visible: true,
      currentTime: currentTime,
      // Interactive props
      pickable: true,
      autoHighlight: true,
      onClick
    }),
  ]);

  
  return (
    <div>
    <DeckGL id='deck' viewState={viewState} controller={true} layers={layers} onViewStateChange={handleChangeViewState}>
      <StaticMap  mapStyle={MAP_STYLE} mapboxApiAccessToken={MAPBOX_ACCESS_TOKEN} />
    </DeckGL>
    </div>
  );
}

/* global document */
render(<Root />, document.body.appendChild(document.createElement('div')));