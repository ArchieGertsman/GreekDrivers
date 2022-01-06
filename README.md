# Greek Drivers (an IGL project)

## Credits
- Contributors: 
	- Archie Gertsman (arkadiy2@illinois.edu)
	- Lloyd Fernandes (lloydf2@illinois.edu)
- Project director: Richard Sowers (r-sowers@illinois.edu, https://publish.illinois.edu/r-sowers/)
- Copyright: Copyright 2019 University of Illinois Board of Trustees. All Rights Reserved. 
- License: MIT

## Project overview
The recently-release pNeuma Dataset (at https://open-traffic.epfl.ch/) contains GPS traces of Athens traffic, annotated with vehicle type.  We would like to see if we could algorithmically distinguish between car and taxi drivers by "driving styles"


## File structure:

```bash
.

├── README.md # you are here

├── src # source code that powers the ml
│   ├── data_loader.py
│   ├── feature_eng.py
│   ├── modeling_helpers.py
│   └── tuning_objective.py

├── ml # data prep, training, testing, tuning, etc.
│   ├── data_processing.ipynb
│   ├── optuna_tuning.py
│   ├── permutation_feature_importance.ipynb
│   └── simple_workflow.ipynb

├── analysis # descriptive stats, data visualization
│   ├── stats 
│   │   └── descriptive_stats.ipynb
│   └── vis
│       ├── code
│       │   ├── map_maker.ipynb
│       │   └── map_maker_selected_seg.ipynb
│       ├── data
│       │   ├── 20181024_d1_0830_0900.csv
│       │   └── edge_data_block4.csv
│       ├── img
│       │   ├── Athens.png
│       │   ├── Athens_closeup.png
│       │   ├── edge_id.png
│       │   └── roadstructure.png
│       ├── tex
│       │   ├── aggregate.tex
│       │   ├── firstlines.tex
│       │   ├── observationframe.tex
│       │   └── ways.tex

```

## Results
