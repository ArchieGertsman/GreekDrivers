# Greek Drivers (an IGL project)

## Contributors
Undergraduates:
* Archie Gertsman (arkadiy2@illinois.edu)
* Anirudh Eswara (aeswara2@illinois.edu)
* Ridha Alkhabaz (ridhama2@illinois.edu)
* Sheil Kumar (sk17@illinois.edu)

Graduate lead: [Lloyd Fernandes](https://www.linkedin.com/in/fernandeslloyd/) (lloydf2@illinois.edu )

Project director: Richard Sowers
* <r-sowers@illinois.edu>
* <https://publish.illinois.edu/r-sowers/>

Copyright 2021 University of Illinois Board of Trustees. All Rights Reserved. Licensed under the MIT license

## Project overview
The recently-release pNeuma Dataset (at https://open-traffic.epfl.ch/) contains GPS traces of Athens traffic, annotated with vehicle type.  We would like to see if we could algorithmically distinguish between car and taxi drivers by "driving styles"

### Relevant Publications
* https://doi.org/10.1016/j.trc.2017.11.021
* https://doi.org/10.1016/j.trc.2020.102644
* https://doi.org/10.1016/j.trc.2018.03.024
* https://doi.org/10.1016/j.trc.2013.09.015


## File structure:

* **Data**: where small samples of the raw data and their processed (through Lib/data_loader) pickled counterparts are stored

* **Lib**
	* data_loader.py: `csv_to_df` function loads a raw csv (e.g. Data/sample.csv) into a multiindexed Pandas dataframe
	* feature_eng.py: provides functions which construct new features and add them to a dataframe which was processed by `csv_to_df`, e.g. `cross_track` computes the cross track distance of a vehicle at each time step.

* **Analysis**
	* ML: 
		* classical_models.ipynb: training/testing classical models
		* TSNE_Analysis_ files: unsupervised methods
		* lstm.py: training an LSTM model
	* Performance: different methods of computation are compared for performance, e.g. how to find nearest road to vehicle
	* Old: previous work from Shekhar Sharma

	* descriptive_statistics.ipynb: descriptive statistics of Data/block_1_sample_df.pkl

* **Documents** contains reports

## Results
* classical methods are able to distinguish between cars and taxis with over 70% accuracy (see Analysis/ML/classical_models.ipynb for visualization)
* methods using lstm's are able to distinguish cars and taxis with approximately 61% accuracy.
