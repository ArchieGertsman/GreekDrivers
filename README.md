### Title ###
by: Author
* author persistent email
* author persistent webpage

Project director: Richard Sowers
* <r-sowers@illinois.edu>
* <https://publish.illinois.edu/r-sowers/>

Copyright 2019 University of Illinois Board of Trustees. All Rights Reserved. Licensed under the MIT license

### Explanation of repository:
what is the point of the repository

### File structure:  help the user out
* file structure
* how to run the code

* **data**: where the raw(ish) data is stored
	* XYZ.csv:  original dataset (always keep the original dataset somewhere)
	* XYZ.p: pickled version of XYZ.csv, with date-times converted to python timezone-aware datetimes.  For development, XYZ.p (and thus XYZ.csv) should be small enough that ALL processing directly uses XYZ.p.
	This directory should/can be write-protected once the original data processing is one

* **images** keep images here.  Label them
* **analysis** keep any temporary .csv files here

* pickler.ipnyb:  makes data/XYZ.p out of data/XYZ.csv
* descriptive_statistics.ipynb: descriptive statistics of data/XYZ.p
	* first several rows
	* number of rows
	* min and max dates
	* min, max, mean, and stdev of relevant numerical columns
	* number of unique values of categorical data
	* plots of data to show
		* normality
		* correlatedness




Other resources at
* <https://www.makeareadme.com/>
* <https://help.github.com/articles/about-project-boards/>