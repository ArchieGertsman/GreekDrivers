# reverse_geocode

# By Richard Sowers
# <r-sowers@illinois.edu>
# <https://publish.illinois.edu/r-sowers/>

# Copyright 2020 University of Illinois Board of Trustees.
# All Rights Reserved. Licensed under the MIT license

# loads OSMNX network and then reverse geocodes

# set up via:
#   conda config --prepend channels conda-forge
#   conda create -n ox --strict-channel-priority osmnx
# and then
#   conda activate ox
#	reverse_geocode.py 
# after;
#   conda deactivate

import osmnx
import sys

place="Champaign, IL, USA"
if __name__ == "__main__": 
    try:
    	place=sys.argv[1]
    except:
    	pass

print("place="+str(place))

osmnx.config(log_file=True, timeout=360,log_console=True, use_cache=True)
G= osmnx.graph_from_place(place, simplify=True, network_type='drive')

def getlinks(point):
    return osmnx.distance.get_nearest_edge(G,point)

def getnode(point):
	return osmnx.distance.get_nearest_node(G,point)

if __name__ == "__main__": 
    point=(40.0771941171838, -88.293878452)
    print(getlinks(point))
    print(getnode(point))