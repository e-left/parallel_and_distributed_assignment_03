#!/bin/bash
#SBATCH --job-name=PDS4_BASE
#SBATCH --partition=batch             
#SBATCH --time=01:00:00                

./fglt/build/fglt datasets/auto/auto.mtx
./fglt/build/fglt datasets/delaunay_n22/delaunay_n22.mtx
./fglt/build/fglt datasets/great-britain_osm/great-britain_osm.mtx