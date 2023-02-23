#!/bin/bash
#SBATCH --job-name=PDS4_BASE
#SBATCH --partition=batch             # Submit to queue/partition named batch
#SBATCH --time=01:00:00               # Run time (days-hh:mm:ss) - (max 7days) 

./fglt/build/fglt datasets/auto/auto.mtx