#!/bin/bash
#SBATCH --job-name=PDS4
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

nvidia-smi

./out/gpu 1 4 datasets/auto/auto.mtx
./out/gpu 1 4 datasets/delaunay_n22/delaunay_n22.mtx
./out/gpu 1 4 datasets/great-britain_osm/great-britain_osm.mtx

./out/gpu 1 8 datasets/auto/auto.mtx
./out/gpu 1 8 datasets/delaunay_n22/delaunay_n22.mtx
./out/gpu 1 8 datasets/great-britain_osm/great-britain_osm.mtx

./out/gpu 1 16 datasets/auto/auto.mtx
./out/gpu 1 16 datasets/delaunay_n22/delaunay_n22.mtx
./out/gpu 1 16 datasets/great-britain_osm/great-britain_osm.mtx

./out/gpu 1 32 datasets/auto/auto.mtx
./out/gpu 1 32 datasets/delaunay_n22/delaunay_n22.mtx
./out/gpu 1 32 datasets/great-britain_osm/great-britain_osm.mtx

./out/gpu 2 4 datasets/auto/auto.mtx
./out/gpu 2 4 datasets/delaunay_n22/delaunay_n22.mtx
./out/gpu 2 4 datasets/great-britain_osm/great-britain_osm.mtx

./out/gpu 2 8 datasets/auto/auto.mtx
./out/gpu 2 8 datasets/delaunay_n22/delaunay_n22.mtx
./out/gpu 2 8 datasets/great-britain_osm/great-britain_osm.mtx