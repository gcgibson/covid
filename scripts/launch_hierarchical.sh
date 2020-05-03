#!/bin/bash

sbatch --job-name=$state \
      --output=log/result-$state.out \
      --error=log/result-$state.err \
      --nodes=1 \
      --ntasks=1 \
      --mem=5000 \
      --partition=longq \
      --time=20:00:00 \
       ./run_hierarchical.sh
