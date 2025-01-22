#!/bin/bash

snakemake -s Snakefile_image_registration.py --cluster "sbatch --ntasks=1 --job-name={params.name} --time={params.time} --mail-type=FAIL --cpus-per-task={threads} --partition={params.partition} --mem={params.mem} -o /oak/stanford/groups/quake/ejerison/lps_phases/in_situ/data_processing/logs/{params.name}.%j.log" --jobs 200 --rerun-incomplete -p
