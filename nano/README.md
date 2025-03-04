## Setup
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Runing an experiment
```
./run_exp.py --config-path <PATH_TO_FOLDER_WITH_CONFIG> --config-name <NAME_OF_THE_CONFIG>
```

## Building singularity image
```
./scripts/build_image.sh
```


# Pipeline stages
- hydra generates final config
- this config is splitted into separate files according to grid search character '^'
- configs are saved under <OUTPUT_DIR>/config_<1:N>.yaml
- slurm sbatch script is created to run all configs as one job with an task array 

