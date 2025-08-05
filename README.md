# GCRL-Landscapes

TODO: introduction to what this project is  
TODO: recommended way to install  

Convergence data for the algorithms to set proper phases can be found on [Huggingface](https://huggingface.co/datasets/jmtoepperwien/GCRL-Landscapes).

Training is supposed to run on a Slurm cluster.

## Training

### Phased HPO

This runs SMAC on a slurm cluster to generate importance data using multiple phases during training. Additionally one gets a tuned hyperparameter-configuration.  
Look at `hpo.sh` and optionally the configuration files in `configs` to get running. Analyze gathered data with [DeepCAVE](https://github.com/automl/DeepCAVE) to get importance scores and more.  
For non-Slurm usage you'll have to modify the launcher in `configs/hpo_algorithm.yaml`

### Phased Landscapes

Look at `train_all.sh` to get running. It offers you almost all options, only hyperparameters are not yet documented here (see `src/gcrl-landscapes/configurations.py`). Comment out the ones you don't want to run. `train_all.sh` is supposed to be submitted using `sbatch train_all.sh`.

For non-Slurm usage you'll have to modify the submitter in `src/gcrl-landscapes/submission.py` and probably also the `train_all.sh` script.

## Evaluation

These commands will generate plots in `plots` and tables in `tables`.  
If you don't want to generate these yourself, logfiles can also be found on [Huggingface](https://huggingface.co/datasets/jmtoepperwien/GCRL-Landscapes).

```
python -m gcrl_landscapes.evaluation.plot --plot_return_distributions --plot_eval_curves --plot_gp_fits --zipfile LOGFILEPATH
python -m gcrl_landscapes.evaluation.tabular --zipfile LOGFILEPATH
```

Inside of the plots subfolder you'll see a `grid_plots` directory, which should help you get a quick overview over all experiments.
