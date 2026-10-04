"""Comet tracking for evaluation runs, set up like train.py's: the neo-rubin-lsst project in the
samkahn-astro workspace, online when COMET_ML_ASTRO_API_KEY is set, otherwise an offline archive in
comet_offline/ (upload later with `comet upload`)."""

import os

PROJECT = "neo-rubin-lsst"
WORKSPACE = "samkahn-astro"


def start_experiment(name, tags, project=PROJECT):
    from comet_ml import Experiment, OfflineExperiment

    kwargs = dict(project_name=project, workspace=WORKSPACE)
    api_key = os.environ.get("COMET_ML_ASTRO_API_KEY")
    if api_key:
        experiment = Experiment(api_key=api_key, **kwargs)
    else:
        print("COMET_ML_ASTRO_API_KEY not set; logging offline to ./comet_offline")
        experiment = OfflineExperiment(offline_directory="comet_offline", **kwargs)
    experiment.set_name(name)
    experiment.add_tags([t for t in tags if t])
    return experiment
