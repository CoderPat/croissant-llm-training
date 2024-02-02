# CroissantLLM: Training Repository

## Installation Instructions

As a pre-requisite, make sure you have [ducttape](https://github.com/CoderPat/ducttape) and [(mini)conda](https://docs.conda.io/en/latest/miniconda.html) installed.

First, clone this repository and its submodules:

```bash
git clone --recurse-submodules git@github.com:CoderPat/croissant-llm-training.git
```

Then, to create a new conda environment with all the necessary dependencies, run the following command:

```bash
export CONDA_HOME="/path/to/(mini)conda3"
bash setup/conda.sh
```

## Running pipelines

The core experimentation and training pipelines rely on ducttape, and are defined in `main.tape`. 
Configuration files for different models and datasets are defined in `configs/`.

Start by creating a configuration with user-dependent variables (like the output folder) in associated `configs/*_uservars.conf` associated with your chosen `.tconf`. E.g, for the `configs/croissant_llm.tconf` configuration, create a `configs/croissant_llm_uservars.conf` file with the following content:
```
global {
    ducttape_output=/path/to/output
    repo=/path/to/croissant-llm-training

    (...)
    # use a simple shell submitter 
    # we are forced to explicitly set the submitter parameters
    # to make it compatible with other submitters (ie the slurm submitter)
    submitter=shell
    dump_account=none
    dump_partition=none
    (...)
}
```

We provide a template for our user variables used in JeanZay.

Then, you can ran the one of the specified pipelines in `main.tape` by running ducttape with the corresponding configuration file:

```bash
conda activate towerllm-env
ducttape main.tape -C configs/tower_llm.conf 
```