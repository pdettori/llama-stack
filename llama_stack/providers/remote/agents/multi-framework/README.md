# Llama Stack with queued workers PoC

Note: we use minconda with two different envs one for the server and one for the workers.

## Running demo

clone this fork:

```shell
git clone https://github.com/pdettori/llama-stack.git
```

cd to project and checkout branch:

```shell
cd llama-stack
git checkout run-queues
```

### Install deps for server

** Note - this is a temporary hack until we figure out the right way **

1. Install conda/miniconda as reccommended. 
2. Create and activate a conda environment with name 'lsenv'
3. After doing the "llama stack build" step, there will be a python 3.10 env within `/opt/homebrew/Caskroom/miniconda/base/envs/lsenv` which has been created by `uv`
4. Use that env to install the extra deps - e.g.
```shell
/opt/homebrew/Caskroom/miniconda/base/envs/lsenv/bin/pip install -r llama_stack/providers/inline/agents/meta_reference_queued/requirements.txt 
```

### Install deps for worker

Follow [these steps](https://pypi.org/project/llama-stack/)

```shell
conda create -n stack python=3.10
conda activate stack
pip install -e .
``` 

### Running

