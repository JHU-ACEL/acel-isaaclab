# Isaac Lab Environments For Training Robots with RL Libraries and Visualizing in Isaac Sim

## Overview

This project/repository was generated using the [Isaac Lab Template Generator](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html). 
It for development in an isolated environment, outside of the core Isaac Lab repository.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  The conda installation is recommended as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- From the root directory, run the below command to install the project:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/acelab

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a training task, explained below:

## Training

To run a training script for a specific task, we run a python file in the "scripts/skrl" folder and pass in the name of the task as an argument. To train an Iris quadcopter to fly to and hover at some goal point, run the below command from the root directory"
  
```bash
python scripts/skrl/train.py --task=Template-Quadcopter-Direct-v0
```

There are three tasks so far: A quadcopter hover task, a Jackal rover navigation task in a flat grid-world, and a Jackal rover navigation task in terrain with varying heights. Because the Jackals use camera sensors, they have to be trained with a training script that uses a CNN instead of a regular MLP: 

```bash
# Jackal navigation in flat grid-world
python scripts/skrl/trainCNN.py --task=Template-Jackal-Grid-Direct-v0
```
```bash
# Jackal navigation in variable-height terrain
python scripts/skrl/trainCNN.py --task=Template-Jackal-Terrain-Direct-v0
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/acelab"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
