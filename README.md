# Isaac Lab Environments For Training Robots and Visualizing in Isaac Sim

## Overview

This project/repository was generated using the [Isaac Lab Template Generator](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html). 
Using the template generator allows for development in an isolated environment, outside of the core Isaac Lab repository.

## Environments
In this repository, there are 3 trainable tasks: A quadcopter hover task, a Jackal rover navigation task in a flat grid-world, and a Jackal rover navigation task in terrain with varying heights. 

### Jackal Rover Navigation in a Flat Grid-World
- [Jackal Grid World Environment Source Code](source/acelab/acelab/tasks/direct/jackal-nav/jackal_grid_env.py)
- [Jackal Grid World Environment Configuration Source Code](source/acelab/acelab/tasks/direct/jackal-nav/jackal_grid_env_cfg.py)

### Jackal Rover Navigation in Variable-Height Terrain (Works in some cases, but still under development)
- [Jackal Terrain Environment Source Code](source/acelab/acelab/tasks/direct/jackal-na/jackal_terrain_env.py)
- [Jackal Terrain Environment Configuration Source Code](source/acelab/acelab/tasks/direct/jackal-nav/jackal_terrain_env_cfg.py)

### Iris Quadcopter Fly-To-And-Hover
- [Quadcopter Environment Source Code](source/acelab/acelab/tasks/direct/drone-nav/quadcopter_env.py): Much of this is adapted from the [official Crazyflie example](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py). What must be changed (depending on the drone being trained) is the thrust-to-weight ratio and the moment scale. Using drone dynamics, the maximum thrust and maximum moment that can be applied to the drone must be calculated to set these parameters. 
- The environment configuration definition is included in the environment file. 

Each environment.py file includes reward function definitions, functions for defining how the agent observes data from sensors, code for setting up the simulation scene, and more. Isaac Lab has two types of environments: Manager-based and Direct, and these environments are all Direct. Resources on Isaac Lab Direct environments can be found [here](https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/technical_env_design.html) and [here](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html). 

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
 
## Other External Downloads

- REQUIRED FOR JACKAL TERRAIN ENV: Download [terrain assets](https://drive.google.com/file/d/16lNJSQZqxh0eJDLeqVaRaFAi0tgYlJKF/view?usp=sharing) here, and unzip. Create a directory named "terrain" at the filepath "source/acelab/acelab". Paste the contents of the zipped file into this "terrain" directory. The terrain asset is from [RLRoverLab](https://github.com/abmoRobotics/RLRoverLab).
  
- REQUIRED FOR QUADCOPTER ENV: Download the Iris USD file from [here](https://github.com/btx0424/OmniDrones/blob/main/omni_drones/robots/assets/usd/iris.usd). Then put the usd file in the directory "source/acelab/acelab/robots/quadcopters/iris", next to the iris.yaml file. This quadcopter asset is from [OmniDrones](https://github.com/btx0424/OmniDrones).

## Training

To run a training script for a specific task, we run a python file in the "scripts/skrl" folder and pass in the name of the task as an argument. To train an Iris quadcopter to fly to and hover at some goal point, run the below command from the root directory"
  
```bash
python scripts/skrl/train.py --task=Template-Quadcopter-Direct-v0
```

Because the Jackals use camera sensors, they have to be trained with a training script that uses a CNN instead of a regular MLP: 

```bash
# Jackal navigation in flat grid-world
python scripts/skrl/trainCNN.py --task=Template-Jackal-Grid-Direct-v0 --enable_cameras
```
```bash
# Jackal navigation in variable-height terrain
python scripts/skrl/trainCNN.py --task=Template-Jackal-Terrain-Direct-v0 --enable_cameras
```

## Evaluation

For the quadcopter task, the below evaluation script can be ran after the training script is completed to see the performance of the RL policy:

```bash
python scripts/skrl/play.py --task=Template-Quadcopter-Direct-v0
```

And for the Jackals, the CNN variants of the eval scripts must be used:

```bash
python scripts/skrl/playCNN.py --task=<Jackal-Task-Name> --enable_cameras
```

A current issue is that to evaluate the Jackal policy, in playCNN.py, the .pt file output by trainCNN.py must be located within a directory "runs" generated by the script, and then manually loaded in to playCNN.py. An example of this is shown below: 

```python
    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

    # After training, the .pt file generated must be located within the "runs/torch/Isaac-Jackal-Nav" directory, and manually loaded into the script using agent.load()
    agent.load("/home/bchien1/acel-isaaclab/runs/torch/Isaac-Jackal-Nav/25-10-31_17-37-14-618688_PPO/checkpoints/best_agent.pt")

    
    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 100000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    trainer.eval()
```

This plumbing issue is currently a TODO and will be fixed.

## Citations

```bibtex
@inproceedings{mortensen2024rlroverlab,
  title={RLRoverLAB: An Advanced Reinforcement Learning Suite for Planetary Rover Simulation and Training},
  author={Mortensen, Anton Bj{\o}rndahl and B{\o}gh, Simon},
  booktitle={2024 International Conference on Space Robotics (iSpaRo)},
  pages={273--277},
  year={2024},
  organization={IEEE}
}

@misc{xu2023omnidrones,
    title={OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control},
    author={Botian Xu and Feng Gao and Chao Yu and Ruize Zhang and Yi Wu and Yu Wang},
    year={2023},
    eprint={2309.12825},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
