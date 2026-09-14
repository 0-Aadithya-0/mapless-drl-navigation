# Mapless Deep-RL Navigation

LiDAR-based mapless goal-reaching navigation for a TurtleBot3, learned end-to-end with deep reinforcement learning. The robot navigates to a goal in an unknown environment without a map or a path planner, using only its laser scan and the goal's relative position. The control policy is trained from scratch (DQN, then TD3) in ROS 2 and Gazebo.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Approach: DQN and TD3](#approach-dqn-and-td3)
- [Interface Contract](#interface-contract)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation and Results](#evaluation-and-results)
- [Project Roadmap](#project-roadmap)
- [Team](#team)
- [Documentation](#documentation)
- [References and Acknowledgements](#references-and-acknowledgements)

---

## Overview

Classical robot navigation, such as the ROS Nav2 stack, first constructs a map of the environment and then plans a path through it. This project adopts a mapless alternative: a neural-network policy maps the robot's current sensor readings directly to motor commands, and the behavior is acquired through reinforcement learning in simulation.

The problem definition:

- **Mapless** — the robot operates without a prior map and constructs none; it responds only to current observations.
- **Goal-reaching** — the robot must reach a specified target position while avoiding obstacles.
- **LiDAR-based** — perception is limited to a downsampled 2D laser scan giving the distance to the nearest obstacle in each direction.
- **Deep reinforcement learning** — the control policy is a neural network trained with Deep Q-Networks (DQN) and Twin Delayed Deep Deterministic Policy Gradient (TD3).

The agent observes a compact state consisting of 24 downsampled LiDAR beams together with the goal's distance and bearing, and outputs a velocity command. It receives positive reward for progress toward the goal and for reaching it, and negative reward for collisions and for elapsed time. Across repeated simulated episodes, the agent learns a reactive navigation policy that transfers to environments not encountered during training.

---

## Key Features

- **End-to-end learned navigation** — no map and no global or local planner; sensor readings are mapped directly to velocity commands.
- **Two algorithms behind a shared interface** — a discrete-action DQN baseline and a continuous-control TD3 policy, both implemented from scratch in PyTorch.
- **Decoupled RL and robotics boundary** — the robot is presented to the learning code as a standard Gymnasium environment, isolating the reinforcement learning components from ROS-specific details.
- **Reproducible training** — seeded runs, hyperparameters defined in version-controlled YAML configuration files, and full TensorBoard logging.
- **Quantitative evaluation** — success rate, collision rate, path length, and time-to-goal, reported as a direct DQN-versus-TD3 comparison that includes an environment withheld from training.

---

## System Architecture

The system operates as a closed control loop executed many times per second. Gazebo simulates the robot and environment, the reinforcement learning environment converts raw sensor topics into a state vector, the policy network selects an action, and the action is published back to the robot as a velocity command.

The data flow, per timestep:

1. **Gazebo** simulates the robot body, its sensors, and the environment's physics.
2. The LiDAR publishes distance readings to `/scan`, and wheel odometry publishes the robot's pose to `/odom`.
3. The **RL environment** subscribes to both topics, assembles the 26-element state vector, and passes it to the policy.
4. The **policy network** (DQN or TD3) outputs an action, which is published to `/cmd_vel`.
5. Gazebo applies the command, the robot moves, and updated sensor data initiates the next iteration. During training, each timestep additionally yields a shaped reward.

The Gymnasium environment constitutes the single interface between the robotics stack and the reinforcement learning stack. The learning code interacts only through `reset()` and `step(action)`. On benchmark tasks this environment is provided by Gymnasium directly (for example `CartPole-v1`); on the robot it is the ROS-backed `TurtleBotEnv`. The same agent code trains against either environment without modification.

aadithya@Orange:~/mapless-drl-navigation$ export GAZEBO_MODEL_PATH=~/mapless-drl-navigation/ros2_ws/src/custom_world/models:$GAZEBO_MODEL_PATH
gazebo ~/mapless-drl-navigation/ros2_ws/src/custom_world/worlds/showcase.world 

A rendered architecture diagram is available at `docs/control-loop.svg`.

---

## Approach: DQN and TD3

Development proceeds through four stages of increasing complexity. Each algorithm is first validated on a standard benchmark environment before being applied to the robot, which isolates implementation errors in the learning code from integration and simulation issues.

| Stage | Algorithm | Environment | Action space | Objective |
|-------|-----------|-------------|--------------|-----------|
| 1 | DQN | CartPole (benchmark) | Discrete | Validate the deep reinforcement learning implementation without robot integration |
| 2 | DQN | TurtleBot3 in Gazebo | Discrete (forward / left / right / stop) | First integrated robotics result: obstacle avoidance |
| 3 | TD3 | Pendulum (benchmark) | Continuous | Validate the actor-critic implementation |
| 4 | TD3 | TurtleBot3 in Gazebo | Continuous (linear and angular velocity) | Primary objective: continuous goal-reaching navigation |

**DQN (Deep Q-Network)** estimates a value for each discrete action and selects the highest-valued one. It provides the discrete-action baseline for this task.

**TD3 (Twin Delayed DDPG)** outputs continuous velocities directly, producing smoother and more capable motion. It improves on its predecessor, DDPG, with three modifications:

1. **Twin critics** — two value networks are trained, and the smaller of their two estimates is used to reduce overestimation bias.
2. **Delayed policy updates** — the actor is updated less frequently than the critics to improve stability.
3. **Target policy smoothing** — clipped noise is added to target actions so the critic cannot exploit narrow peaks in the value estimate.

---

## Interface Contract

The fixed agreement between the reinforcement learning and robotics components. The complete specification is in [`docs/interface_contract.md`](docs/interface_contract.md).

| Item | Specification |
|------|---------------|
| **State** | `[24 downsampled LiDAR beams, distance_to_goal, angle_to_goal]`, 26 normalized floats |
| **Action (DQN)** | Discrete: forward / left / right / stop |
| **Action (TD3)** | Continuous: `[linear_vel, angular_vel]`, rescaled to the robot's velocity limits |
| **Reward** | Positive for reaching the goal, positive for progress toward the goal, negative for collision, small negative per-step time penalty |
| **Topics** | `/scan` and `/odom` as input, `/cmd_vel` as output |
| **Termination** | Goal reached or collision (`terminated`); step limit exceeded (`truncated`) |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Robotics middleware | ROS 2 Humble |
| Simulation | Gazebo Classic, TurtleBot3 (Burger) |
| Learning | Python 3.10, PyTorch, Gymnasium |
| Logging and analysis | TensorBoard, NumPy |
| Sensors | 2D LiDAR (`sensor_msgs/LaserScan`), wheel odometry (`nav_msgs/Odometry`) |

---

## Repository Structure

```
mapless-drl-navigation/
├── docs/                      # knowledge document, interface contract, diagram
├── rl/                        # pure-Python reinforcement learning (no ROS); runs in a venv
│   ├── agents/                # dqn.py, td3.py, networks.py, replay_buffer.py
│   ├── configs/               # dqn.yaml, td3.yaml (hyperparameters)
│   ├── utils/                 # seeding, config loading, action scaling
│   ├── train.py               # unified entry point: --algo {dqn,td3} --env {cartpole,pendulum,turtlebot}
│   └── evaluate.py            # produces the DQN-versus-TD3 metrics table
└── ros2_ws/src/turtlebot3_rl_nav/   # ament_python ROS 2 package
    ├── turtlebot3_rl_nav/
    │   ├── env.py             # Gymnasium environment bridging ROS/Gazebo and RL
    │   ├── ros_interface.py   # rclpy node: subscribes to /scan and /odom, publishes /cmd_vel
    │   └── lidar_processing.py# downsamples LiDAR to 24 beams and cleans inf/NaN values
    ├── launch/                # brings up Gazebo and the robot
    └── worlds/                # custom Gazebo training world(s)
```

---

## Getting Started

### Prerequisites

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic and the TurtleBot3 simulation packages
- Python 3.10

Set the robot model before launching the simulation:

```bash
export TURTLEBOT3_MODEL=burger
```

### Installation

ROS 2 Humble's `rclpy` is built against the system Python 3.10 installation. Create the virtual environment with `--system-site-packages` so that `rclpy` remains importable alongside PyTorch; a fully isolated virtual environment will prevent `import rclpy` from succeeding.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/mapless-drl-navigation.git
cd mapless-drl-navigation

# 2. Create the ROS-compatible Python environment
python3.10 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Build the ROS 2 package
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
cd ..
```

---

## Usage

Validate the algorithms on benchmark environments first; these require no ROS installation.

```bash
python -m rl.train --algo dqn --env cartpole      # DQN validation
python -m rl.train --algo td3 --env pendulum      # TD3 validation
```

Train on the robot using two terminals — one to launch the simulation and one to run training.

```bash
# Terminal A
ros2 launch turtlebot3_rl_nav training.launch.py

# Terminal B
python -m rl.train --algo dqn --env turtlebot     # DQN obstacle avoidance
python -m rl.train --algo td3 --env turtlebot     # TD3 goal-reaching
```

Monitor training:

```bash
tensorboard --logdir runs/
```

Evaluate a trained policy and generate the comparison metrics:

```bash
python -m rl.evaluate --algo td3 --env turtlebot --ckpt checkpoints/td3_turtlebot_final.pt --episodes 100
```

---

## Evaluation and Results

Agents are evaluated with exploration disabled over a fixed set of start and goal pairs, including at least one environment withheld from training in order to measure generalization.

<!-- Populate these values after the Week 3 evaluation runs. -->

| Metric | DQN | TD3 |
|--------|-----|-----|
| Success rate | TBD | TBD |
| Collision rate | TBD | TBD |
| Average path length (m) | TBD | TBD |
| Average time-to-goal (steps) | TBD | TBD |
| Average return | TBD | TBD |

Demo recordings:

- DQN obstacle avoidance — link to be added
- TD3 goal-reaching — link to be added

Anticipated outcome: the discrete-action DQN policy produces coarser and less efficient trajectories, whereas TD3's continuous control yields smoother and shorter paths and a higher success rate.

---

## Project Roadmap

A three-week, two-person build with a go/no-go checkpoint at the end of each week.

- **Week 1 — Foundations.** DQN solves CartPole; the TurtleBot3 drives in Gazebo. (Gate 1)
- **Week 2 — DQN in Gazebo.** The Gymnasium environment is built, the interface contract is frozen, and DQN performs obstacle avoidance, verified by a recorded demonstration. (Gate 2)
- **Week 3 — TD3.** TD3 is implemented from scratch, achieves goal-reaching, and is evaluated against DQN, with the repository finalized for release. (Gate 3)

Final deliverables: a DQN obstacle-avoidance recording, a TD3 goal-reaching recording, a DQN-versus-TD3 comparison table, and this document with an architecture diagram.

A tiered contingency plan, detailed in the knowledge document, safeguards the schedule. If the project falls behind, scope is reduced in stages: fine-tuning is dropped first, an established TD3 implementation is substituted for the demonstration next, and, if necessary, the DQN obstacle-avoidance system is delivered as the final result. The plan prioritizes delivering a functional system over an incomplete implementation of a more advanced one.

---

## Team

| Member | Role | Responsibilities |
|--------|------|------------------|
| **Ak** | Robotics Engineer | ROS 2 nodes, Gazebo world, LiDAR and odometry state, ROS-to-RL bridge, integration |
| **Aashi** | RL Engineer | DQN and TD3 implementation, PyTorch, reward design, training, evaluation |

---

## Documentation

- [`docs/KNOWLEDGE_DOC.md`](docs/KNOWLEDGE_DOC.md) — the complete reference covering reinforcement learning foundations, DQN and TD3 internals, the robotics stack, reward shaping, the week-by-week plan, debugging, and evaluation.
- [`docs/interface_contract.md`](docs/interface_contract.md) — the fixed data contract between the reinforcement learning and robotics components.

---

## References and Acknowledgements

- V. Mnih et al., *Human-level control through deep reinforcement learning* (DQN), 2015.
- S. Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods* (TD3), 2018.
- [ROBOTIS TurtleBot3](https://github.com/ROBOTIS-GIT/turtlebot3) and [TurtleBot3 Simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations).
- [Gymnasium](https://gymnasium.farama.org/), the reinforcement learning environment interface.
- Related deep reinforcement learning navigation projects that informed this work, including `reiniscimurs/DRL-Robot-Navigation-ROS2` and `tomasvr/turtlebot3_drlnav`.

---
