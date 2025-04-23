# SSAC-MGI

**Safe and Energy-Efficient Trajectory Planning for Heterogeneous Multi-UAVs in Stochastic Mobile Edge Computing Environments**

## Introduction

This project implements a Shared Safety-Aware Multi-Agent Reinforcement Learning framework, called SSAC-MGI, which aims to achieve safe and energy-efficient trajectory planning and task scheduling for heterogeneous multi-UAV systems operating in uncertain mobile edge computing (MEC) environments.

## Project Structure

```
.
├── environment/            # Environment simulation module
├── policies/               # Reinforcement learning policies
├── stable-baselines3/      # Reinforcement learning library
├── plots/                  # Evaluation visualizations
├── main.py                 # Main execution script
├── requirements.txt        # Dependency list
├── run.sh                  # Shell script for automated running
├── README.md               # Project documentation
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the project

```bash
python main.py
```

Or use the shell script:

```bash
bash run.sh
```

## Core Algorithm: SSAC-MGI

- Multi-agent reinforcement learning with integrated safety intervention mechanism
- Intelligent task scheduling and UAV energy efficiency
- Decentralized policy execution with safe trajectory planning

## Experimental Results

> SSAC-MGI outperforms multiple baselines in terms of lower task miss rate and reduced energy consumption.  
> Visual results are available in the `plots/` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact

Maintainer: **Xiuling Zhang**  
Email: `xiuling@nudt.edu.cn`
