# SSAC-MGI

**Safe and Energy-Efficient Trajectory Planning for Heterogeneous Multi-UAV Enabled Mobile Edge Computing**

## Introduction

This project implements a multi-agent safe reinforcement learning algorithm that integrate a Shared Soft Actor-Critic (SSAC) architecture for extracting UAV-specific heterogeneous features and a two-agent Markov Game of Intervention (MGI) for collision avoidance, named SSAC-MGI. It is designed to address safe and energy-efficient trajectory planning and task scheduling for heterogeneous multi-UAV systems operating in uncertain mobile edge computing (MEC) environments.

This research has been submitted for publication to IEEE Transactions on Mobile Computing (TMC).

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

- A Shared Soft Actor-Critic (SSAC) architecture for extracting UAV-specific heterogeneous features
- A two-agent Markov Game of Intervention (MGI) for collision avoidance
- A Round-Robin instance allocation method for assigning computational instances to workflow queues on UAVs. 

## Experimental Results

> SSAC-MGI outperforms multiple baselines in terms of lower task miss rate and reduced average energy consumption.  


## Acknowledgement

This project is built upon the codebase of [changmin-yu/desta-lunarlander](https://github.com/changmin-yu/desta-lunarlander), 
which provided the foundation for environment setup and baseline implementations. 
We sincerely thank the authors for making their work publicly available.


## Citation

If you use this code in your research, please cite the following:

```bibtex
@misc{ssacmgi2025,
  author       = {Xiuling Zhang},
  title        = {Safe and Energy-Efficient Trajectory Planning for Heterogeneous Multi-UAV Enabled Mobile Edge Computing},
  year         = {2025},
  howpublished = {GitHub},
  url          = {https://github.com/xiulingzhang22/SSAC-MGI}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact

Maintainer: **Xiuling Zhang**  
Email: `xiuling@nudt.edu.cn`
