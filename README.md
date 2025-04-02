# Cultural Evolution of Cooperation Among LLM Agents

This repository implements a framework for studying the cultural evolution of cooperation among LLM agents through an iterated Donor Game. Paper by Aron Vallinder and Edward Hughes: https://arxiv.org/abs/2412.10270.


## Overview

The framework simulates multiple generations of LLM agents playing a Donor Game where agents can observe the recent behavior of their peers. It explores:
- Cultural evolution of indirect reciprocity across generations
- Emergence of cooperative norms and strategies
- Impact of costly punishment mechanisms
- Comparative behavior across different LLM models (Claude 3.5 Sonnet, Gemini 1.5 Flash, GPT-4o)

## Key Features

- **Multi-generational Simulation**: Implements 10 generations with 12 agents per generation
- **Donor Game Framework**: 
  - Initial endowment of resources for each agent
  - Paired interactions with donation decisions
  - Cooperation multiplier (receiver gets 2x donated amount)
  - Optional costly punishment mechanism
- **Cultural Evolution**:
  - Strategy inheritance between generations
  - Selection based on accumulated resources
  - Transmission of successful strategies
- **Multi-Model Support**:
  - Claude 3.5 Sonnet
  - Gemini 1.5 Flash
  - GPT-4o
  - Extensible to other models

## Requirements

```bash
pip install openai
pip install scipy
pip install anthropic
pip install google-generativeai
pip install threading
```

## Setup

1. Configure API keys:
```python
OPENAI_API_KEY = "your-key"
ANTHROPIC_API_KEY = "your-key"
GOOGLE_API_KEY = "your-key"
```

2. Mount Google Drive (if using Colab):
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Core Components

### Classes
- `Agent`: Manages agent state, resources, reputation, and strategy
- `SimulationData`: Stores simulation parameters and results
- `AgentRoundData`: Records detailed per-round interaction data

### Key Functions

- `initializeAgents()`: Creates new agents with strategies
- `bipartiteRoundRobin()`: Generates pairing schedules
- `donorGame()`: Handles game logic and interactions
- `promptLLM()`: Manages LLM interactions
- `selectTopAgents()`: Implements evolutionary selection

### Key Parameters
```python
cooperationGain = 2        # Multiplier for donations
numGenerations = 10        # Number of evolutionary generations
numAgents = 12            # Agents per generation
initial_endowment = 10    # Starting resources
punishmentLoss = 2        # Punishment multiplier (optional)
```

## Running Experiments

```python
runGenerations(
    numGenerations=10,
    numAgents=12, 
    initialEndowment=10,
    selectionMethod='top'
)
```

## Data Collection

The framework automatically saves comprehensive data including:
- Agent strategies and their evolution
- Individual interactions and decisions
- Resource distribution over time
- Reputation dynamics
- Cultural transmission patterns

Data is saved in JSON format with configuration parameters and timestamps included in filenames.


## Citation

If you use this code in your research, please cite: https://arxiv.org/abs/2412.10270

## License

This project is licensed under Creative Commons Attribution 4.0 International (CC-BY 4.0).
