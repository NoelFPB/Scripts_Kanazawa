# Decoder Optimizer with Persistence

This enhanced version of the Decoder Optimizer adds data persistence capabilities to save and reuse model data between optimization runs, allowing you to build on previous results instead of starting from scratch each time.

## Key Features

- **Persistence**: Saves and loads model data, evaluation history, and best configurations
- **Incremental Learning**: Builds on previous evaluations to improve efficiency
- **Multiple Operation Modes**: Command-line and interactive interfaces
- **Configuration Management**: Save, list, and load multiple configurations

## Requirements

- Python 3.7+
- Required packages: `serial`, `pyvisa`, `numpy`, `scikit-learn`, `scipy`

## Installation

1. Ensure you have all the required packages installed:
   ```
   pip install pyserial pyvisa numpy scikit-learn scipy
   ```

2. Place the decoder optimizer script in your working directory

## Usage

### Command Line Interface

The script can be run with various command-line arguments:

```
python decoder_optimizer.py [options]
```

Options:
- `--no-load`: Do not load previous data, start fresh
- `--full`: Run full optimization even with existing data
- `--samples N`: Specify number of initial samples (default 20)
- `--list`: List all saved best configurations
- `--load N`: Load and test a specific configuration by index
- `--test`: Test the best configuration without optimization

Examples:
```
# Normal operation (loads previous data if available)
python decoder_optimizer.py

# List all saved configurations
python decoder_optimizer.py --list

# Load and test configuration #3
python decoder_optimizer.py --load 3

# Run full optimization with 50 initial samples
python decoder_optimizer.py --full --samples 50
```

### Interactive Mode

For easier use, you can run the script in interactive mode:

```
python decoder_optimizer.py --interactive
```

This provides a menu-driven interface with options to:
1. List saved configurations
2. Load and test specific configurations
3. Test current best configuration
4. Run quick optimization (building on existing data)
5. Run full optimization
6. Exit

## Data Storage

The script creates a `decoder_data` directory to store:
- `surrogate_model.pkl`: The trained machine learning model
- `evaluation_history.pkl`: Configuration and score history
- `best_configs.json`: History of best configurations found (human-readable)

## How It Builds Knowledge Over Time

1. **Data Persistence**: Each time you run the optimizer, it saves all evaluated configurations and their scores
2. **Model Transfer**: The surrogate model is saved and reloaded, preserving learned relationships
3. **Incremental Learning**: New runs start with the best configurations from previous runs
4. **Reduced Exploration**: When building on existing data, the optimizer can focus on promising regions

## Optimization Strategy

1. **Initial Sampling**: Uses Latin Hypercube Sampling for efficient space exploration (reduced when building on existing data)
2. **Surrogate Modeling**: Trains a Random Forest model to predict configuration scores
3. **Evolutionary Optimization**: Uses the model to guide exploration of promising areas
4. **Adaptive Grid Search**: Focuses on the most important heater parameters
5. **Differential Evolution**: Refines the search in the most promising regions
6. **Local Search**: Fine-tunes the best configuration found

## Customization

You can modify the following parameters in the script:
- `INPUT_PINS`: Pins used for decoder inputs
- `LOW_VOLTAGE` and `HIGH_VOLTAGE`: Input voltage levels
- `LOW_THRESHOLD` and `HIGH_THRESHOLD`: Output threshold values
- `OPTIMAL_LOW` and `OPTIMAL_HIGH`: Target values for outputs
- `VOLTAGE_OPTIONS`: Allowed voltage values for heaters

## Tips for Effective Use

1. Start with a full optimization run to build initial knowledge
2. For subsequent runs, use the quick optimization mode to build on previous knowledge
3. If targeting a specific behavior, test saved configurations first
4. If performance degrades, try a new full optimization
5. Use the interactive mode for easier experimentation