# DeepLearningBaseline
This is set up for deep learning baseline
- TODO:
+ rewrite the losses in train script
+ rewrite evaluate function and yaml config
-> metrics select base on input (metric -> function -> call)
+ rewrite validation step 
+ Test pretrained models
+ Test resume training

# Set up + Installation
## 1. Clone the repository

```bash
git clone git@github.com:LeHoang510/DeepLearningBaseline.git
cd DeepLearningBaseline
```
## 2. Create a conda environment

### 2.1. Create a conda environment with the required packages
- You can create a conda environment or use an existing one or use a virtual environment. To create a conda environment, run the following command:
    
    ```bash
    # Create and activate a conda environment
    conda create -n <env-name> python=<python-version>
    conda activate <env-name>

    # Example:
    conda create -n dl python=3.10
    conda activate dl
    ```

### 2.2. Install the required packages

#### 2.2.1. Install the required packages using uv

- You can install the required packages using uv. To do this, run the following command:

    ```bash
    # Install uv
    pip install uv
    ```
- Install the required packages using uv:

    ```bash
    # Install specific package 
    uv pip install <package-name>

    # Example:
    uv pip install numpy
    ```

- Add dependencies to the `pyproject.toml` file as needed:

    ```bash
    # Add specific package to pyproject.toml
    uv add <package-name>

    # Example:
    uv add numpy
    ```
- Install the dependencies in the `requirements.txt` file for existing projects:

    ```bash
    # Install all dependencies in requirements.txt
    uv pip install -r requirements.txt
    ```

#### 2.2.2. Install the required packages using poetry
- You can install the required packages using poetry. To do this, run the following command:

    ```bash
    # Install poetry
    pip install poetry
    ```
- Add dependencies to the `pyproject.toml` file as needed:

    ```bash 
    # Add specific package to pyproject.toml
    poetry add <package-name>

    # Example:
    poetry add numpy
    ```
- Add source to the `pyproject.toml` file as needed:

    ```bash
    # Add specific source to pyproject.toml
    poetry source add --priority=explicit <source-name> <source-url>
    poetry add --source <source-name> <package-name>

    # Example:
    poetry source add --priority=explicit pytorch https://download.pytorch.org/whl/cu118
    poetry add --source pytorch torch
    ```

- Install the dependencies in the `pyproject.toml` file for existing projects:

    ```bash
    # Install all dependencies in pyproject.toml
    poetry install
    ```

## 2.3. Check hardware compatibility
- You can check if your hardware is compatible by running the following command (only with torch installed):

    ```bash
    python utils/check_hardware.py
    ```

# Usage
## 1. Run the training script
- You can run the training script by running the command below. Modify verbose=False to verbose=True to see more details of hardware information during training process. The output in the terminal will show the training progress and the logs will be saved in the `logs/train/<timestamp>` directory and the model checkpoints as well as the config file and tensorboard logs will be saved in the `outputs/train/<experiment-name>` directory.:

    ```bash
    # Train the model
    python train/<train-script>.py

    # Example:
    python train/mnist_train.py
    ```

- The training script will automatically delete the previous experiment with the same name in the `outputs/train/` directory. If you want to keep the previous experiment, you can change the experiment name in the config.

## 2. Run the test script
- You can run the test script by running the command below. #TODO:

    ```bash
    # Test the model
    python test/<test-script>.py
    
    # Example:
    python test/mnist_test.py
    ```

## 3. Run the inference script
- You can run the inference script by running the command below. #TODO:

    ```bash
    # Inference the model
    python inference/<inference-script>.py  

    # Example:
    python inference/mnist_inference.py
    ```

## 4. Run the tensorboard
- You can run the tensorboard by running the following command:

    ```bash
    # Run tensorboard to visualize the training logs
    tensorboard --logdir=outputs/train/<experiment-name>

    # Example:
    tensorboard --logdir=outputs/train/mnist_experiment
    ```

# Quick start
#### TODO