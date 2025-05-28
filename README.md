# DeepLearningBaseline
This is set up for deep learning baseline


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
    conda create -n <env-name> python=<python-version>
    conda activate <env-name>
    ```
- Example:

    ```bash
    conda create -n dl python=3.11
    conda activate dl
    ```

### 2.2. Install the required packages

#### 2.2.1. Install the required packages using uv

- You can install the required packages using uv. To do this, run the following command:

    ```bash
    pip install uv
    ```

- Add dependencies to the `pyproject.toml` file as needed:

    ```bash
    uv add <package-name>
    ```
- Install the dependencies in the `requirements.txt` file for existing projects:

    ```bash
    uv pip install -r requirements.txt
    ```

#### 2.2.2. Install the required packages using poetry
- You can install the required packages using poetry. To do this, run the following command:

    ```bash
    pip install poetry
    ```
- Add dependencies to the `pyproject.toml` file as needed:

    ```bash 
    poetry add <package-name>
    ```

- Install the dependencies in the `pyproject.toml` file for existing projects:

    ```bash
    poetry install
    ```

## 2.3. Check hardware compatibility
- You can check if your hardware by running the following command:

    ```bash
    python utils/check_hardware.py
    ```

# Usage
## 1. Run the training script
- You can run the training script by running the following command:

    ```bash
    python train.py --config configs/<config-file>.yaml
    ```

## 2. Run the evaluation script
- You can run the evaluation script by running the following command:

    ```bash
    python evaluate.py --config configs/<config-file>.yaml
    ```

## 3. Run the inference script
- You can run the inference script by running the following command:

    ```bash
    python inference.py --config configs/<config-file>.yaml
    ```

## 4. Run the tensorboard
- You can run the tensorboard by running the following command:

    ```bash
    tensorboard --logdir=logs/<experiment-name>
    ```