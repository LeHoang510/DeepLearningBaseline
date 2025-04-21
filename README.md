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
- You can install the required packages using uv. To do this, run the following command:

    ```bash
    pip install uv
    ```