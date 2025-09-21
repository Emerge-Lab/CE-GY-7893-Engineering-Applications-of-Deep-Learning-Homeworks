# CE-GY-7893 Engineering Applications of Deep Learning Homeworks

## Getting Started

```bash
git clone https://github.com/Emerge-Lab/CE-GY-7893-Engineering-Applications-of-Deep-Learning-Homeworks.git
cd CE-GY-7893-Engineering-Applications-of-Deep-Learning-Homeworks
```

## Setup

This project supports both [uv](https://docs.astral.sh/uv/) and Conda for Python package management.

### Option 1: Using uv (Recommended)

#### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Install dependencies

```bash
uv sync
```

#### Activate the environment

```bash
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

Alternatively, you can run commands directly with uv:

```bash
uv run python your_script.py
```

### Option 2: Using Conda

#### Create and activate environment

```bash
conda create -n deep-learning python=3.9
conda activate deep-learning
```

#### Install dependencies

```bash
# Install pip in the conda environment
conda install pip

# Install dependencies from pyproject.toml
pip install -e .
```

## VS Code Setup (Optional)

For the best experience, we recommend using VS Code with Python support.

### Install VS Code Extensions

Install these recommended extensions:

1. **Python** (by Microsoft) - Useful for Python development
2. **Jupyter** (by Microsoft) - For working with Jupyter notebooks
3. **Pylance** (by Microsoft) - Advanced Python language support

You can install them via:
- VS Code Extensions marketplace
- Command line: `code --install-extension ms-python.python ms-toolsai.jupyter ms-python.pylance`

### Configure Python Interpreter

1. Open VS Code in your project directory: `code .`
2. Open Command Palette (`Cmd/Ctrl + Shift + P`)
3. Type "Python: Select Interpreter"
4. Choose the appropriate interpreter:
   - **uv users**: Select the interpreter from `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows)
   - **Conda users**: Select the interpreter from your `deep-learning` environment

### Working with Jupyter Notebooks in VS Code

With the Jupyter extension installed, you can:
- Create and edit `.ipynb` files directly in VS Code
- Run cells interactively with `Shift+Enter`
- View plots and outputs inline
- Debug notebook cells

## Working with Jupyter Notebooks

To work with the homework notebooks:

```bash
# Launch Jupyter Lab
uv run jupyter lab

# Then navigate to and open the .ipynb files
```
If you're not using uv, just launch them the normal way you might a jupyter notebook.