# WTSoftwareUtilitiesPyQt

# Concentration GUI Application

A Python-based software for concentration analysis accessible over a GUI,  built with PyQt5 and scientific computing libraries.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation & Setup

You can set up this application using one of two methods:

### Method 1: Direct Installation (System-wide)

Install the required dependencies directly to your system Python:
(!!It is recommended to install dependencies in an environment (method 2 or 3) to avoid potential problems with the versions of library dependencies required for other software installations on your system.!!)

```bash
pip install -r requirements.txt
```

### Method 2: Virtual Environment (Recommended)

Create an isolated Python environment for this project:

####Method 3: Using venv (Built-in)

```bash
# Create virtual environment
python3 -m venv WTconc

# Activate the environment
# On Linux/macOS:
source WTconc/bin/activate
# On Windows:
WTconc\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Using conda (Alternative)

```bash
# Create virtual environment
conda create -n WTconc python=3.9

# Activate the environment
conda activate WTconc

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

Once you have installed the dependencies, start the application with:

```bash
python3 concentration_gui.py
```

Or simply:

```bash
python concentration_gui.py
```


### Deactivating Virtual Environment

When you're done working with the application, you can deactivate the virtual environment:

```bash
deactivate WTconc
```

## Dependencies

This application requires the following Python packages:

- **PyQt5** (≥5.15.0) - GUI framework
- **matplotlib** (≥3.5.0) - Plotting and visualization
- **pandas** (≥1.3.0) - Data manipulation and analysis
- **numpy** (≥1.21.0) - Numerical computing
- **scipy** (≥1.7.0) - Scientific computing

## Troubleshooting

### Common Issues

**ImportError: No module named 'PyQt5'**
- Make sure you've installed the dependencies: `pip install -r requirements.txt`
- If using a virtual environment, ensure it's activated

## Support
If you encounter any issues, please check that:
1. You're using Python 3.7 or higher
2. All dependencies are properly installed
3. Your virtual environment is activated (if using Method 2)
   
4. For further questions or ideas feel free to contact me: Sabrina.ebert@studium.uni-hamburg.de



(/d D:)
