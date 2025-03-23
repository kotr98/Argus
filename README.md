## Project Overview

**ARGUS** (Autonomous Reconnaissance & Guidance for Unmanned Swarms) is designed to implement an advanced drone swarm search algorithm. Developed during the 48-hour European Defence Tech Hub Hackathon 2025 (https://lu.ma/hlzlyfvd?tk=CHSGxU), ARGUS autonomously downloads satellite images, detects street intersections as potential points of interest (POIs), and synchronizes a swarm of drones to search, engage, and confirm targets. This configuration-driven approach gives users flexibility to define mission parameters right from the start.

### Key Features
- **Autonomous Orchestration:** Manage a self-directed swarm of drones with a single command.
- **Resilience:** Maintain effective operations even if individual drones are compromised.
- **Scalability:** Adjust the drone swarm size based on budget and mission demands.
- **Enhanced Mission Efficiency:** Outperform traditional methods by autonomously identifying and engaging targets.

## Installation Instructions

To install the project dependencies in editable mode, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/TJB-99/Argus.git
    cd Argus
    ```

2. **Set up a virtual environment**:

    **Using `venv`**:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

    **Using `conda`**:
    ```sh
    conda create --name myenv python=3.13
    conda activate myenv
    ```

3. **Install the dependencies in editable mode**:
    ```sh
    pip install -e .
    ```

Ensure `pip` is installed and configured correctly. The `pyproject.toml` file manages your dependencies.

## Running Instructions

Before running the application, update the configuration file:

1. **Edit Configuration**: Open the `config.yml` file with your preferred text editor. Adjust the settings to match your mission parameters, such as inserting bounding box coordinates sourced from mapping services like Google Maps. The application will automatically download a corresponding satellite image, detect street intersections as potential points of interest, and configure the drone swarm to initiate the mission.

2. **Start the Application**: Run the repository by executing:
    ```sh
    python run.py
    ```

## Demo
![Demo](demo.gif)