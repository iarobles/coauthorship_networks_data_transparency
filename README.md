# Generation of Predictions and Graphs

This project generates prediction reports using classification models, specifically Random Forest and Dummy classifiers. The reports include graphs and detailed analyses of the prediction results.

## Requirements

- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Clone the repository to your local machine:

    ```sh
    git clone <REPOSITORY_URL>
    cd <REPOSITORY_NAME>
    ```
2. Create a virtual environment in a `.venv` folder:

    ```bash
    # Create the virtual environment
    python -m venv .venv
    ```

 3. Once the virtual environment is created, activate it with the following command:

    ```bash
    # Activate the virtual environment
    # On Windows
    .venv\Scripts\activate
    ```

    ```bash
    # On macOS/Linux
    source .venv/bin/activate
    ```    

4. Install the project dependencies using `pip`:

    ```sh
    pip install -r requirements.txt
    ```

## Execution

### Generate Prediction Reports

To generate the prediction reports and images, run the following scripts (the data will be generated in the data folder): `main_prediction_reports.py`:

```sh
python main_prediction_reports.py
python main_paper_reports.py
