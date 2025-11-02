
# 📈 Data and reports for transparency

This project generates **prediction reports** using classification models, specifically **Random Forest** and **Dummy classifiers**. The reports include **graphs** and detailed **analyses** of the prediction results.

All data utilized is available in the **`data`** folder.

After running the Python scripts, all generated images and reports will be saved in the **`reports`** folder.

The following sections describe how to set up the project.

-----

# 🛠️ Installation Instructions

This project can be run using two main alternatives:

  * Using a **Virtual Environment**.
  * Using **VS Code** with **Docker** and **Dev Containers**.

The following sections explain the installation instructions for each of the aforementioned options.

-----

## 🐍 Install Using a Virtual Environment

### Prerequisites

  * **Python 3.13** or higher.
  * **`pip`** (Python package manager).

### Installation

1.  Clone the repository to your local machine:

    ```sh
    git clone <REPOSITORY_URL>
    cd <REPOSITORY_NAME>
    ```

2.  Create a virtual environment in a **`.venv`** folder:

    ```bash
    # Create the virtual environment
    python -m venv .venv
    ```

3.  Once the virtual environment is created, activate it with the following command:

    ```bash
    # Activate the virtual environment on Windows
    .venv\Scripts\activate
    ```

    ```bash
    # Activate the virtual environment on macOS/Linux
    source .venv/bin/activate
    ```

4.  Install the project dependencies using **`pip`**:

    ```sh
    pip install -r requirements.txt
    ```

-----

## 🐳 Install Using VS Code with Docker and Dev Containers

This method allows you to use a complete, pre-configured development environment within a Docker container, ensuring all dependencies are isolated and consistent.

### Prerequisites

  * **Docker Desktop** (or equivalent Docker engine) installed and running.
  * **Visual Studio Code** (VS Code).
  * **VS Code Dev Containers Extension** installed.

### Installation

1.  Clone the repository to your local machine:

    ```sh
    git clone <REPOSITORY_URL>
    cd <REPOSITORY_NAME>
    ```

2.  **Open the project folder** in Visual Studio Code.

3.  When VS Code opens, a notification should appear prompting you to **"Reopen in Container"**. Click this button.

      * *If the notification doesn't appear:* Open the **Command Palette** (Ctrl+Shift+P or Cmd+Shift+P) and select **`Dev Containers: Reopen in Container`**.

4.  VS Code will build the Docker image (if it's the first time) and connect to the running container. The terminal inside VS Code will automatically be ready with the project dependencies pre-installed based on the **`.devcontainer/devcontainer.json`** configuration.

You are now ready to generate the reports.

-----

# 🚀 Generate Prediction Reports

To generate the prediction reports and images, run the following scripts **from the root directory of the project** (the outputs will be generated in the **`reports`** folder):

```sh
python main_prediction_reports.py
python main_paper_reports.py
```
