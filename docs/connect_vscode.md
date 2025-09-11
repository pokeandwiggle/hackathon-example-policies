# Connect VS Code to JupyterHub

This guide explains how to connect a local VS Code instance to a remote JupyterHub server. This allows you to use your local editor's features while executing code on the remote machine.

### Prerequisites

-   [Visual Studio Code](https://code.visualstudio.com/)
-   [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) for VS Code

### Instructions

#### 1. Generate an API Token

1.  In a browser, go to `<server-address>/jupyterhub/hub/token`.
2.  Click **Request new API token**.
3.  Name the token and click **Request**.
4.  Copy the generated token.

> **Important:** Treat this token like a password. Do not share it.

#### 2. Configure VS Code

1.  Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`).
2.  Run the command `Jupyter: Specify Jupyter server for connections`.
3.  For the URL, enter: `<server-address>/jupyterhub/`.
4.  Paste your API token when prompted.

#### 3. Select the Remote Kernel

1.  Open a Jupyter Notebook (`.ipynb`) file.
2.  Click **Select Kernel** in the top-right corner.
3.  Choose the kernel from the remote JupyterHub server.

Your setup is complete. Code executed in the notebook will now run on the remote server.

