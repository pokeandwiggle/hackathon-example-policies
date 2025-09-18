# Connect VS Code to a Remote JupyterHub Server

This guide outlines the process of connecting a local Visual Studio Code instance to a remote JupyterHub server. This setup allows you to leverage the features of your local editor while executing code on the remote server's resources.

## Prerequisites

*   [Visual Studio Code](https://code.visualstudio.com/) installed locally.
*   The [JupyterHub extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyterhub) for VS Code installed.

## Connection Steps

### 1. Log into JupyterHub

First, establish a session on the JupyterHub server through your web browser.

1.  Navigate to `http://<server-address>/jupyterhub/`.
    *   **Note**: The trailing slash `/` is required.
2.  Log in using the credentials:
    *   **Username**: `minga`
    *   **Password**: `minga`
3.  After a successful login, copy the new URL from your browser's address bar. It will be in the format `https://<server-address>/jupyterhub/user/minga/lab`.

### 2. Configure VS Code Kernel

Next, configure VS Code to use the remote server as its kernel.

1.  Open a Jupyter Notebook in VS Code, such as [notebooks/00_setup.ipynb](../notebooks/00_setup.ipynb)
2.  Click the **Select Kernel** button in the top-right corner of the notebook editor.
3.  Choose **Select Another Kernel...** from the command palette dropdown.
4.  Select **Existing Jupyter Server...**.
    ![Select Existing Jupyter Server](https://code.visualstudio.com/assets/docs/datascience/jupyter-kernel-management/select-enter-server-url.png)
5.  Paste the URL you copied earlier, but **change the protocol from `https://` to `http://`**. The URL should look like this:
    `http://<server-address>/jupyterhub/user/minga/lab`
    *   **Important**: Using `https://` will result in a connection error.
6.  When prompted, enter the same credentials (`minga`/`minga`).

Your VS Code instance is now connected to the remote JupyterHub server. Any code executed in the notebook will run on the server.
