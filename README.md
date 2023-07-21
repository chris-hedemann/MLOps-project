# MLOps Project

## Project Description

In this project I will create a Github Action that trains a model, refactored from a pre-exisiting juypter notebook, and uses DVC for the versioning of the train data. This model will be used in an API and monitored with evidently, prometheus and grafana.

## Project Structure

The project structure is as follows:

- Take the code from the training notebook and put it into a python file 
- Create a Github Action that trains the model and tracks it with MLflow and uses DVC for the versioning of the train data
- Update the API that uses this model and deploy the API
- Monitor the API and the model using evidently, prometheus and grafana
- Send data to the API and monitor the data with evidently
- Use the Github Action to retrain the model with new data if you see data drift 

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
``````
