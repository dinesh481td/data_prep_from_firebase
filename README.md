# Installation instruction

* clone the repo
* Install pipx
    * `pip install pipx`
    * `pipx ensurepath`
* Install poetry
    * `pipx install poetry`
* Run `poetry install`
    * This should create a virtual environment and install all dependencies in the venv
If there is an error with keyring:
   To check the installation process:poetry install -vvv
   export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
If you are using custom package
   * `poetry self update && poetry self add keyrings.google-artifactregistry-auth`
   * `poetry install`
* # Setup pre-commit and pre-push hooks
poetry run pre-commit install -t pre-commit

poetry run pre-commit install -t pre-push

* # Documentation Link in Notion:
   * https://www.notion.so/Data-pipelines-581043adc8be470f9203e87ab7d74679
