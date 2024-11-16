
---
id: python_poetry
title: Python Dependency "Adulting" with Poetry
---

# Python Dependency "Adulting" with Poetry

If you're interested in this topic enough to have clicked on it, I'll assume you've already skimmed a few accounts of people trying to untangle the python install issues they've made for themselves.  My account is no different.  Below is a copy/paste of useful notes to self on how to use poetry along with pyenv and jupyter.  My highlights:

* I did look at uv vs poetry.  I went with poetry in the end because it looks more likely to remain free and well-maintained.  Also, I had learned to use and appreciate it at Zillow via their ML Ops team.

* Note the bit on how to get poetry to find the python version you've installed with pyenv.  This was the part that gave me the most trouble in using poetry.

* Note the stuff on using jupyter notebooks in a poetry/pyenv context.  I found that poetry couldn't resolve "jupyter" when installed inside the init process, but the below way works.

HTH....


# OS and Python Setup

## Sources:
https://medium.com/nerd-for-tech/what-is-pipenv-5b552184852
https://realpython.com/intro-to-pyenv/
https://evinsellin.medium.com/an-opinionated-python-setup-for-mac-2021215dba8f

## Terminal/Unix Setup

### find a file globally



## penv

Pyenv allows you to update and install versions of python.

### Sources

https://mac.install.guide/python/install-pyenv

### Install Pyenv

$ brew install pyenv

### Update Profile
```
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

$ source ~/.zprofile #
```

### Verify pyenv was installed

`$ pyenv --version`

### Install dependency manager xz

`brew install xz`

### Actually Install Another Version of Python:

`pyenv install 3.12`

### List Versions of Python installed by pyenv:

`echo $(pyenv root)` --this will tell you where pyenv is storing its versions and shims.
`cd (pyenv root)/versions/`
`ls -a` -- look for version numbers


## Poetry

### Re: Poetry
https://www.youtube.com/watch?v=0f3moPe_bhk

11.3.2024 decided to go with poetry instead of competing uv product, because it's about as good and should stay free with good support.

### Install Poetry
`pip install poetry`

### Cause poetry to always install the venv inside your project.
--This is generally a good idea bc it makes the venv easier to find.
`poetry config virtualenvs.in-project true`

### Add Poetry to an Existing Project
https://github.com/python-poetry/poetry/issues/46
`poetry init`

### Install/Remove Dependencies:

Don't do the .toml file by hand; use `poetry add [packagename]` instead.
Remove a package; use `poetry remove [packagename]` .

### Create virtual environment:

`poetry install`

### Make Poetry use the python version you installed with pyenv:

Source: https://stackoverflow.com/questions/70950511/using-poetry-with-pyenv-and-having-python-version-issues

```
poetry init # ... during the init phase, you have to specify the python version you want to use.
            # do it precisely; I'm not sure how to wildcard python versions e.g. "3.10.*"

cd test-project/ ## cding into existing project

pyenv local 3.10.15  ## having gone through the init phase, cause pyenv to point to its local install

poetry env use 3.10 # now cause poetry to overwrite its default version to your desired local pyenv version
                      # which you also specified in the init phase.
```

### Get info about virtual environement:

`poetry env info`

### Activate poetry environment

`poetry shell`

### Get out of poetry environment:

`exit`

### Initialize jupyter notebook:

```
$ poetry init ## go through the init phase without actually adding the below.
## NOTE: you can't actually add jupyter within the initial "init" phase for some reason.
# you have to add jupyter post init, as follows:

$ poetry add numpy # libraries you want to use
$ poetry add -D jupyter # libraries for development use only

poetry run jupyter notebook
```
### List active poetry environments:

`poetry env list`

### Deactivate poetry environment:

`deactivate` --this is prob better than `exit` in most cases

### Delete a virtual environment

--Simply delete the .venv folder that was created.


