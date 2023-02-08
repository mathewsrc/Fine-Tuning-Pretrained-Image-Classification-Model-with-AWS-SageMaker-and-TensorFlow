setup:
	python -m venv ./.env

install:
	pip install --upgrade pip &&\
		python install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C *.py

test:
	python -m pytest -vv --cov=myrepolib tests/*.py	

all:
	install lint test
