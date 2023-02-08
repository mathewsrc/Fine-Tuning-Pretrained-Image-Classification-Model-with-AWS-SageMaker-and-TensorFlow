setup:
	python -m venv ./.env

source:
	source .env/Scripts/activate

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C *.py

test:
	python -m pytest -vv --cov=myrepolib tests/*.py	

all:
	install lint test
