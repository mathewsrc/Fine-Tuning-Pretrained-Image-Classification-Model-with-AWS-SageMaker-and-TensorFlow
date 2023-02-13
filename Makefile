setup:
	python -m venv ./.env

source:
	source .env/Scripts/activate

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black scripts/*.py

lint:
	pylint --disable=R,C scripts/*.py	

all:
	install format lint 
