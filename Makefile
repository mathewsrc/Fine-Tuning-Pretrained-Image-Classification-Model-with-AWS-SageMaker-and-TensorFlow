setup:
	python -m venv ./.env

install_tf:
	pip install --upgrade pip &&\
		python install -r requirements_tf.txt

install_pytor:
	pip install --upgrade pip &&\
		python install -r requirements_pytor.txt

format:
	black *.py

lint:
	pylint --disable=R,C *.py

test:
	python -m pytest -vv --cov=myrepolib tests/*.py	

all:
	install lint test
