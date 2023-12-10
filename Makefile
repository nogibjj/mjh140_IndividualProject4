venv:
	python3 -m venv myenv
		source myenv/bin/activate

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:	
	black *.py

lint:
	ruff check *.py

container-lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

refactor: format lint

deploy:
	#deploy goes here
		
all: venv install format lint refactor deploy

######################################

venv:
	python3 -m venv myenv
		source myenv/bin/activate

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:	
	black *.py 

lint:
	ruff check *.py 

container-lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

refactor: format lint
		
all: install lint test format deploy
