venv: requirements.txt
	python -m venv venv
	. ./venv/bin/activate
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

clean:
	rm -rf venv
