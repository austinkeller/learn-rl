.PHONY: clean venv

venv: requirements.txt
	python -m venv venv
	. ./venv/bin/activate; pip install -U pip; pip install -Ur requirements.txt

clean:
	rm -rf venv
