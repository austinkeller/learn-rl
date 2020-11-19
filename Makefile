.PHONY: clean venv

venv: requirements.txt
	python -m venv venv
	. ./venv/bin/activate; pip install -U pip; pip install --use-feature=2020-resolver -Ur requirements.txt

clean:
	rm -rf venv
