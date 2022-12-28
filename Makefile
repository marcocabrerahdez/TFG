run:
	python3 src/main.py

pylint:
	pylint --rcfile=.pylintrc src/**/*.py

help:
	@echo "make run"
	@echo "       Run the program"
	@echo "make pylint"
	@echo "       Run pylint on the source code"
	@echo "make help"
	@echo "       Show this help"