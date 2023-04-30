run:
	python3 src/main.py $(OPTIONS) $(DATA) $(SHEET) $(MODEL_CONFIG) $(COMPARE) $(PLOT)

dev:
	cd app && npm run dev

server:
	python3 src/server.py

lint:
	flake8

help:
	@echo "make run"
	@echo ""
	@echo "       Ejecuta el programa."
	@echo "       OPTIONS: -h, -v, -f, --help, --version, --file"
	@echo "       	-h | --help, --help: Muestra la ayuda."
	@echo "       	-v | --version, --version: Muestra la versión."
	@echo "       	-f | --file: Activa el modo para pasar archivos por línea de comandos."
	@echo "       DATA: Archivo de datos. Debe ser un archivo Excel."
	@echo "       SHEET: Hoja del archivo de datos."
	@echo "       MODEL_CONFIG: Archivo de configuración del modelo. Debe ser un archivo JSON."
	@echo "       COMPARE: Archivo de comparación. Debe ser un archivo JSON."
	@echo ""
	@echo "make pylint"
	@echo "       Ejecuta pylint en el código fuente."
	@echo ""
	@echo "make help"
	@echo " 		 Muestra esta ayuda."