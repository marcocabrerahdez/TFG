# Trabajo de Fin de Grado

[![PythonVersion](https://img.shields.io/pypi/pyversions/gino_admin)](https://img.shields.io/pypi/pyversions/gino_admin)

## Machine Learning como alternativa a los modelos de simulación para la toma de decisiones

> realizado en `Universidad de la Laguna`

## Objetivo

El objetivo principal de este proyecto es adaptar el modelo de simulación existente para la predicción del curso de la diabetes tipo 1 a modelos de aprendizaje automático (Machine Learning). La idea es comparar los resultados obtenidos por el modelo de simulación con los obtenidos por los modelos de aprendizaje automático y evaluar si estos últimos son capaces de realizar predicciones de forma más rápida y precisa.

Para llevar a cabo esta tarea, se seleccionarán y entrenarán distintos modelos de aprendizaje automático utilizando técnicas y algoritmos adecuados (modelos subrogados) para el problema en cuestión. Una vez entrenados, se compararán los resultados obtenidos por los modelos de aprendizaje automático con los del modelo de simulación en términos de precisión y tiempo de cómputo.

## Resultados

`SHORT SUMMARY AND LINK TO REPORT`

## Uso

Para utilizar los modelos de aprendizaje automático especificados en esta configuración, deberá tener las bibliotecas necesarias instaladas. Por ejemplo, los modelos en esta configuración son parte de la biblioteca sklearn, por lo que necesitará tener esta biblioteca instalada en su entorno de Python.

Una vez que tenga las bibliotecas necesarias instaladas, puede utilizar los siguientes pasos para entrenar y evaluar los modelos:

1. Cargue sus datos en un DataFrame de Pandas y extraiga las características de entrada relevantes y la variable objetivo.

2. Divida los datos en conjuntos de entrenamiento y prueba utilizando una función como sklearn.model_selection.train_test_split.

3. Itere a través de la lista de modelos en la configuración e instancie cada uno utilizando la clase y los hiperparámetros especificados.

4. Entrene cada modelo en el conjunto de entrenamiento utilizando el método fit.

5. Evalúe cada modelo en el conjunto de prueba utilizando el método score y la métrica de evaluación especificada en la configuración (en este caso, r2).

6. Seleccione el modelo con el mejor rendimiento como el resultado final.

También puede considerar utilizar sklearn.model_selection.GridSearchCV para automatizar el proceso de entrenamiento y evaluación de cada modelo con diferentes combinaciones de hiperparámetros. Esto puede ayudarle a encontrar el mejor conjunto de hiperparámetros para cada modelo y mejorar su rendimiento.

## Configuración

> desarrollador principal `Marco Antonio Cabrera Hernández`
>
> contacto `marco.cabrerahdez@gmail.com`
