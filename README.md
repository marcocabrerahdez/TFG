# Trabajo de Fin de Grado 

This repository contains my bachelor's thesis in computer engineering submitted on May 24, 2023, which received a grade of 10 from the examining board on June 2, 2023.

## Machine Learning as an alternative to simulation models for decision making

> conducted at `Universidad de la Laguna`

## Abstract

There are many fields of application where it is common to use simulation to predict the long-term behavior of the modeled system. As the complexity of these models increases, so does the computational time required to obtain results. An alternative is to create a model with machine learning techniques that mimics the behavior of the original model, but whose computation time is almost instantaneous: these are called surrogate models. 

In this project, a comparison has been made between a simulation model to evaluate the effectiveness, safety and cost-effectiveness of real-time continuous interstitial glucose monitoring systems for type 1 diabetes and surrogate machine learning models, analyzing performance metrics such as the coefficient of determination and the mean absolute percentage error. The results obtained have shown that surrogate models are able to accurately and efficiently capture the behavior of the modeled system in significantly less time, making them a valuable tool for real-time decision making applications.

## Method & results

[Report](https://github.com/marcocabrerahdez/TFG/blob/main/Report.pdf)

## Conclusions
In this project we have been able to adapt machine learning models to a simulation model using surrogate models. As shown in Chapter 3, this has allowed faster predictions to be made without a noticeable loss of accuracy compared to the simulation model described, which is especially useful in situations where the simulation model is very slow or costly to run.

Several machine learning models have been selected and trained using techniques and algorithms suitable for the problem at hand. The evaluation results show that for the first experiment an R2 of 0.9984340817 maximum and 0.999294901 for the second experiment have been obtained. 
After the evaluation and comparison of the results obtained by the machine learning models, the models providing the best results have been selected to be connected to the web application prototype.

In the future, these models should be adjusted from real data to further improve their accuracy and predictive ability in these situations. The possibility of using reinforcement learning techniques could also be explored to optimize the results and improve the decision-making capability of the models.

In short, the use of surrogate machine learning models as an alternative to simulation models offers great potential for improving the efficiency and accuracy of models, which may have important implications in a wide range of areas, from engineering to resource management and decision making.

## Authors

> main developer `Marco Antonio Cabrera Hernández`
> lead professor `Iván Castilla Rodríguez`
> associate professor `Rafael Arnay del Arco`
