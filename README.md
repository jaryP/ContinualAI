# Continual learning framework

This is a Continual Learning library based on **Pytorch**, mainly born for personal use, which can be used for fast 
prototyping, training and to compare different build-in methods over a various numbers of scenarios and benchmarks.  

# Installation

Type
    
    pip install continual-learning

# Continual learning framework

The library is organized in four main modules:

- [Benchmarks](continual_learning/datasets): This module contains the most used dataset in CL, reimplemented to give more flexibility.
- [Logging](continual_learning/scenarios): This module provides different supervised scenarios which you can use in combination with a dataset to create your own Cl scenario.
- [Extras](continual_learning/methods): It contains many Cl methods, that can be easily used and evalauted.
- [Training](continual_learning/backbone_networks): This module contains many popular networks used to extract the features from the input samples.
- [Evaluation](continual_learning/eval): This modules provides a unified way to evaluate a method over a flexible numbers of metrics/  
- [Models](continual_learning/solvers): In this module you will find different solvers, used to classify the features extracted by a backbone network.

Disclaimer
----------------
This is a framework which is born to improve coding, and the reproducibility of the papers in which I have worked during the years. 
Being constantly under development, it may be unstable.  
