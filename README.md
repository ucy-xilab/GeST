# GeST - SAGA

GeST (Generating Stress-Tests) is a framework for automatic hardware stress-test generation.

A scientific paper about GeST is published in ISPASS 2019 https://ieeexplore.ieee.org/document/8695639 
Hadjilambrou, Z., Das, S., Whatmough, P.N., Bull, D. and Sazeides, Y., 2019, April. GeST: An Automatic Framework For Generating CPU Stress-Tests. In 2019 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS) (pp. 1-10). IEEE.

SAGA (Surrogate-Assisted Genetic Algorithm) is an enhancement to the GeST framework that reduces the number of costly measurements during stress-test generation by integrating a surrogate (prediction) model into the genetic algorithm (GA). SAGA uses a subset of real measurements to train a predictive model that estimates fitness, thus accelerating convergence while preserving result quality.
