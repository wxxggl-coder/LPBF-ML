# Overview
This repository provides the implementation of a machine learning assisted optimization framework for LPBFed Ti-B/AA2024.The workflow integrates predictive model (LGBM) with GA to identify high performance composition-process parameter combinations.
# Repository Structure
├── 11.csv          # A synthetic dataset generated to match the statistical characteristicsof the original experimental data. It is intended for code execution and validation. It does not represent the original raw experimental dataset used in this study.
├── GA_iteration.py
├── GA_strength.py
├── Train_of_models_plasticity_LGBM.py
├── Train_of_models_plasticity_Linear.py
├── Train_of_models_plasticity_Ridge.py
├── Train_of_models_plasticity_SVM.py
├── Train_of_models_plasticity_Extra.py
├── Train_of_models_plasticity_Stacking.py
├── Train_of_models_strength_LGBM.py
├── Train_of_models_strength_Linear.py
├── Train_of_models_strength_Ridge.py
├── Train_of_models_strength_SVM.py
├── Train_of_models_strength_Extra.py
├── Train_of_models_strength_Stacking.py
├── VIF.py
├── bianli_2inputs_plasticity.py
├── bianli_2inputs_strength.py
├── bianli_3inputs_plasticity.py
├── bianli_3inputs_strength.py
├── bianli_4inputs_plasticity.py
├── bianli_4inputs_strength.py
├── bootstrap.py
├── hyperopt_plasticity.py
├── hyperopt_strength.py
├── image_recognition.py
├── optuna_plasticity.py
├── optuna_strength.py
├── person_of_parameters.py
