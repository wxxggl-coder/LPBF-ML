# Overview
This repository provides the implementation of a machine learning assisted optimization framework for LPBFed Ti-B/AA2024.The workflow integrates predictive model (LGBM) with GA to identify high performance composition-process parameter combinations.
# Repository Structure
├── 11.csv  
    Synthetic dataset generated to match the statistical characteristics of the original experimental data.  
    Intended for code execution and validation only; does not represent the original raw dataset.

├── GA_iteration.py  
    Performs 100 independent GA runs on the trained LGBM strength surrogate model, records the optimal parameter combinations and predicted strength from each run, and exports statistical summaries and visualizations for stability analysis.

├── GA_strength.py  
    Applies GA to optimize process parameters for tensile strength.

├── Train_of_models_plasticity_LGBM.py  
    Train of the LightGBM model for plasticity prediction.

├── Train_of_models_plasticity_Linear.py  
    Trains of the linear regression model for plasticity prediction.

├── Train_of_models_plasticity_Ridge.py  
    Train of the Ridge regression model for plasticity prediction.

├── Train_of_models_plasticity_SVM.py  
    Train of the SVM model for plasticity prediction.

├── Train_of_models_plasticity_Extra.py  
    Train of the Extra Trees model for plasticity prediction.

├── Train_of_models_plasticity_Stacking.py  
    Trains of the stacking ensemble model for plasticity prediction.

├── Train_of_models_strength_LGBM.py  
    Train of the LightGBM model for strength prediction.

├── Train_of_models_strength_Linear.py  
    Train of the linear regression model for strength prediction.

├── Train_of_models_strength_Ridge.py  
    Train of the Ridge regression model for strength prediction.

├── Train_of_models_strength_SVM.py  
    Train of the SVM model for strength prediction.

├── Train_of_models_strength_Extra.py  
    Train of the Extra Trees model for strength prediction.

├── Train_of_models_strength_Stacking.py  
    Trains of the stacking ensemble model for strength prediction.

├── VIF.py  
    Used to perform variance inflation factor (VIF) analysis.

├── bianli_2inputs_plasticity.py  
    Exhaustive search of two-feature combinations (plasticity).

├── bianli_2inputs_strength.py  
    Exhaustive search of two-feature combinations (strength).

├── bianli_3inputs_plasticity.py  
    Exhaustive search of three-feature combinations (plasticity).

├── bianli_3inputs_strength.py  
    Exhaustive search of three-feature combinations (strength).

├── bianli_4inputs_plasticity.py  
    Exhaustive search of four-feature combinations (plasticity).

├── bianli_4inputs_strength.py  
    Exhaustive search of four-feature combinations (strength).

├── bootstrap.py  
    Bootstrap resampling for uncertainty and robustness analysis.

├── hyperopt_plasticity.py  
    Hyperparameter optimization of plasticity predictive model using Hyperopt.

├── hyperopt_strength.py  
    Hyperparameter optimization of strength predictive model using Hyperopt.

├── image_recognition.py  
    Image-based precipitations quantification analysis.

├── optuna_plasticity.py  
    Hyperparameter optimization of plasticity predictive model using Optuna.

├── optuna_strength.py  
    Hyperparameter optimization of strength predictive model using Optuna.

├── person_of_parameters.py  
    Pearson correlation analysis.
# Contact
For further questions, please contact the authors.    
