# Machine Learning Reinforced Genetic Algorithm for Massive Targeted Discovery of Selectively Cytotoxic Inorganic Nanoparticles

## Abstract
Nanoparticles (NPs) have been employed as drug delivery systems (DDSs) for several decades, primarily as passive carriers, with limited selectivity. However, recent publications have shed light on the emerging phenomenon of NPs exhibiting selective cytotoxicity against cancer cell lines, attributable to distinct metabolic disparities between healthy and pathological cells. In this study, we revisit the concept of NPs selective cytotoxicity, and for the first time propose a high-throughput in silico screening approach to massive targeted discovery of selectively cytotoxic inorganic NPs. In the first step, we train a gradient boosting regression model to predict viability of NP-treated cell lines. The model achieves mean cross-validation (CV) Q2 = 0.80 and root mean square error (RMSE) of 13.6. In the second step, we developed a machine learning (ML)-reinforced genetic algorithm (GA), capable of screening >14,900 candidates/min, to identify the best-performing selectively cytotoxic NPs. As proof-of-concept, DDS candidates for the treatment of liver cancer were screened on HepG2 and hepatocytes cell lines resulting in Ag NPs with selective toxicity score of 42%. Our approach opens the door for clinical translation of NPs, expanding their therapeutic application to a wider range of chemical space of NPs and living organisms such as bacteria and fungi.


## Guidelines
There are two folders in this repo: `GA_main` and `GA_fixed_zeta`. The `GA_main` provides the code, data, models and results reported in the original manuscript. `GA_fixed_zeta` is an alternative version of the genetic algorithm that we implemented during the revision phase. This GA implementation treats zeta potential as a fixed parameter predicted based on the other input variables. This may be preferable for some materials of specific concentrations and radii. 

### Data
For each of the two GA versions, we provide the preprocessed CSV data files to ensure reproducibility.

### Model selection
- Performance of various regression models were compared and evaluated using `ml_model_comparision.py` and the results were saved in `other/Model_comparision_test.csv`
- Hyperparameter tuning was carried out for the XGB and LGBM regressor models
- 10-fold cross-validation was performed before testing the model performance on the validation set (previously unseen data)

### Genetic Algorithm
- All related files are stored in the `code` folder
- `ga_compd_generation.py` file is used to generate nanomaterials with unique features. The toxicity of these nanomaterials was tested on two different cell lines to establish selectivity of nanoparticles on one cell over another: hepatocytes (normal liver cell) and HepG2 (cancer liver cell)  
- `ga_crossing_mutation.py` file is used to mutate, cross over and evolve the nanoparticle features during the GA
- `ga_main.py` is the main file generating nanomaterials with unique features, introducing cross-overs and mutations, and calculating the fitness score for screening the selectively cytotoxic nanoparticles 

### Results
The results obtained for each version of the GA are stored in the corresponding `results` folder.
