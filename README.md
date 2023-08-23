# Machine Learning Reinforced Genetic Algorithm for Massive Targeted Discovery of Selectively Cytotoxic Inorganic Nanoparticles

## Abstract
Nanoparticles (NPs) have been employed as drug delivery systems (DDSs) for several decades, primarily as passive carriers, with limited selectivity. However, recent publications have shed light on the emerging phenomenon of NPs exhibiting selective cytotoxicity against cancer cell lines, attributable to distinct metabolic disparities between healthy and pathological cells. In this study, we revisit the concept of NPs selective cytotoxicity, and for the first time propose a high-throughput in silico screening approach to massive targeted discovery of selectively cytotoxic inorganic NPs. In the first step, we train a gradient boosting regression model to predict viability of NP-treated cell lines. The model achieves mean cross-validation (CV) Q2 = 0.80 and root mean square error (RMSE) of 13.6. In the second step, we developed a machine learning (ML)-reinforced genetic algorithm (GA), capable of screening >14,900 candidates/min, to identify the best-performing selectively cytotoxic NPs. As proof-of-concept, DDS candidates for the treatment of liver cancer were screened on HepG2 and hepatocytes cell lines resulting in Ag NPs with selective toxicity score of 42%. Our approach opens the door for clinical translation of NPs, expanding their therapeutic application to a wider range of chemical space of NPs and living organisms such as bacteria and fungi.
 
## Guidelines
There are two repositories GA_main and GA_fixed_zeta. The GA_main consists separate folders for code, data, models, results and other preprocesing data. GA_fixed_zeta is diferent than GA_main in case of use of zeta potential as a fixed parameter for prediction rather than as a independent variable. 

### Data
Using data preprocessing steps present in data_preprocesing.py preprocessed dataset is generated and kept in folder: data 

### Model selection
- Performance of various regression models were compared and evaluated using ml_model_comparision.py and the results were be stored in other/Model_comparision_test.csv
- Hyperparameter tuning were carried out in two best performing XGB and LGBM regressor model
- 10 fold cross validation is carried out before testing the models performance in validation set (unseen data)

### Genetic Algorithm
- All related files were stored in code folder.
- ga_compd_generation.py file help us to generate nanomaterials with unique features. The toxicity of these nanomaterials is tested on two different cell line (for eg. hepatocytes (normal liver cell ) and HepG2 (cancer liver cell)) to discover selectivity of nanoparticle on one cell over another. 
- ga_crossing_mutation.py file mutate, cross over and evolve the nanoparticle feature for improving the feature of nanoparticles.
-ga_main.py is the main file which generate nanomaterials with unique features, introduce cross_over and mutation, calculate fitness of nanoparticles and screen for selectively cytotoxic nanoparticles 
- results were available in results folder
