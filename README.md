# Selective-Cytotoxicity-ML_reinforced_GA
## Overview
The use of nanomaterials, particles with size less than 100nm, are increasing used in food processing, cosmetics and medical applications. This raise the concern of toxicity of nanoparticles in human health and to address these issues researchers are developing predictive models to predict the cytotoxicity of the nanoparticles. Eventhough these developed models could predict cytotoxicity of nanomaterials it still fail to address one of the key issue, selectivity of the nanoparticle. NMs are considered to be non-selective and treated as passive carriers, where additional functionalities are given through additional chemical modifications. However, these modifications not only make the system more complex hindering its clinical translation, but also change its behavior in the living organisms in unpredictable ways altering therapeutic action of the delivered drug molecules. At the same time, despite metabolic differences between healthy and pathological cells, there are no pipelines for automatic computational design of selective NMs. In this work we for the first time propose a powerful and fully computational screening approach to massive targeted discovery of selectively cytotoxic inorganic NMs.
In this project, for the first time, we combined a machine learning model with a genetic algorithm tool to discover potential inorganic nanoparticles with selective toxicity against specific cell lines. 
## Guidelines
The project combine two parts, Machine learning and Genetic Algorithm. Machine learning is used to predict the cytotoxicity of nanoparticle while genetic alforithm helps in generating and screening selectively cytotoxic nanoparticles. 
The repository consist all the necessary files and results. To make it more convinient and simples various files are created with their own purpose with multiple function in it. There are file for data processing (), model selection and tuning (), and genetic algorithm ()

### steps for data preprocessing
- Inside the data folder there is a file named new_combined.csv, which consist all the datasets that are used for model building, validation and generating individuals for genetic algorithm. Additional datasets can be added in this files to improve the model predictibility in the future.
- Data preprocessing can be carried out in these collected dataset using the file data_preprocessing.py

### Model selection
- Among various regression models, best model can be identified and selected by running the file ml_model_comparision.py and the results will be stored in results/lazypredict
- XGB and LGBM regressor model are selected and optimized by hyperparameter tuning
- 10 fold cross validation is carried out before testing the models performance in validation set (unseen data)

### Genetic Algorithm
- ga_compound_generation.py file help us to generate nanomaterials with unique features. The toxicity of these nanomaterials is tested on two different cell line (for eg. hepatocytes (normal liver cell ) and HepG2 (cancer liver cell)) to discover selectivity of nanoparticle on one cell over another. 
- ga_crossing_mutation.py file mutate, cross over and evolve the nanoparticle feature for improving the feature of nanoparticles.
-ga_main.py is the main file and it generate nanomaterials with unique features, introduce cross_over and mutation for improving the features and run the process for specific generation for selecting best selectively cytotoxic nanomaterial for provided cell lines. And the result is stored in results/GA 
