ObesityClassification: Predicting Obesity Levels Using Machine Learning

Overview: ObesityClassification is a machine learning project aimed at predicting obesity levels based on various features. 
          The project utilizes multiple machine learning models and evaluates their performance to determine the most accurate approach. 
          The models considered include Random Forest, Support Vector Machine (SVM), Logistic Regression, K-Nearest Neighbors (KNN), Gradient Boosting, and a Stacking Ensemble method.

Project Structure:
The repository contains the following files:

        1. ObesityDataSet_raw_and_data_sinthetic.csv: The raw and synthetic dataset used for training and evaluation.
        2. Dataset_Info.py: Generates an exploratory data analysis report with visualizations including histograms, box plots, bar plots, and a correlation heatmap.
        3. report_template.html: The template used for generating the HTML report.
        4. report.html: The generated HTML report from the exploratory data analysis.
        5. Data_Preprocessing.py: Preprocesses the dataset by mapping binary variables, encoding categorical variables, and standardizing numeric features.
        6. processed_dataset.csv: The processed dataset ready for model training.
        7. FeatureSelection_Brute.py: Implements brute-force feature selection to identify the best subset of features and runs the models.
        8. ErrorRateAnalysis.py: Analyzes the error rates of the best model.


Getting Started

Prerequisites: Ensure you have the following libraries installed:  pandas, numpy , matplotlib, seaborn, scikit-learn, jinja2, plotly

You can install these libraries using pip:

          Copy code:
                   pip install pandas numpy matplotlib seaborn scikit-learn jinja2 plotly

Exploratory Data Analysis:

Generate an exploratory data analysis report using Dataset_Info.py:
        
         Copy code:
                  python Dataset_Info.py
The script creates visualizations such as histograms, box plots, bar plots, and a correlation heatmap. These visualizations are saved as images and embedded in the report.html file.



Data Preprocessing:
Load and preprocess the dataset using Data_Preprocessing.py:

        Copy code:
                 python Data_Preprocessing.py
This script maps binary variables, manually encodes ordinal variables, performs one-hot encoding for nominal variables, label encodes the target variable, and standardizes numeric variables. 
The processed dataset is saved as processed_dataset.csv.



Model Training and Evaluation:
Train and evaluate the models using FeatureSelection_Brute.py:

        Copy code:
                python FeatureSelection_Brute.py
This script performs the following steps:

      1. Splits the data into training and test sets.
      2. Initializes various classifiers, including the Stacking Classifier.
      3. Defines performance metrics (Accuracy, Precision, F1 Score).
      4. Trains and evaluates classifiers using K-Fold cross-validation with multiple k values (10, 20, 30).
      5. Selects features based on Random Forest importance and performs brute-force feature selection.
      6. Evaluates classifiers after feature selection.
      7. Saves error details for the best-performing model.
      8. Plots the results, showing the variation of accuracy, precision, and F1 score with k folds.



Error Analysis: 
Analyze misclassification errors using ErrorRateAnalysis.py:

       Copy code:
               python ErrorRateAnalysis.py
This script loads the misclassification errors, performs inverse transformation to get the original scale of features, and generates a matrix and scatter plot to visualize the misclassifications.

Results and Discussion:
The results of the model evaluations are summarized in the report.html file. Key findings include:

    1. The Stacking Classifier consistently achieved the highest accuracy, precision, and F1 score across different k values.
    2. Random Forest and Gradient Boosting also performed well but had slightly lower accuracy as the number of folds increased.
    3. Error analysis highlighted that most misclassifications occurred between adjacent weight categories.

Future Work:

    1. Enhance feature selection using advanced techniques like recursive feature elimination and genetic algorithms.
    2. Fine-tune hyperparameters of individual models within the ensemble to improve predictive accuracy and reduce computational time.
    3. Explore other ensemble methods such as boosting (e.g., AdaBoost) and bagging (e.g., Bagged Trees).
    4. Adapt the model for real-time prediction scenarios and integrate it into healthcare applications.
    5. Incorporate a more diverse dataset, including different demographics and geographical regions, to improve generalizability and robustness.
