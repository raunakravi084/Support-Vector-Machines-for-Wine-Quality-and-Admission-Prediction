# Support Vector Machines for Wine Quality and Admission Prediction

## Project Overview
This project implements **Support Vector Machines (SVM)** for two tasks:
1. **Support Vector Classifier (SVC)**: Classifies red wine quality based on physicochemical properties using the Wine Quality dataset.
2. **Support Vector Regression (SVR)**: Predicts the chance of graduate admission based on academic and professional metrics using the Admission Prediction dataset.

The project is implemented in a Jupyter notebook (`Support vector Classifier_Regression.ipynb`) using Python and libraries such as scikit-learn, pandas, numpy, and matplotlib. The notebook includes data loading, exploratory data analysis (EDA), preprocessing, model training, evaluation, and hyperparameter tuning. For the SVC, the model achieves an accuracy of approximately 60%, while the SVR model achieves an R² score of 0.8191 after hyperparameter tuning.

## Datasets
### 1. Wine Quality Dataset
- **Description**: Contains 1599 records of red wine samples with 11 physicochemical features and a quality score (3–8).
- **Features**:
  - Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- **Target**: Quality (integer score from 3 to 8)
- **Source**: [UCI Machine Learning Repository](https://raw.githubusercontent.com/aniruddhachoudhury/Red-Wine-Quality/master/winequality-red.csv)
- **Note**: The dataset is loaded directly from a URL in the notebook.

### 2. Admission Prediction Dataset
- **Description**: Contains 500 records of graduate admission applications with 8 features and a chance of admission probability.
- **Features**:
  - GRE Score, TOEFL Score, University Rating, SOP (Statement of Purpose), LOR (Letter of Recommendation), CGPA, Research (binary)
- **Target**: Chance of Admit (continuous value between 0 and 1)
- **Source**: [Kaggle](https://raw.githubusercontent.com/srinivasav22/Graduate-Admission-Prediction/master/Admission_Predict_Ver1.1.csv)
- **Note**: The dataset is loaded directly from a URL in the notebook.

## Requirements
To run the notebook, you need the following Python libraries:
- pandas
- numpy
- scikit-learn
- matplotlib (optional, for visualizations)

Install the dependencies using pip:
```bash
pip install pandas numpy scikit-learn matplotlib
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/svm-wine-admission-prediction.git
   cd svm-wine-admission-prediction
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
   (Create a `requirements.txt` file with the above libraries if needed.)
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook "Support vector Classifier_Regression.ipynb"
   ```

## Usage
1. Open the Jupyter notebook in your environment.
2. Run the notebook cells sequentially to:
   - **Wine Quality (SVC)**:
     - Load and explore the Wine Quality dataset.
     - Perform data preprocessing (feature scaling with `StandardScaler`).
     - Train and evaluate SVC and Logistic Regression models.
     - Compare model performance using accuracy scores.
   - **Admission Prediction (SVR)**:
     - Load and explore the Admission Prediction dataset.
     - Preprocess data (drop `Serial No.`, scale features).
     - Train and evaluate an SVR model.
     - Perform hyperparameter tuning using `GridSearchCV`.
     - Evaluate the best model on the test set.
3. Modify the notebook to experiment with different models, hyperparameters, or datasets if desired.

## Project Structure
- `Support vector Classifier_Regression.ipynb`: Main notebook with the complete workflow for both tasks.
- `README.md`: Project documentation.
- `requirements.txt` (optional): List of required Python libraries.

## Methodology
### 1. Wine Quality Classification (SVC)
- **Data Loading and Exploration**:
  - Load the Wine Quality dataset using pandas.
  - Explore dataset features, distributions, and target variable (`quality`) distribution.
- **Preprocessing**:
  - Split features and target (`quality`).
  - Scale features using `StandardScaler`.
  - Split data into training (67%) and testing (33%) sets using `train_test_split`.
- **Model Training**:
  - Train an SVC model with default parameters.
  - Train a Logistic Regression model for comparison.
- **Evaluation**:
  - Compute accuracy scores for both models.
  - SVC accuracy: ~60%.
  - Logistic Regression accuracy: ~57%.

### 2. Admission Prediction (SVR)
- **Data Loading and Exploration**:
  - Load the Admission Prediction dataset using pandas.
  - Remove `Serial No.` column and strip whitespace from column names.
  - Check for missing values (none found).
- **Preprocessing**:
  - Split features and target (`Chance of Admit`).
  - Scale features using `StandardScaler`.
  - Split data into training (67%) and testing (33%) sets.
- **Model Training**:
  - Train an SVR model with default parameters (RBF kernel).
  - Perform hyperparameter tuning using `GridSearchCV` over `C`, `epsilon`, `gamma`, and `kernel`.
- **Evaluation**:
  - Default SVR: MSE = 0.0049, RMSE = 0.0703, R² = 0.7602.
  - Tuned SVR (best parameters: `C=10`, `epsilon=0.01`, `gamma=0.01`, `kernel='rbf'`):
    - Best cross-validated R²: 0.8000.
    - Test R²: 0.8191.

## Results
### Wine Quality Classification
- **SVC Accuracy**: ~60% on the test set.
- **Logistic Regression Accuracy**: ~57% on the test set.
- **Observation**: The SVC slightly outperforms Logistic Regression, but both models have moderate performance, likely due to class imbalance and the multi-class nature of the target variable.

### Admission Prediction
- **Default SVR**:
  - MSE: 0.0049
  - RMSE: 0.0703
  - R² Score: 0.7602
- **Tuned SVR**:
  - Test R² Score: 0.8191
  - Best Parameters: `C=10`, `epsilon=0.01`, `gamma=0.01`, `kernel='rbf'`
- **Observation**: Hyperparameter tuning significantly improves the SVR's performance, achieving a high R² score, indicating strong predictive capability for admission chances.

## Future Improvements
- **Wine Quality Classification**:
  - Address class imbalance using techniques like SMOTE or class weighting.
  - Experiment with other classifiers (e.g., Random Forest, XGBoost).
  - Perform hyperparameter tuning for SVC using `GridSearchCV`.
  - Include cross-validation for more robust evaluation.
- **Admission Prediction**:
  - Explore feature importance to identify key predictors of admission.
  - Test other regression models (e.g., Random Forest Regressor, Gradient Boosting).
  - Expand the hyperparameter grid for SVR to further optimize performance.
  - Include visualizations of predictions vs. actual values.
- **General**:
  - Add visualizations (e.g., confusion matrix for SVC, scatter plots for SVR).
  - Save trained models using pickle for reuse.
  - Modularize code into functions or scripts for better reusability.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- UCI Machine Learning Repository for the Wine Quality dataset.
- Kaggle for the Admission Prediction dataset.
- Scikit-learn documentation for implementation guidance.

## Contact
For questions or feedback, feel free to reach out via [GitHub Issues](https://github.com/your-username/svm-wine-admission-prediction/issues) or email at your-email@example.com.