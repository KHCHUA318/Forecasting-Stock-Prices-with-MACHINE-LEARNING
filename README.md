# Forecasting-Stock-Prices-with-MACHINE-LEARNING
Forecasting Stock Prices with Machine Learning:  A Comparative Study of Regression, LSTM, and GRU Models
Libraries required to download:
- Core Libraries: numpy(version <2.0), pandas, matplotlib, seaborn
- Machine Learning: sklearn, keras
- Data Source: yfinance


Quick Summary:
1. dataset.ipynb will download and preprocess the selected dataset from yfinance. After preprocessed, the dataset will be divided into 80% training, 10% validation, and 10% testing.
2. Linear.ipynb will train on the training dataset provided and test on the validation set.
3. Random.ipynb will train on the training dataset provided and test on the validation set.
4. GRU.ipynb will take the entire preprocessed dataset and apply window size=100 before being split into 80% training, 10% validation, and 10% testing. Building and training on the model will be executed and will be tested on validation set for adjustments to the model.
5. LSTM.ipynb will take the entire preprocessed dataset and apply window size = 100 before being split into 80% training, 10% validation, and 10% testing. Building and training on the model will be executed and will be tested on validation set for adjustments to the model.
6. main.py will execute all models and test them all on same testing dataset to ensure fairness. Performance statistics will be provided for each prediction.


How the program works:
Sequence of execution as follows

dataset.ipynb --> Linear.ipynb --> Random.ipynb --> GRU.ipynb --> LSTM.ipynb --> main.py

dataset.ipynb
1. Start by defining the start and end date of the dataset wanted (begin_at, end_at) and also the dataset wanted (Ex: "MCD").
2. Presence of null/missing value will be checked.
3. Presence of duplicates will be checked.
4. A correlation matrix will be plotted to show the correlation between each data feature.
5. Boxplots will be plotted for each feature to find presence of outliers. If outliers are presence, then capping process will be conducted on the feature containing outliers. 
(Note: features containing outliers will be detected however the feature itself requires manually adding to the for loop in Capping outliers cell)
6. Once capped, the preprocessed dataset will be saved into "Preprocessed_MCD.csv".
7. The preprocessed dataset will also be split into 80% training ("MCD_train_data.csv"), 10% validation ("MCD_validation_data.csv"), and 10% testing ("MCD_test_data.csv").

Linear.ipynb
1. Read the training dataset ("MCD_train_data.csv") and validation dataset ("MCD_validation_data.csv").
2. Both training and validation dataset will be split into X and y, X will take features ["High", "Low", "Open", "Volume"] and assigned to variables (X_train, X_val) while y will take feature ["Close"] and assigned to variables (y_train, y_val)
3. X_train and X_val will be scaled using MinMaxScaler and assigned to X_train scaled and X_val_scaled respectively.
4. The Linear Regression model will be fit with X_trained_scaled and y_train. The fitted model will be saved as "Saved_Linear_Model.pkl".
5. The model will then be tested on X_val_scaled and assigned to variable (y_pred).
6. The graph of actual value (y_val) and predicted values (y_pred) will be plotted.
7. Performance on the validation set will be calculated.

Random.ipynb
Process is the same as "Linear.ipynb" with some changes to step 5.
5. The Random Forest model will be fit with X_trained_scaled and y_train along with parameters n_estimator=100. The fitted model will be saved as "Saved_Random_Model.pkl".

GRU.ipynb
1. Read the preprocessed dataset ("Preprocessed_MCD.csv").
2. Calculation on the index for start and end validation after window size = 100 will be conducted.
3. Variables X and y will take features ["High", "Low", "Open", "Volume"] and ["Close"] respectively.
4. Both X and y are scaled using MinMaxScaler.
5. List X_temp and y_temp will contain data of X and y respectively after applying window size=100 (input_size=100). X_temp and y_temp will be converted to numpy arrays for easy access.
6. Both X_temp and y_temp will be split into 80% training (X_train, y_train), 10% validation (X_val, y_val), and 10% (X_test, y_test).
7. If "Saved_LSTM_Model.h5" is present, then model will just be compiled. Otherwise, model will be constructed with various layers and compiled.
8. If "Saved_LSTM_Model.h5" is present, then will just indicate that it is present. Otherwise, model will be fit, trained, and saved as "Saved_LSTM_Model.h5".
9. The model will be tested on X_val and assigned to variable (y_pred). y_pred and y_val are rescaled back to their original form for easier visualization.
10. The graph of actual value (y_val) and predicted values (y_pred) will be plotted.
11. Performance on the validation set will be calculated.

LSTM.ipynb
Process is the same as "GRU.ipynb" with changes of model name to "Saved_GRU_Model.h5".

main.py
1. Define variables to store saved model names.
2. Read the dataset of "Preprocessed_MCD.csv", "MCD_train_data.csv", and "MCD_test_data.csv".
3. Get same features for x and y as before.
4. Scale the x and y data.
5. Display the last 10% of the original "Close" data on a plot which will be used as a reference for prediction.
6. Split the data based on window size=100 for LSTM and GRU prediction.
7. Perform prediction on the last 10% of the original "Close" data using saved models of Linear Regression, Random Forest, LSTM, and GRU.
8. Graph of actual value vs predicted value is plotted for each prediction model.
9. Performance for each prediction model is calculated.


**Important notes**
1. The actual testing happens at "main.py" where all models will be tested on the last 10% of the original size, in this case is 126 data to be predicted.
2. Validation will conducted for each model on their respective .ipynb files in order to perform modifications on them.

**Troubleshooting**
Environment Issues:
1. Ensure that the numpy version is lower than 2.0. Others libraries should be updated to the latest version.
2. Ensure the compatibility of keras and tensorflow versions.

Model Not Saving/Loading:
1. Ensure the .pkl and .h5 files are being saved in the correct directory.
2. Check file permissions if models fail to load.

Data Issues:
1. If data faces missing or null values, manually handle the value using techniques like mean/median imputation or forward fill.
2. If data contains excessive outliers, review and adjust capping thresholds in dataset.ipynb.

Performance Variations:
1. If Graph of predictions does not match expected patterns, ensure that the testing data (MCD_test_data.csv) is consistent during progress.
2. Experiment with different feature combinations or model architectures to improve results.



