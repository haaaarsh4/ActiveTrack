# Active Track

The Python scripts were developed to handle, visualize, and model data from accelerometers and gyroscopes. These scripts were pivotal in constructing a machine learning model capable of categorizing barbell exercises and tallying repetitions. This project stemmed from an instructional course led by Dave Ebbelaar, available on YouTube.

## Dataset
The dataset was curated through a Meta Motion sensor incorporated within a wristband worn by diverse individuals while engaging in five distinct exercise types: bench press, deadlift, overhead press, rowing, and squats. These exercises were segmented into medium sets comprising 10 repetitions and heavy sets comprising 5 repetitions. All series data were captured using a Bluetooth-connected phone and subsequently exported in .csv format for analysis and processing.

## Processing Raw Data

The raw data, stored in .csv files, underwent initial processing. Initially, the files were read, generating two distinct DataFrames: one containing accelerometer data and another containing gyroscope data. Subsequently, these DataFrames were merged into a unified dataset, followed by a frequency conversion process.

## Data Visualization

Various types of visualizations have been created to better understand the accelerometer and gyroscope data.

1.Compare medium vs. heavy sets
2.Compare participants
3.Compare exercises

## Detecting Outliers in Sensor Data

Checking whether there are any outliers (extreme values) in our data that we want to remove using various methods(IQR, Chauvenet’s Criterion, Local Outlier Factor (LOF))
The Chauvenet’s Criterion method was chosen, and the outliers detected using this method were replaced with NaN.

## Feature Engineering

After merging the datasets, interpolation techniques were employed to fill missing or NaN values within rows. Subtle noise, distinct from outliers, was effectively filtered out to enhance data quality. Further analysis was conducted to pinpoint segments within the dataset that contribute significantly to its variance. Following this, additional features were engineered, encompassing numerical, temporal, and frequency attributes. Additionally, cluster-based features were integrated to enrich the dataset.

## Predictive Modelling

Experiment with feature selection, model selection and hyperparameter tuning with grid search to find the combination that results in the highest classification accuracy.

Generated separate sets for training and testing.
Divided the features into subsets.
Conducted forward feature selection employing a simple decision tree.
Executed a grid search to identify optimal hyperparameters and select the model.
Illustrated a grouped bar plot to compare the outcomes.
Identified the superior model and assessed its performance.
Segregated training and testing data based on individual participants.
Employed the best model again and assessed its performance.

## Counting Repetitions

Visualizing the data to spot patterns, configuring the LowPassFilter, applying and fine-tuning the LowPassFilter settings, developing a function to count repetitions, establishing a benchmark dataframe
