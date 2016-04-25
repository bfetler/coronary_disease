## Coronary Heart Disease
How well can we predict coronary heart disease from patient data?  An early dataset is given in the [UCI Machine Learning Heart Disease Dataset](http://archive.ics.uci.edu/ml/datasets/Heart+Disease).  

Data was collected from 303 patients at the Cleveland Clinic, both without and with varying degrees of coronary heart disease.  Since there are so few data points, patients with heart disease were grouped together for a binary target variable.  Some of the seventy-five columns of original data were corrupted, and replaced with fourteen columns by the data author.  After data cleaning, 297 patients were left.  Despite the small size, it is a reasonable dataset to start exploring coronary disease prediction.  

#### Exploration
Data exploration and prediction is given in __coronary_predict.py__.  The data was randomly split into 70% training data and 30% test data.  A scatter matrix of training data shows some correlation between variables, but no strong trends are visible (are they?).  

<img src="https://github.com/bfetler/coronary_disease/blob/master/coronary_disease_plots/scatter_matrix.png" alt="scatter matrix" />

Histograms of train and test data typically show similar patterns, so that variable column values are typically uniformly distributed between train and test.  

<img src="https://github.com/bfetler/coronary_disease/blob/master/coronary_disease_plots/hist_coronary_train.png" alt="coronary training data histograms" />

<img src="https://github.com/bfetler/coronary_disease/blob/master/coronary_disease_plots/hist_coronary_test.png" alt="coronary test data histograms" />

#### Modeling and Fitting
As there are no significant anomalies in the data, we proceed to fit the training set using:
+ Logistic Regression
+ LinearSVC Support Vector Classification

After normalizing the data columns, we find the training data fits the presence or absence of coronary disease with 80% accuracy using either method.  A standard error of 10% was estimated by cross-validation.  

#### Evaluating Incoming Test Data
If we pretend that the test data comes in batches periodically in production, we can compare the variable distributions between train and test data to see if any anomalies stand out, to check if incoming data is statistically different from training data and may need attention.  This was done using an [Independent T-Test](http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.ttest_ind.html) comparing each column of variables in train and test data, with typical p-values given in the table below (all > 0.05, no significant difference).  

<table>
<tr>
<td><strong>variable</strong></td>
<td>age</td>
<td>sex</td>
<td>chest_pain</td>
<td>blood_pressure</td>
<td>cholesterol</td>
<td>blood_sugar_up</td>
<td>rest_ecg_type</td>
<td>max_heart_rate</td>
<td>exercise_angina</td>
<td>exer_depression</td>
<td>exer_peak_slope</td>
<td>fluor_count</td>
<td>thal_defect</td>
</tr>
<tr>
<td><strong>p-value</strong></td>
<td>0.843</td>
<td>0.406</td>
<td>0.311</td>
<td>0.807</td>
<td>0.258</td>
<td>0.713</td>
<td>0.175</td>
<td>0.131</td>
<td>0.076</td>
<td>0.697</td>
<td>0.332</td>
<td>0.414</td>
<td>0.807</td>
</table>

#### Prediction
Assuming we are satisfied there are no significant anomalies in the incoming test data, we proceed with test data prediction using:
+ Logistic Regression
+ LinearSVC Support Vector Classification

After normalizing, we find a prediction score of about 80% using either classifier.   

The fit and prediction scores vary by up to 15% depending on the random split between train and test data, and so are not highly reliable, but indicate a general feasibility of the method.  
