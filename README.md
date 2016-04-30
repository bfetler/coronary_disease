## Coronary Heart Disease
How well can we predict coronary heart disease from patient data?  A dataset from 1988 is given in the [UCI Machine Learning Heart Disease Dataset](http://archive.ics.uci.edu/ml/datasets/Heart+Disease).  

Data was collected from 303 patients at the Cleveland Clinic, both without and with varying degrees of coronary heart disease.  Since there are so few data points, patients with heart disease were grouped together for a binary target variable.  Some of the seventy-five columns of original data were corrupted, and replaced with fourteen columns by the data author.  After data cleaning, 297 patients were left.  Despite the small size, it is a reasonable dataset to start exploring coronary disease prediction.  

#### Exploration
Data exploration and prediction is given in __coronary_predict.py__.  The data was randomly split into 70% training data and 30% test data.  A scatter matrix of training data shows some correlation between variables, for example between *maximum heart rate* and *fluoroscopy vessel count*, but no strong trends are apparent.

<img src="https://github.com/bfetler/coronary_disease/blob/master/coronary_disease_plots/scatter_matrix.png" alt="scatter matrix" />

Histograms of train and test data typically show similar patterns, so that variable column values are typically uniformly distributed between train and test.  

<img src="https://github.com/bfetler/coronary_disease/blob/master/coronary_disease_plots/hist_coronary_train.png" alt="coronary training data histograms" />

<img src="https://github.com/bfetler/coronary_disease/blob/master/coronary_disease_plots/hist_coronary_test.png" alt="coronary test data histograms" />

#### Evaluating Incoming Test Data
If the test data comes in batches periodically in production, we could compare the variable distributions between train and test data to see if any anomalies stand out, to check if incoming data is statistically different from training data and needs attention.  We may use the test data to model this process, using an [Independent T-Test](http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.ttest_ind.html) comparing each column of variables in train and test data.  Typical p-values are given in the table below, all > 0.05 (no significant difference).  

<table>
<tr>
<td><strong>variable</strong></td>
<td><strong>p-value of train, test</strong></td>
</tr>
<tr>
<td>age</td>
<td>0.90</td>
</tr>
<tr>
<td>sex</td>
<td>0.61</td>
</tr>
<tr>
<td>chest_pain</td>
<td>0.42</td>
</tr>
<tr>
<td>b_pressure</td>
<td>0.68</td>
</tr>
<tr>
<td>cholesterol</td>
<td>0.21</td>
</tr>
<tr>
<td>b_sugar_up</td>
<td>0.99</td>
</tr>
<tr>
<td>ecg_type</td>
<td>0.77</td>
</tr>
<tr>
<td>heart_rate</td>
<td>0.78</td>
</tr>
<tr>
<td>exer_angina</td>
<td>0.67</td>
</tr>
<tr>
<td>exer_depression</td>
<td>0.08</td>
</tr>
<tr>
<td>exer_slope</td>
<td>0.07</td>
</tr>
<tr>
<td>fluor_count</td>
<td>0.90</td>
</tr>
<tr>
<td>thal_defect</td>
<td>0.31</td>
</tr>
</table>

#### Modeling and Fitting
If there are no significant anomalies in the data, we proceed to fit the training set using:
+ Logistic Regression
+ LinearSVC Support Vector Classification

After scaling the data columns, we find the training data fits the presence or absence of coronary disease with 83% accuracy using either method.  A standard error of 4% was estimated from cross-validation scores (cv = 5).  Repeated fits sometimes gave larger error estimates (up to 8%), due to random variation in the train test split.

[Logistic regression from Statsmodels](http://statsmodels.sourceforge.net/0.6.0/generated/statsmodels.discrete.discrete_model.Logit.html) of normalized data gives an idea of variable importance.  The coefficients and their absolute value order depends on the random split between train and test datasets, but in general *fluoroscopy vessel count (fluor_count)* is always at top, *chest_pain type* is in the top five, and *sex* has more influence than *age*.  Typical values are given below.

<table>
<tr>
<td><strong>variable</strong></td>
<td><strong>logistic regression coefficient</strong></td>
</tr>
<tr>
<td>fluor_count</td>
<td>1.14</td>
</tr>
<tr>
<td>thal_defect</td>
<td>0.64</td>
</tr>
<tr>
<td>exer_depress</td>
<td>0.46</td>
</tr>
<tr>
<td>sex</td>
<td>0.46</td>
</tr>
<tr>
<td>chest_pain</td>
<td>0.41</td>
</tr>
<tr>
<td>b_pressure</td>
<td>0.37</td>
</tr>
<tr>
<td>exer_angina</td>
<td>0.37</td>
</tr>
<tr>
<td>b_sugar_up</td>
<td>-0.34</td>
</tr>
<tr>
<td>heart_rate</td>
<td>-0.32</td>
</tr>
<tr>
<td>age</td>
<td>-0.31</td>
</tr>
<tr>
<td>cholesterol</td>
<td>0.23</td>
</tr>
<tr>
<td>exer_slope</td>
<td>0.22</td>
</tr>
<tr>
<td>ecg_type</td>
<td>0.21</td>
</tr>
<tr>
<td>constant</td>
<td>-0.20</td>
</tr>
</table>

#### Prediction
Assuming we are satisfied there are no significant anomalies in the incoming test data, and that the training data is not overfit and is reasonable, we proceed with test data prediction using:
+ Logistic Regression
+ LinearSVC Support Vector Classification

After normalizing test data, we find a prediction score of about 80% using either classifier.   

The fit and prediction scores vary by up to 10% depending on the random split between train and test data, and so are not highly reliable, but indicate a general feasibility of the method.  
