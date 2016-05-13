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
<td>0.21</td>
</tr>
<tr>
<td>sex</td>
<td>0.43</td>
</tr>
<tr>
<td>chest_pain</td>
<td>0.23</td>
</tr>
<tr>
<td>b_pressure</td>
<td>0.42</td>
</tr>
<tr>
<td>cholesterol</td>
<td>0.82</td>
</tr>
<tr>
<td>b_sugar_up</td>
<td>0.47</td>
</tr>
<tr>
<td>ecg_type</td>
<td>0.11</td>
</tr>
<tr>
<td>heart_rate</td>
<td>0.56</td>
</tr>
<tr>
<td>exer_angina</td>
<td>0.92</td>
</tr>
<tr>
<td>exer_depress</td>
<td>0.68</td>
</tr>
<tr>
<td>exer_slope</td>
<td>0.65</td>
</tr>
<tr>
<td>fluor_count</td>
<td>0.80</td>
</tr>
<tr>
<td>thal_defect</td>
<td>0.37</td>
</tr>
</table>

##### A Note on Unit Testing
Unit testing of data science methods may not be very useful, unless one is writing a new algorithm.  One may test if an algorithm returns a result, or good results, or implements a particular API.  Most of the Scikit-learn methods are already tested for this, provided one follows suggestions for sensible data as described in the docs, such as scaling the data beforehand if needed.  

__*However!*__ One very useful task in data science is to test not the routines, but the data.  These types of tests are represented above.  For example, is the distribution of parameters in the training and test sets the same, statistically speaking?  Are they the same for incoming data in a new data stream?  If not, one may flag the data in production.  For example, for each data column one may write:

    assertGreater(ttest.pvalue, 0.05)

This of course depends on what your [Null Hypothesis](https://en.wikipedia.org/wiki/Null_hypothesis) is, your assumptions about the data and models, and what issues you are attempting to solve with data science.    

#### Modeling and Fitting
If there are no significant anomalies in the data, we proceed to fit the training set using:
+ Logistic Regression
+ LinearSVC Support Vector Classification

After scaling the data columns, we find the training data fits the presence or absence of coronary disease with 82% accuracy using either method.  A standard error of 7% was estimated from cross-validation scores (cv = 5), which varies by 1-2% due to the small number of CV folds and random variation in the train test split.

[Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) of normalized data gives an idea of variable importance.  The coefficients and their absolute value order depends on the random split between train and test datasets, but in general *fluoroscopy vessel count (fluor_count)* is always at top, *chest_pain type* is in the top five, and *sex* has more influence than *age*.  Typical values are given below.

<table>
<tr>
<td><strong>variable</strong></td>
<td><strong>logistic regression coefficient</strong></td>
</tr>
<tr>
<td>fluor_count</td>
<td>1.17</td>
</tr>
<tr>
<td>exer_depress</td>
<td>0.84</td>
</tr>
<tr>
<td>thal_defect</td>
<td>0.67</td>
</tr>
<tr>
<td>chest_pain</td>
<td>0.51</td>
</tr>
<tr>
<td>b_pressure</td>
<td>0.48</td>
</tr>
<tr>
<td>sex</td>
<td>0.43</td>
</tr>
<tr>
<td>ecg_type</td>
<td>0.40</td>
</tr>
<tr>
<td>b_sugar_up</td>
<td>-0.38</td>
</tr>
<tr>
<td>cholesterol</td>
<td>0.23</td>
</tr>
<tr>
<td>exer_angina</td>
<td>0.21</td>
</tr>
<tr>
<td>heart_rate</td>
<td>-0.21</td>
</tr>
<tr>
<td>age</td>
<td>-0.04</td>
</tr>
<tr>
<td>intercept</td>
<td>0.03</td>
</tr>
<tr>
<td>exer_slope</td>
<td>0.003</td>
</tr>
</table>

#### Prediction
Assuming we are satisfied there are no significant anomalies in the incoming test data, and that the training data is not overfit and is reasonable, we proceed with test data prediction using:
+ Logistic Regression
+ LinearSVC Support Vector Classification

After normalizing test data, we find a prediction score of about 80% using either classifier.   

The fit and prediction scores vary by up to 7-8% depending on the random split between train and test data, and so are not highly reliable, but indicate a general feasibility of the method.  
