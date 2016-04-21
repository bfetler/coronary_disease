## Coronary Heart Disease
How well can we predict coronary heart disease from patient data?  An early dataset is given in the [UCI Machine Learning Heart Disease Dataset](http://archive.ics.uci.edu/ml/datasets/Heart+Disease).  

Data was collected from 303 patients at the Cleveland Clinic, both without and with varying degrees of coronary heart disease.  Since there are so few data points, patients with heart disease were grouped together for a binary target variable.  Some of the seventy-five columns of original data were corrupted, and replaced with fourteen columns by the data author.  After data cleaning, 297 patients were left.  Despite the small size, it is a reasonable dataset to start exploring coronary disease prediction.  

#### Exploration
Data exploration and prediction is given in __coronary_predict.py__, and script output in __coronary_output.txt__ with plots in __coronary_plots/__.  The data was randomly split into 70% training data and 30% test data.  A scatter matrix of training data shows some correlation between variables, but no strong trends are visible.  

<img src="" />

Histograms of train and test data typically show similar patterns, so that variable column values are typically uniformly distributed between train and test.  

<img src="https://github.com/bfetler/coronary_disease/blob/master/coronary_disease_plots/hist_coronary_train.png" alt="coronary training data histograms" />
<img src="https://github.com/bfetler/coronary_disease/blob/master/coronary_disease_plots/hist_coronary_test.png" alt="coronary test data histograms" />

If we pretend that the training data is much larger, and that the test data comes in batches periodically, we can compare the variable distributions between train and test data to see if any anomalies stand out, showing that the incoming data is statistically different from training data and may need attention.  This was done using a T-Test comparing each column of variables in train and test, with typical p-values (all > 0.05, not significantly different) given in the table below.  

<table>
<tr>
<td><strong>variable</strong></td>
<td>age</td>
<td>sex</td>
<td>cp</td>
<td>trestbps</td>
<td>chol</td>
<td>fbs</td>
<td>restecg</td>
</tr>
<tr>
<td><strong>p-value</strong></td>
<td>0.8432</td>
<td>0.4058</td>
<td>0.3108</td>
<td>0.8072</td>
<td>0.2580</td>
<td>0.7127</td>
<td>0.1750</td>
</tr>
</table>
<table>
<tr>
<td><strong>variable</strong></td>
<td>thalach</td>
<td>exang</td>
<td>oldpeak</td>
<td>slope</td>
<td>ca</td>
<td>thal</td>
</tr>
<tr>
<td><strong>p-value</strong></td>
<td>0.1305</td>
<td>0.0758</td>
<td>0.6974</td>
<td>0.3320</td>
<td>0.4137</td>
<td>0.8071</td>
</table>

#### Prediction
Assuming we are satisfied there are no significant anomalies in the incoming test data, we proceed with train data fitting and test data prediction using:
+ Logistic Regression
+ LinearSVC Support Vector Classification

After scaling the data (SVM is sensitive to scale), we find training data fit with 80% accuracy for both methods.  A standard error of 10% was estimated by cross-validation.  A test data prediction score of about 80% was found using both classifiers.   The numbers vary by 5% depending on the random split between train and test data, and so are not highly reliable, but indicate a general feasibility of the method.  
