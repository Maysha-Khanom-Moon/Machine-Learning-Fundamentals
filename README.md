1. Part - 01: Data Preprocessing

<br>

2. part - 02: Regression
    - predicting a real value

    1. Simple Linear Regression
    2. Multiple Linear Regression
    3. Polynomial Regression
    4. Support Vector for Regression (SVR)
    5. Decision Tree Regression
    6. Random Forest Regression

<br>

3. part - 03: Classification
    - predict a category
    - feature scaling is typically recommended for most classification models

    1. Logistic Regression
    2. K-Nearest Neighbors (K-NN)
    3. Support Vector Machine (SVM)
    4. Kernel SVM
    5. Naive Bayes
    6. Decision Tree Classification
    7. Random Forest Classification

<br>

4. part - 04: Clustering
    - Clustering groups similar data points without labels (unsupervised)
    - discovering patterns and relationships in datasets
    - Feature scaling is usually needed for distance-based algorithms

    1. K-Means Clustering
    2. Heirarchical Clustering

<br>

5. part - 05: ARL (Association Rule Learning)
    - finds things that often happen together
    - discovering secrets about how things are connected!

    1. Apriori --> more preferable
    2. Eclat

<pre>

You have a candy shop and want to know which candies people buy together. Here's what you found:

1. Alice: chocolate and gummies
2. Bob: chocolate and cookies
3. Charlie: gummies and cookies
4. David: chocolate, gummies, and cookies
5. Emma: gummies

You ask: "If someone buys chocolate, will they also buy gummies?"


<b>Steps</b>
----------
Support: how frequently an itemset appears
    - Chocolate and gummies were bought together 2 out of 5 times (40%).

Confidence: likelihood that gummies is purchased when chocolate is purchased
    - 2 out of 3 chocolate buyers also bought gummies (67%).

Lift: more likely someone is to buy gummies when they buy chocolate
    - Lift = Confidence ÷ Gummies alone
           = 0.67 ÷ 0.8 = 0.84
    - Confidence: 67% of chocolate buyers also buy gummies.
    - Gummies alone: 80% of people bought gummies.


<b>Conclusion:</b>
Since the lift is less than 1 (0.84), it means buying chocolate doesn't make it more likely to buy gummies because gummies are already popular on their own. If the lift were greater than 1, it would mean chocolate increases the chance of buying gummies.

</pre>

<br>
<br>


6. part - 06: RL (Reinforcement Learning)
    - an agent learns by interacting with an environment to achieve a goal. 
    - The agent takes actions, receives feedback as rewards or penalties, and updates its strategy to maximize long-term rewards.
    - game playing, robotics, recommendation systems, and autonomous vehicles.

    1. UCB (Upper Confidence Bound)
    2. Thompson Sampling


<br>
<br>


7. part - 07: NLP (Natural Language Procession)
    - to process and understand text and human language
    - most NLP algorithms are classification models
    - Bag of Words model --> a common preprocessing technique in NLP

    1. Artificial Neural Networks
    2. Convolution Neural Network

```
ANN stands for "Artificial Neural Network", 
which is a broad category of machine learning models inspired by the human brain, 
while CNN stands for "Convolutional Neural Network," a specific type of ANN 
particularly well-suited for image recognition tasks due to its "convolution" operation 
that effectively extracts features from images
```

<br>
<br>


8. part - 08: Deep Learning
    - Geoffrey Hinton --> god father of deep learning
    <br>

    - uses multi-layered neural networks to automatically learn patterns from large datasets
    -  improve by adjusting internal parameters through training, without manual feature engineering.

    - examples:
        - image recognition
        - speech recognition
        - natural language processing

```
ReLU Activation: ReLU (Rectified Linear Unit) in CNNs introduces non-linearity, 
allowing the network to learn complex patterns in data. It is computationally 
efficient (max(0, x)), helps avoid the vanishing gradient problem, 
and creates sparse activations by outputting zero for negative values. 
This sparsity improves efficiency and reduces overfitting. While it accelerates training convergence, 
it can suffer from the <b>dying ReLU problem</b>, where neurons stop learning (gradient becomes zero). 
Variants like Leaky ReLU and PReLU address this limitation. 
ReLU is widely used due to its simplicity and effectiveness in deep learning.

    - The main limitation of ReLU is the dying ReLU problem, 
    where some neurons output 0 for all inputs and stop learning because 
    their gradients become zero. This reduces the network's capacity to learn. 
    Solutions include using variants like Leaky ReLU (small gradient for negative inputs),
    PReLU (learnable slope for negatives), or ELU (smooth transitions for negatives), 
    and applying proper weight initialization methods.

```


<br>
<br>


9. part - 09: Dimensionality Reduction
    - simplifies datasets by reducing the number of features while retaining essential information
    
There are 2 main approaches:
1. feature selection: 
    - picks the most relevant features
    - based on variance or correlation
    - backward elimination, forward elimination, score comparison, etc
        
2. feature extraction: 
    - creates new features by transforming the original ones
    - pca, lda, kernel pca, qda, etc

<b>Benefits: </b> Faster computation, easier visualization, and improved model performance <br>
<b>Challenges: </b> Potenial information loss and reduced interpretability


<br>
<br>


10. part - 10: Model Selection & Boosting

    1. k-Fold Cross Validation --> model evaluation and validation technique
    2. Grid Search --> hyperparameter optimization technique

- XGBoost: one of the most powerful and popular machine learning model


<br>


### Hyperparameters vs Parameters
- Hyperparameters: set manually before training (e.g., learning rate, number of layers)

- Parameters: Learned from the data during training (e.g., weights and biases in neural network)

<br>
<br>


### n_jobs = N
In scikit-learn it used to specify the number of CPU cores to be used for parallel computation during certain operations

- N = -1 --> Uses all available CPU cores for parallel processing.
- N = 1 --> Uses a single CPU core (no parallelization).
- N > 1 --> Uses N CPU cores for parallel processing, where N is a positive integer.


<br>
<br>


### Analogy:
1. Regression --> Prediction 
2. Classification --> Detection
3. Clustering --> Grouping / Pattern 
4. ARL --> Correlation / Dependency
5. RL --> Optimization

<br>
<br>

### Machine Learning process
1. Data pre-processing
    - Import the data
    - Clean the data
    - Split into training & test sets
    - Feature scaling
        <br>

2. Modelling --> Class
    - Build the model
    - Train the model
    - Make predictions
        <br>

3. Evaluation
    - Calculate performance metrics
    - Make a verdict
<br>

<br>

### 5 assumption of Linear Regression
- 📈 Linearity
- 🔵 independence 
- 📊 homoscedasticity
- 🔔 normality
- 🚫 no multicollinearity
```
Here’s an explanation of the 5 assumptions of Linear Regression:

Linearity (📈):
What it means: The relationship between the independent variables (predictors) and the dependent variable (outcome) should be linear. This means that changes in the predictors should result in proportional changes in the outcome.
Why it’s important: If the relationship is not linear, the model will not accurately capture the relationship between the variables, leading to poor predictions.
How to check: Plot a scatterplot between the independent variables and the dependent variable. If the plot forms a straight line, linearity is satisfied. You can also check residual plots to ensure the errors are randomly scattered without patterns.

Independence (🔵):
What it means: The observations in the dataset should be independent of each other. This means that one observation should not influence another.
Why it’s important: Violating this assumption (e.g., if there is a time-based trend or spatial autocorrelation) can lead to underestimated standard errors and inflated significance of predictors.
How to check: If you're dealing with time series or spatial data, plot the residuals to look for patterns over time or space. Durbin-Watson test can also be used to detect autocorrelation.

Homoscedasticity (📊):
What it means: The variance of the errors (residuals) should be constant across all levels of the independent variables. In other words, the spread of the residuals should be roughly the same for all predicted values.
Why it’s important: If the errors have increasing or decreasing variance (heteroscedasticity), the model’s predictions may be biased, and statistical tests might not be reliable.
How to check: Plot the residuals against the predicted values. If the spread of residuals is consistent and does not form patterns (like a funnel shape), then homoscedasticity holds.

Normality (🔔):
What it means: The residuals (differences between observed and predicted values) should be normally distributed. This assumption is more critical for hypothesis testing and confidence intervals rather than for the accuracy of predictions.
Why it’s important: Non-normal residuals can affect the validity of significance tests (t-tests, F-tests) used in linear regression.
How to check: Plot a histogram or Q-Q plot of the residuals to see if they roughly follow a normal distribution. You can also use statistical tests like the Shapiro-Wilk test.

No Multicollinearity (🚫):
What it means: The independent variables should not be highly correlated with each other. Multicollinearity occurs when two or more predictors are strongly correlated, which makes it difficult to isolate the individual effect of each variable on the dependent variable.
Why it’s important: Multicollinearity can lead to large standard errors for regression coefficients, making it difficult to determine the effect of individual variables and weakening the statistical power of the model.
How to check: Calculate the Variance Inflation Factor (VIF). A VIF greater than 5 (or sometimes 10) is typically a sign of multicollinearity. Correlation matrices can also help spot highly correlated predictors.

These assumptions help ensure that your linear regression model provides valid, reliable, and interpretable results. If any of these assumptions are violated, your model might produce inaccurate predictions or unreliable statistical inferences.
```
<br>
<br>

### Understanding the p-value
Statistical significance in the context of the p-value refers to the likelihood that the observed results from a sample are not due to random chance, but rather indicate a real effect or relationship in the population being studied.

Understanding the p-value:
P-value: It is the probability of obtaining the observed results, or something more extreme, assuming that the null hypothesis (H₀) is true. The null hypothesis generally states that there is no effect, relationship, or difference between groups.
Interpreting the p-value:
Low p-value (typically ≤ 0.05): The result is statistically significant, meaning there is strong evidence against the null hypothesis. You would reject the null hypothesis and conclude that there is a meaningful effect or relationship.

High p-value (> 0.05): The result is not statistically significant, meaning there is not enough evidence to reject the null hypothesis. This implies that any observed effect might simply be due to random chance.

Statistical significance in hypothesis testing:
Null hypothesis (H₀): Assumes that no effect or relationship exists (e.g., the coefficients in a regression are 0, meaning the independent variables have no impact on the dependent variable).
Alternative hypothesis (H₁): Assumes that an effect or relationship does exist.
Significance level (α): This is a threshold chosen before the test (commonly 0.05). If the p-value is below this threshold, the result is deemed statistically significant.
Example of interpretation:
p-value = 0.03: Since 0.03 is less than the common significance level of 0.05, we reject the null hypothesis. This suggests that the result is statistically significant and that the observed effect is likely not due to random chance.

p-value = 0.10: Since 0.10 is greater than 0.05, we do not reject the null hypothesis. The result is not statistically significant, meaning that the evidence is insufficient to claim that there is a real effect.

Key Points:
A statistically significant result (low p-value) means there is enough evidence to suggest a real effect or relationship.
It does not necessarily imply the result is practically significant or large, just that it is unlikely to have occurred by random chance.
<br>
<br>


### 5 methods of building models:
1. All-in 
2. Backward Elimination       |
3. Forward Selection           >  Stepwise Regression --> followed by p value
4. Bidirectional Elimination  |
5. Score Comparisonb 

- it can be done automatically by sklearn algorithm

<br>
<br>


### R square
- Goodness of fit --> greater is better

- a statistical metric used to measure how well a model fits the data. Specifically, it represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

- R^2 = 1 - (SS_res / SS_tot)
    - SS_res = the residual sum of squares (sum of squared differences between actual and predicted values)
    - SS_tot = the total sum of squares (sum of squared differences between actual values and the mean of actual values)


```
R^2 --> [0, 1]

1.0 --> perfect fit (suspicious)
~0.9 --> very good
<0.7 --> not great
<0.4 --> terrible
<0 --> model makes no sense for this data

Although it's depends on industry requirement
```
<br>

### Adjusted R Square
y = b0 + b1.X1 + b2.X2 <-- + b3.X3
```
* if we want to add a new feature then 
    - SS_res doesn't changes
    - SS_res will decrease or stay the same (b3 = 0)
        - because of Ordinary Least Square: SS_res --> min

Adj R^2 = 1 - (1 - R^2) * (n - 1) / (n - k - 1)
k = no of features
n = sample size
```

```
Adjusted R^2 is a version of R^2 that accounts for the number of predictors in a regression model. 
It adjusts for model complexity, penalizing the inclusion of irrelevant variables. 
Unlike regular R^2, which always increases as more predictors are added, 
adjusted R^2 only increases if a new variable improves the model significantly. 
This makes it useful for evaluating models by balancing explanatory power with simplicity.
```
<br>

### Regularization
A technique used in the model training process to prevent overfitting by adding a penalty term to the model's cost function, which discourages large coefficients. This helps keep the model simpler and improves its generalization to new data.
<br>
<br>


### How do I know which regression model to choose for a particular problem/ dataset?
- try all these models then select the best one
    - which has the most R square value
    - R^2 = (coefficient of determination) regression score function

- for missing and categorical data --> do preprocessing

- [code templates](https://drive.google.com/drive/folders/1O8vabaxga3ITjCWfwD79Xnyf8RavYuyk)

- [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)
<br>

### How do I know which classification model to choose for a particular problem/ dataset?
- try all these models then select the best one
    - which has the most accuracy_score

- for missing and categorical data --> do preprocessing


- [code templates](https://drive.google.com/drive/folders/1O8vabaxga3ITjCWfwD79Xnyf8RavYuyk)

- [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)
<br>
<br>


### Confusion Matrix & Accuracy
```
cm = [Actual negative
      Acutal positive]

cm =     0   1
     0 [[TF, FP]
     1 [FN, TP]]

TN = True Negative
FP = False Positive
FN = False Negative
TP = True Positive

It shows the percentage (rate)!
```

```
Accuracy Rate, AR = Correct / Total
                  = (TN + TP) / Total


Error Rate, ER = Incorrect / Total
               = (FP + FN) / Total
```
<br>

### Accuracy Paradox
The phenomenon where a model achieves high accuracy but performs poorly, particularly on important or minority classes, due to class imbalance in the dataset.

- In an imbalanced dataset, one class (majority) dominates the other(s).
- A model that predicts only the majority class can achieve high accuracy without truly understanding or predicting the minority class.

```
Example
-------
In a dataset with 95% negatives and 5% positives,
predicting only negatives gives 95% accuracy 
but ignores the minority class entirely.
```
<br>

- Solution: Use metrics like Precision, Recall, F1-Score, or ROC-AUC for better evaluation in imbalanced datasets.
<pre>
1. Precision: How many predicted positives are actual positives. 
    - precision = TP / (TP + FP)
<br>
2. Recall: How many actual positives are correctly predicted. 
    - recall = TP / (TP + FN)
<br>
3. F1-Score: Balances precision and recall. 
    - F1-score = 2 * (precision * recall) / (precision + recall)
<br>
4. ROC-AUC: Considers true positive and false positive rates.
    - ROC-AUC: Receiver Operating Characteristic - Area Under Curve
    - plots TPR vs FPR aka ROC curve shows the trade-off between TPR and FPR
        - TPR (true positive rate) = TP / (TP + FN) --> recall
        - FPR (false positive rate) = FP / (FP + TN)
</pre>
<br>
<br>

### CAP
The Cumulative Accuracy Profile (CAP) evaluates binary classifiers by showing how well the model captures positives (e.g., fraud cases) within a sorted population.

Key Features:
- Axes: X-axis = proportion of the population, Y-axis = proportion of positives captured.

- Curves: 
    - Random Line: Captures positives proportionally.

    - Perfect Model: Captures all positives immediately.

    - Model Curve: Shows the model's actual performance.

- <b>Goal: A steeper curve closer to the perfect model indicates better performance.</b>

<br>

### Types of Decision Tree algorithms
1. CART (Classification And Regression Tree):
    - produces binary trees (two branches per node)
    - uses the <b>Gini index</b> (for classification) or <b>mean squared error</b> (for regression) to split data

2. ID3: 
    - uses entropy and information gain for splits

3. C4.5:
    - An improvement over ID3, handling both <b>numeric and categorical data</b>, supporting <b>pruning</b>, and using <b>gain ratio</b> for splits


<br>