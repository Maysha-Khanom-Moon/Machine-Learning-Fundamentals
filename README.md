1. Part - 01: Data Preprocessing

2. part - 02: Regression
    - predicting a real value

3. part - 03: 


#### Machine Learning process
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

#### 5 assumption of Linear Regression
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


#### Understanding the p-value
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


#### 5 methods of building models:
1. All-in 
2. Backward Elimination       |
3. Forward Selection           >  Stepwise Regression --> followed by p value
4. Bidirectional Elimination  |
5. Score Comparisonb  
<br>
