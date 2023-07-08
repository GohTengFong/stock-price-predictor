# stock-price-predictor

A very simple Python stock price predictor implemented using numpy to perform calculations on the data, scikit-learn to build a model and matplotlib to plot my data

## Technology Used
Python

## What I Have Learnt
### Technical
I learnt the basic outline and structure of how to train a predictive model.
1. Attain data
   1.1 Preprocess the data if necessary
3. Choose dependencies that would best suit the job
4. Train the model using the data
5. Evaluate the model
   5.1 Perform cross-validation or other techniques to obtain a more robust estimation of the model's performance
   5.2 Compare it with historial data

### Non-technical
More than technical skills, I learnt more about how companies use models in predicting stock prices, more specifically, SVR. SVR maps the input data into a high-dimensional feature space using a technique called the kernel trick. This mapping allows SVR to find a non-linear decision boundary in the original input space by transforming it into a higher-dimensional space where a linear decision boundary can be identified.



Kernal Function : calculates the similarity between pairs of data points in the feature space. Common kernel functions include linear, polynomial, Gaussian (RBF), and sigmoid functions. The choice of kernel function affects the shape of the decision boundary and the model's ability to capture complex patterns in the data. (In plain words, when using a polynomial kernel against a set of data with 2 inputs, we are trying to find a polynomial equation that would best fit this set of data)
