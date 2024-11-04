# Pedestrain-Data-Analytics---ML
Regression Task where we target to predict the attention of a pedestrian with their head and body orientations

The dataset was taken from the UCI Machine Learning Repository. The dataset contains a number of pedestrian
tracks recorded from a vehicle driving in a town in southern Germany. The data is particularly well-suited for
multi-agent motion prediction tasks. The raw data was acquired from a vehicle equipped with multiple sensors
while driving, for approximately five hours, in an urban area in southern Germany. The sensor set included one
mono-RGB camera, one stereo-RGB camera, an inertial measurement system with differential GPS and a lidar
system. The preprocessed data available from this repository consists of 45 pedestrian tracks (in world
coordinates) together with a semantic map of the static environment. For each track and at each time-step, not
only the agent position is provided, but also body and head orientation attributes, as well as the position of all
other agents and their type (e.g., car, cyclist, pedestrian etc.).
UCI Machine Learning Repository: Pedestrian in Traffic Dataset Data Set
Attribute Information:
Pedestrian tracks are stored in the tracks.csv. Each row in such files contains 14 comma-separated attributes,
with missing values denoted by ˜None”. The attributes are in order:
oid: unique agent id (int),
timestamp: time in seconds (float),
x: x component of position vector (float),
y: y component of position vector (float),
body_roll: roll body angle in degrees (float),
body_pitch: pitch body angle in degrees (float),
body_yaw: yaw body angle in degrees (float),
head_roll: roll head angle in degrees (float),
head_pitch: pitch head angle in degrees (float),
head_yaw: yaw head angle in degrees (float),
other_oid: list of ids of agents currently present in the scene ([list of int]),
other_class: list of other agents class labels ([list of int]),
other_x: list of other agents x coordinates ([list of float]),
other_y: list of other agents y coordinates ([list of float]).
Labels used to identify agent types are available in agent_class_label_info.csv.
The file semantic_map.png contains a map of the static environment, where semantic labels are color-encoded
according to the mapping available in semantic_map_label_info.csv. Information needed to transform between
image and world coordinates is stored in the file map2world_info.txt.

DATA PREPROCESSING:
Data Pre-Processing is done since Incomplete data leads to unusable Dataset. Inconsistent and non-standardized
data with outliers exist and need to be normalized and validated to prevent poor prediction of target value.
1.Data Cleaning:
• Detecting and removing outliers in the Dataset using Inter-Quartile Range
• All the missing values in the Dataset is replaced by the mean of the data in the respective column.
• Rounding off data to maintain consistency throughout all the columns in the Dataset.
RESULTS OF DIFFERENT MODELS:
After preprocessing the dataset, we have used below models to fit the data into:
• Linear Regression
• K-Nearest Neighbors
• Random Forest
• Ridge and Lasso
• Decision Trees
• SVM with nonlinear Kernel
• Gradient Boosting Regressor
• SVM Linear Kernel
Splitting data into train, validation, test sets:
1. Linear Regression:
Before fitting the data into Linear Regression model, we have scaled the data as this is a gradient descent-based
algorithm to put the feature values into the same range. To avoid overfitting used regularization methods Ridge
Regression and Lasso Regression. After validating the model on test dataset, to find best parameters used
Gridsearch CV and Randomized CV techniques.
Below are the test results:
Ridge and Lasso Regression

2. K-Nearest Neighbors:
A k-nearest-neighbor algorithm, often abbreviated k-nn, is an approach to data classification that estimates how
likely a data point is to be a member of one group or the other depending on what group the data points nearest
to it are in. Initially data has been scaled to fitting to the model and evaluated model on validation data by finding
RMSE for different values of K. We have used Grid search CV to tune the hyperparameter ‘K’.
Results:
Heatmap:
Error Plots:
Before and After Grid Search CV :
3. Random Forest:

Random forests or random decision forests is an ensemble learning method for classification, regression and
other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the
output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction
of the individual trees is returned. Random decision forests correct for decision trees' habit of overfitting to
their training set. Random forests generally outperform decision trees, but their accuracy is lower than gradient
boosted trees. However, data characteristics can affect their performance.
Results:
Using Random Search:
Using Grid Search:
4. Decision Trees:
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences,
including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only
contains conditional control statements.Model has been built by fitting train data to function
DecisionTreeRegressor() and computed mean square error for different values of depth of the tree. We have used
GridSearchCV to select best parameter(maximum depth of the tree). Using best parameters decision tree has been
built and calculated MSE for train data to evaluate the performance of model.
5. SVM with linear and nonlinear Kernel:

In SVM nonlinear kernel used pipeline module to build support vector regressor model with default
parameters. Then computed R^2 coefficient for train and test data to see how predictions vary from variance
and errors on validation data. We have computed cross validation score using CV=5. To perform
hyperparameter tuning used RandomizedSearchCV with scaled data for 3-fold and these are the optimal
values we have for the model. Linear Kernel is used when the data is Linearly separable, that is, it can be
separated using a single Line. It is one of the most common kernels to be used. It is mostly used when there
are a Large number of Features in a particular Data Set.
Results:
RMSE with RBF kernel with hyperparameter tuning is 0.006
COMPARING RESULTS OF ALL MODELS:
• To measure how well the model fits the data we have used Mean Squared Error as loss functions.
• To see how well the model predicts an outcome used coefficient of determination R^2.
EDS 6340 INTRODUCTION TO DATA SCIENCE
GROUP 18 – Pedestrian in Traffic Dataset
• Accuracy is generally used in classification and here in regression used to measure how far off the prediction
value will be from actual value in terms of percentage.
FIRST VARIABLE SELECTION:
These models perform by learning from the data we provide so variable selection is considered as crucial to ensure
that data includes most significant relevant information. This can reduce model complexity and enhance model
efficiency. We chose Correlations and Lasso techniques to perform best variables.
Bi-Directional:
Bidirectional elimination: which is essentially a forward selection procedure but with the possibility of
deleting a selected variable at each stage, as in the backward elimination, when there are correlations between
variables.
Using the best model from Project -2 (Random Forest Regressor) with the optimized features:
Best features : [‘body_yaw’ , ’body_pitch’ , ’head_roll’ , ’head_pitch’]
MSE for Random Forest using Bidirectional variable selection: 0.0058
Using the best model from Project -3 (SVM Non-Linear) with the optimized features
Best features : [‘body_yaw’ , ’body_roll’ , ’head_roll’ , ’head_pitch’]
MSE for SVM using Bidirectional variable selection: 0.0058
Lasso:
To get the best variables Pipeline has been built with scaled data and LassoRegressor module with 0.5. To find
the best parameters used GridSearchCV and the best parameter we have is model_alpha as 0.5. As coefficients
EDS 6340 INTRODUCTION TO DATA SCIENCE
GROUP 18 – Pedestrian in Traffic Dataset
with less importance shrink to 0, we have 4 coefficients left with 1 which are considered as best parameters.
Below are the best 4 variables we have:
Lasso Model α = 0.5
Using the best model from Project - 2 (Random Forest Regressor) with the optimized features and Grid
Search:
Best features : [‘body_yaw’ , ’body_pitch’ , ’head_roll’ , ’head_pitch’]
Model Performance
MSE on the validation data : 0.0057
Using the best model from Project - 3 (SVM Non-Linear) with the optimized features:
Best features : [‘body_yaw’ , ’body_pitch’ , ’head_roll’ , ’head_pitch’]
MSE on the validation data : 0.013
Clustering:
To perform clustering on the data initially we have scaled the dataset. As we have high number of features to see
the behavior among the variables checked for dependency with respect to head_yaw. By taking K value as 4, data
has been fit to the KMeans model and we have 4 classifications as 0,1,2,3. Below is the representation of cluster
for selected variables.
By considering kmeans.inertia, we have elbow at 4 clusters which represents the best value of K will be 4. To see
the goodness of clustering technique silhouette has been computed.
silhouette_score : 0.6302
K-Means Score : 48.8225
VISUALIZATION USING DIMENSIONALITY REDUCTION:

PCA: The main idea of Principal Component Analysis (PCA) is to reduce the dimensionality of datasets consisting
of many strongly or weakly correlated variables while preserving the maximum amount of variability present in
the dataset. The same is done by transforming the variables into a new set of variables called principal components
(or simply PC). These variables are orthogonal and are arranged so that the variation present in the original
variables decreases as you move down. A self-organizing map or self-organizing feature map is an unsupervised
machine learning technique used to produce a low-dimensional representation of a higher dimensional data set
while preserving the topological structure of the data.
SOM is a non-linear clustering and mapping (or dimensionality reduction) techniques to map multidimensional
data onto lower-dimensional which allows people to reduce complex problems for easy interpretation.
What is Predictive Modeling: Predictive modeling is a probabilistic process that allows us to forecast
outcomes, on the basis of some predictors. These predictors are basically features that come into play when
deciding the final result, i.e. the outcome of the model.
Why is Dimensionality Reduction important in Machine Learning and Predictive Modeling?
An easy email classification issue, where we must determine whether or not the email is spam, can be used to
illustrate dimensionality reduction. This might encompass a wide range of characteristics, including if the email
employs a template, its content, whether it has a generic subject, etc. Some of these characteristics, nevertheless,
might overlap. In another case, because to the strong correlation between the two, a classification issue that
depends on both rainfall and humidity can be reduced to just one underlying characteristic.
Consequently, we can lower the number of characteristics in these issues. While a 2-D classification problem may
be reduced to a straightforward 2-dimensional space and a 1-D problem to a straightforward line, a 3-D
classification problem can be challenging to picture. This idea is illustrated in the image below, in which a 3-D
feature space is divided into two 2-D feature spaces. If the two feature spaces are later found to be connected, the
number of features can be further decreased.
ENSEMBLE MODELING:
Ensemble learning involves taking the opinions of multiple experts and clubbing them together to get better
accuracy in the output than the accuracy of a single model alone. The single models we have trained as part of
this project was kernel SVM, Random Forest, Decision Tree Regressor and KNN which shows prominent results
on testing data was with MSE 0.0057 respectively. The voting regressor an ensemble model with 20 regressor
results in MSE with 0.0047 which was not highly significant from the single models.

CONCLUSIONS:
We have been effective in using several regression algorithms to comprehend the nature of a pedestrian traffic
movement and to recommend the necessary course of action. We can maximize the value of the data. We
discovered elements (factors) that are important for generating predictions. By analyzing this data, one can find
unexpected accidents and take the necessary precautions. We were able to recommend the proper course of action
based on the pedestrians' body movement data. Making predictions enabled us to locate the important feature. We
compared the output of various regression techniques and chose the model that best suited our needs. pictured
and comprehended the importance of the features in the regression model.
On comparing all the MSE with the models got the best results for Random Forest Model with LASSO feature
technique which is 0.0056. Now performing the predictions on the test data.
