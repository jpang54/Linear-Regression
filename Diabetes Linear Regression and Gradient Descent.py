"""

We will implement linear regression and gradient descent and visualize the results. 

## Getting Started with Linear Regression

We'll first do a 1-dimensional problem where we're learning a linear relationship between Y and X. 
This has a simple analytic solution. Then we use the scikit-learn package to do the n-dimensional regression. 
Finally, we use visualization methods to visualize the results as well as the iterations of gradient descent. 


The two regression problems use the [*diabetes* dataset (see Table 1)](https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf). 
It contains real records of patients with diabetes. Do not worry as:

* These records have been completely anonymized and are 20 years old.
* Patients gave their consent

The dataset contains 442 records of diabetes patients with the following information and measurements per patient: 
age, sex, body mass index (BMI), average blood pressure, and six blood serum measurements. 
For each patient, the  *Diabetes Expression Level*  attribute contains a quantitative measure of disease progression one year later. 
Physicians assigned a number between 25 to 346 to each patient indicating severity of the disease (higher = more severe). 
 
A predictor for the patient's status a year later would be very useful for designing and executing preventative care for the coming year and possibly reversing the course of the disease. 
An accurate predictor would probably have a strong effect on preventive behavior as well: faced with a choice between death in two years and 
eating half a pound of kale every day, many patients may agree to the kale. 

We will construct linear predictors, calculate their error on train & test data, and visualize the prediction of our predictors.
"""

# Import files
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

"""When encountering a new dataset, a little *exploratory data analysis* is often useful.  
A simple first step is to see how well just **one** attribute predicts diabetes expression. 
Let's plot the BMI vs diabetes expression and see if it looks vaguely linear to us!
"""

age_index = 0
sex_index = 1
bmi_index = 2
plt.scatter(diabetes_X[:, bmi_index], diabetes_y)
plt.xlabel('BMI (Normalized)')
plt.ylabel('Diabetes Expression Level')
plt.title('Diabetes Expression vs BMI')

"""Looks pretty good -- we see some correlation. How about age? """

plt.scatter(diabetes_X[:, age_index], diabetes_y)
plt.xlabel('Age (Normalized)')
plt.ylabel('Diabetes Expression Level')
plt.title('Diabetes Expression vs Age')

"""Hmm, surprisingly, age isn't very correlated. Seems like diabetes hits people of all ages. 
This visual check whether attribute & target have a coarse linear relationship is very useful first step for researchers building regression models, 
and trying to identify relevant variables. Let us proceed with BMI as our sole attribute for now.

As responsible machine learners we would like to prevent overfitting. 
We will only fit the model using half of our data and holding out the other half for final testing of the model. 

"""

# Use only one attribute -- choose between age, sex, or BMI and to form a predictor of diabetes expression level.

X = diabetes_X[:,bmi_index]

# Use a half of the dataset for test
test_size = X.shape[0] // 2

# Split the data into train & test
x_train = X[:-test_size]
x_test = X[-test_size:]

# Split the targets into training/testing sets
y_train = diabetes_y[:-test_size]
y_test = diabetes_y[-test_size:]

"""### Find a line $\hat{y} = ax + b$ by minimizing $\sum_i (\hat{y}_i - y_i)^2$ 

Below when we say "find" we mean that we should use calculus to find the expression, and then write simple code to evaluate it on the dataset. 
In this simple setting, there's no need to use gradient descent. 

1. Assume first that $a=0$ to find $b$. 
2. Find $a$ such that $\sum_i (a x_i + b - y_i)^2$ is minimized w.r.t $a$. 
3. Calculate the average prediction error (MSE) $1/m * \sum_{i=1}^m (a x_i + b - y_i)^2$ 
on train and test data. Print the results. 

The mean squared error is a common measurement of error! Food for thought: Why do we square the error? 
"""

# Derivation of b: multiply out the square in the equation given in step 2,
# take the first derivative of this equation w.r.t b, set equal to 0, and find
# that b = y_average - a * x_average. Assuming a = 0, b = average of y values
b_train = sum(y_train) / test_size

# Derivation of a: substitute b with y_average in the equation given in step 2,
# multiply the square out, take the first derivative w.r.t a and set equal to 0, 
# and isolate a. This below is a helper function to calculate a.
def calculate_a(xcol, ycol, b):
    denominator = sum(xcol*xcol)
    numerator = sum(xcol*ycol) - b * sum(xcol)
    return numerator / denominator

a_train = calculate_a(x_train, y_train, b_train)

# Helper function to calculate average prediction error (mse)
def calculate_mse(xcol, ycol, a, b):
    pred_error = (a * xcol + b - ycol) ** 2
    return sum(pred_error) / test_size

mse_train = calculate_mse(x_train, y_train, a_train, b_train)
mse_test = calculate_mse(x_test, y_test, a_train, b_train)

print(mse_train)
print(mse_test)

"""Now that we have fit a model and computed its error, it is our job to communicate our work effectively to people who may not know how linear regression works.  
Visualizing the data and our model is often a crucial step for debugging and evaluating how well our algorithm works.

### Plot the results
1. Plot $\hat{y}$ vs $y$ for the training set (in blue) and for the testing set (in red). 
2. Plot the line $\hat{y} = y$ in black. This line indicates where all the points would fall if our predictor was exactly correct, so it provides a good visual reference.
"""

# Predicted y values for all x values
yhat = a_train * X + b_train # X = diabetes_X[:,bmi_index] from above

# Split predicted y values for training and test
yhat_train = yhat[:-test_size]
yhat_test = yhat[-test_size:]

# Predicted vs actual y values for training data
plt.scatter(y_train, yhat_train, color = 'blue', label = 'training')
 
# Predicted vs actual y values for test data
plt.scatter(y_test, yhat_test, color = 'red', label = 'testing')
 
# Line where all points would fall if predictor was exactly correct
plt.plot(yhat, yhat, color = 'black')
 
plt.xlabel('Diabetes Expression Level (Actual)')
plt.ylabel('Diabetes Expression Level Predictions')
plt.title('Linear Regression Model of Diabetes Expression Levels')
plt.legend()
plt.show()

"""Now that we have a sense for how good of a predictor BMI is for diabetes expression, 
we can try to use all of the features available to us to build a better model and see how much better we do at prediction using all features.

To keep things simple, we will use scikit's off-the-shelf solution. 

### Run regression on all dimensions of the data 
Instead of looking at just one feature from the input, use all 10 of them:

1. Use the [LinearRegression object from scikit.linear_model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) to fit the data.
2. Train the model on *all* features from diabetes_X. Use half of the dataset for training and half for testing.
3. Calculate the MSE of the fit on the train and test data. Print the results.
4. Create the same plot as we did above.
"""

# Split the data into train & test
x_train_all = diabetes_X[:-test_size]
x_test_all = diabetes_X[-test_size:]

# Split the targets into training/testing sets
y_train_all = diabetes_y[:-test_size]
y_test_all = diabetes_y[-test_size:]

# Train the model on all features from the training set
reg = linear_model.LinearRegression().fit(x_train_all, y_train_all)

# Calculate predictions for y values 
y_all_predict = reg.predict(diabetes_X)
y_train_predict = y_all_predict[:-test_size]
y_test_predict = y_all_predict[-test_size:]

# Helper function to calculate mse on training and testing data
def calculate_mse_all(ycol, ypred):
    pred_error = (ypred - ycol) ** 2
    return sum(pred_error) / test_size

mse_train_all = calculate_mse_all(y_train_all, y_train_predict)
mse_test_all = calculate_mse_all(y_test_all, y_test_predict)
print(mse_train_all)
print(mse_test_all)

# Predicted vs actual y values for training data
plt.scatter(y_train_all, y_train_predict, color = 'blue', label = 'training')
 
# Predicted vs actual y values for test data
plt.scatter(y_test_all, y_test_predict, color = 'red', label = 'testing')
 
# Line where all points would fall if predictor was exactly correct
plt.plot(y_all_predict, y_all_predict, color = 'black')
 
plt.xlabel('Diabetes Expression Level (Actual)')
plt.ylabel('Diabetes Expression Level Predictions')
plt.title('Linear Regression Model of Diabetes Expression Levels')
plt.legend()
plt.show()

"""## Gradient Descent 

Next, we are going to use gradient descent on a function of two variables, and visualize it.
 
In order to run machine learning algorithms efficiently we will be using the numpy library. 
We are first going to slightly re-define the funky function (this is to allow us to drive home the important property that initialization matters). 
"""

import numpy as np

# returns funkier function over a reasonable range of X, Y values to plot
def funkier(delta=0.01):
    delta = 0.01
    x = np.arange(-3.3, 3.3, delta)
    y = np.arange(-2.8, 2.8, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2) # centered at (0,0)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2) # centered at (1,1)
    Z3 = np.exp(-(X + 1)**2 - (Y + 1)**2) # centered at (-1,-1)
    Z = Z1 - Z2 - 0.7*Z3
    return X, Y, Z

# given X and Y, returns Z
# X and Y can be arrays or single values...numpy will handle it!
def funkier_z(X, Y):
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z3 = np.exp(-(X + 1)**2 - (Y + 1)**2)
    Z = Z1 - Z2 - 0.7*Z3
    return Z

X, Y, Z = funkier()

# Setting the figure size and 3D projection
ax = plt.figure(figsize=(12,10)).gca(projection='3d')

# Creating labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
_ = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

"""This function has a clear maximum and at least two minima, so it's a great place for us to practice gradient descent! 
Take a guess at how the trajectory should go if we start at the red peak, and keep that visualization in mind as we go forward. 
It also appears that one of the minima is smaller than the other (i.e., there is a global and local minimum). 
We will explore the conditions under which we might end up at one over the other.

First, we have to implement the gradient function. For a particular $x$ and $y$, we will need to compute $dfunkier/dx$ and $dfunkier/dy$. 
`funkier_grad` should return the gradient in tuple form (i.e., should return 2 values). 
"""

def funkier_grad(x,y):
  
    # The partial derivative of z with respect to x
    dx = (-2 * x * np.exp(-x * x - y * y) 
          + 2 * (x - 1) * np.exp(-(x - 1)**2 - (y - 1)**2)
          + 1.4 * (x + 1) * np.exp(-(x + 1)**2 - (y + 1)**2) )
    
    # The partial derivative of z with respect to y
    dy = (-2 * y * np.exp(-x * x - y * y) 
          + 2 * (y - 1) * np.exp(-(x - 1)**2 - (y - 1)**2)
          + 1.4 * (y + 1) * np.exp(-(x + 1)**2 - (y + 1)**2) )
    
    return dx, dy

"""We next implement gradient descent in order to find the minimum of **funkier**. """

def funkier_minimize(x0, y0, eta):
    # Keep track of our position on the graph after each step in an array
    x = np.zeros(len(eta) + 1)
    y = np.zeros(len(eta) + 1)

    # Initialize our starting position
    x[0] = x0
    y[0] = y0

    print('\n Using starting point: ', x[0], y[0])

    # Keep stepping towards the minimum unless we run out of iterations
    for i in range(len(eta)):
        # Print our current position for every multiple of 5
        if i % 5 == 0:
            print('{0:2d}: x={1:6.3f} y={2:6.3f} z={3:6.3f}'.format(i, x[i], y[i], funkier_z(x[i], y[i])))

        # Calculate the next position and update our array
        next_position = funkier_grad(x[i], y[i])
        x[i+1] = x[i] - eta[i] * next_position[0]
        y[i+1] = y[i] - eta[i] * next_position[1]

        # Stop stepping if very small difference between next & current position
        if (abs(x[i+1] - x[i]) < 1e-6):
            return x[:i+2], y[:i+2]

        # Stop stepping if our next position becomes too high
        if abs(x[i+1]) > 100:
            print('Oh no, diverging?')
            return x[:i+2], y[:i+2]

    return x, y

"""We will now set for the maximum number of iterations and the step size."""

max_iter = 30
eta = 0.1 * np.ones(max_iter)

"""We provide a function to plot the trajectory of gradient descent. It will plot the trajectory on the 3D plot in the style we generated above."""

def plot_3D(xs, ys, zs):
    ax = plt.figure(figsize=(12,10)).gca(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    elev=ax.elev
    azim=ax.azim
    ax.view_init(elev= elev, azim = azim)
    _ = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.5)
    ax.plot(xs, ys, zs, color='orange', markerfacecolor='black', markeredgecolor='k', marker='o', markersize=5)

"""Finally, it is time to unleash gradient descent and look for the optimum of **funkier**. 
Examine the results, and provide a short explanation at the end. Note that we provide two different initializations. 
When we plot the trajectory gradient descent takes, we should see that even though both initializations are very close to each other, they find different minima!

Note that **funkier_minimize** only gives us the $(x,y)$ trajectory. To get the corresponding $z$ values along the trajectory, we will have to call **funkier_z**, defined previously. 
Here, we demonstrate another powerful feature of numpy. We can pass in single scalars to **funkier_z**, as we did in the **funkier_minimize** function, 
or we can pass in entire arrays. If we give the function arrays of $x$ and $y$ values, then it will compute the function for each pair of corresponding $(x,y)$ values and return an array of $z$ values. 
"""

x_opt, y_opt = funkier_minimize(0.05, 0.05, eta)
# the power of numpy!
z_opt = funkier_z(x_opt, y_opt)
plot_3D(x_opt, y_opt, z_opt)

# Explanation: From the starting point, we continuously step towards the minimum
# by stepping in the direction of the steepest descent. How large of a step we
# take is limited by our chosen step size. We stop stepping if there is a very 
# small difference between our next and current position or our next position 
# becomes too high.

"""Now try it again, starting at $(x,y) = (-0.05, -0.05)$ and using the same `eta`. Run gradient descent and plot the results below. """

x_opt2, y_opt2 = funkier_minimize(-0.05, -0.05, eta)
z_opt2 = funkier_z(x_opt2, y_opt2)
plot_3D(x_opt2, y_opt2, z_opt2)