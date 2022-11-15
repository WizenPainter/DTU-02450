# %%
import sys
sys.path.insert(0, 'C:/Users/guzma/OneDrive/Documents/TEC/DTU/02450/Exercises/toolbox/02450Toolbox_Python/Tools')
from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net, visualize_decision_boundary


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import scipy.stats as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
%matplotlib inline

# %% [markdown]
# ## Data Cleansing

# %% [markdown]
# Data can be found in https://archive.ics.uci.edu/ml/datasets/heart+disease. The data taht will be used in this proyect requires we change the encoding of our pandas function so that it may read the characters correctly. Since the data doesn't include the header on the file we must set them ourselves utilizng the heart-disease.names file to guide us in their exact name and position, as well as see what we may expect from the variables in the file.

# %%
cleveland = pd.read_csv('processed.cleveland.csv', encoding="ISO-8859-1", header=None)
cleveland.set_axis(['age','sex','cp','testbps','chol','fbs', 'restecg','thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'], inplace=True, axis=1)
cleveland.head()

# %%
cleveland.shape

# %% [markdown]
# We find that the columns **thal** and **ca** both have the datatype object as they posses the **?** character amongs their values, so we have to replace this with a nan value and then delete the columns, as we find there are only 6 columns. We cannot change the values as this may affect the model later on even if it's just a few samples.

# %%
cleveland = cleveland.replace('?', np.nan)
cleveland.thal = cleveland.thal.astype('category')
cleveland.ca = cleveland.ca.astype(np.float64)
# cleveland.cp = cleveland.cp.astype('category')
# cleveland.fbs = cleveland.fbs.astype(bool)
# cleveland.sex = cleveland.sex.astype(bool)
# cleveland.exang = cleveland.exang.astype(bool)
cleveland.restecg = cleveland.restecg.astype(int)
cleveland.age = cleveland.age.astype(int) 
cleveland.thalach = cleveland.thalach.astype(int)
cleveland.slope = cleveland.slope.astype(int)

# %%
cleveland.isna().sum()

# %%
cleveland.dtypes

# %%
(6/303)*100

# %%
cleveland.dropna(inplace=True)

# %%
cleveland.isna().sum()

# %% [markdown]
# ## EDA

# %% [markdown]
# Now that our data has been cleaned correctly we can start the exploratory data analysis. We will first begin with a correlation matrix to identify how each variable affect one-another. We will do this utilizing the corr function from the pandas library and the heatmap visualization function from the seaborn library.

# %%
plt.figure(figsize=(13, 6), dpi=80)
sns.heatmap(cleveland.corr(), annot=True)
plt.show()

# %% [markdown]
# Now that we see all the correlation between the variables we can begin by searching if there are any outliers in the data that may affect the models we are looking to train.

# %%
# from seaborn_qqplot import pplot
figure, ax = plt.subplots(2,4,figsize=(18,10))
sns.distplot(x =cleveland.age ,ax=ax[0,0], kde=True, kde_kws={'bw':0.8})
ax[0,0].set_title('Age')
ax[0,0].set(xlabel=None)
sns.distplot(x=cleveland.cp, ax=ax[0,1], kde=True, kde_kws={'bw':0.8})
ax[0,1].set_title('Cp')
ax[0,1].set(xlabel=None)
sns.distplot(x=cleveland.testbps, ax=ax[0,2], kde=True, kde_kws={'bw':0.8})
ax[0,2].set_title('Testbps')
ax[0,2].set(xlabel=None)
sns.distplot(x=cleveland.chol, ax=ax[0,3], kde=True, kde_kws={'bw':0.8})
ax[0,3].set_title('Chol')
ax[0,3].set(xlabel=None)
sns.distplot(x=cleveland.thalach, ax=ax[1,0], kde=True, kde_kws={'bw':0.8})
ax[1,0].set_title('Thalach')
ax[1,0].set(xlabel=None)
sns.distplot(x=cleveland.oldpeak, ax=ax[1,1], kde=True, kde_kws={'bw':0.8})
ax[1,1].set_title('Oldpeak')
ax[1,1].set(xlabel=None)
sns.distplot(x=cleveland.slope, ax=ax[1,2], kde=True, kde_kws={'bw':0.8})
ax[1,2].set_title('Slope')
ax[1,2].set(xlabel=None)
sns.distplot(x=cleveland.ca, ax=ax[1,3], kde=True, kde_kws={'bw':0.8})
ax[1,3].set_title('Ca')
ax[1,3].set(xlabel=None)

# %%
figure, ax = plt.subplots(2,4,figsize=(18,10))
sns.boxplot(y=cleveland.age, ax=ax[0,0])
ax[0,0].set_title('Age')
ax[0,0].set(ylabel=None)
sns.boxplot(y=cleveland.cp, ax=ax[0,1])
ax[0,1].set_title('Cp')
ax[0,1].set(ylabel=None)
sns.boxplot(y=cleveland.testbps, ax=ax[0,2])
ax[0,2].set_title('Testbps')
ax[0,2].set(ylabel=None)
sns.boxplot(y=cleveland.chol, ax=ax[0,3])
ax[0,3].set_title('Chol')
ax[0,3].set(ylabel=None)
sns.boxplot(y=cleveland.thalach, ax=ax[1,0])
ax[1,0].set_title('Thalach')
ax[1,0].set(ylabel=None)
sns.boxplot(y=cleveland.oldpeak, ax=ax[1,1])
ax[1,1].set_title('Oldpeak')
ax[1,1].set(ylabel=None)
sns.boxplot(y=cleveland.slope, ax=ax[1,2])
ax[1,2].set_title('Slope')
ax[1,2].set(ylabel=None)
sns.boxplot(y=cleveland.ca, ax=ax[1,3])
ax[1,3].set_title('Ca')
ax[1,3].set(ylabel=None)

# %% [markdown]
# The variables that were not included in this graph are variables that contain boolean values, as such there is no need to plot this variables to check if there are any outliers, for we can easily find those with the unique function from the pandas library.

# %% [markdown]
# ## PCA

# %%
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# We need to change the values of num from a categorical variable to a boolean value that represents wheter the patient presents heart diseas of not.

# %%
cleveland.dtypes

# %%
# bol_df = cleveland.iloc[:,[1,5,8]]
X = cleveland.iloc[:,[0,2,3,4,6,7,9,10,11,12]]
# X = cleveland.iloc[:,0:13]
y = cleveland.iloc[:,13]
std = StandardScaler()
transformed = StandardScaler().fit_transform(X)

# %%
thres = 0.9
plt.figure(figsize=(10,8))
pca = PCA().fit(transformed)
plt.grid(visible=True)
plt.plot(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, 'x-')
plt.plot(range(1,len(pca.explained_variance_ratio_)+1),np.cumsum(pca.explained_variance_ratio_), 'o-')
plt.plot([1,len(pca.explained_variance_ratio_)],[thres, thres],'k--')
plt.text(1, thres-0.04, f"{thres}", ha="center", fontsize=12)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.legend(['Individual','Cumulative','Threshold'])
plt.show()

# %%
feature_weights = pca.components_
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15, 8))
ax1.barh(X.columns,feature_weights[0],color=['b','g','r','c'])
ax2.barh(X.columns,feature_weights[1],color=['b','g','r','c'])
ax1.set_title('Principal Component 1')
ax2.set_title('Principal Component 2')
# ax1.xlabel()
plt.show()

# %%
import seaborn as sns
feature_weights = pca.components_
# plt.figure(figsize=(13,1))
colors = ['orange', 'c', 'mediumspringgreen', 'lemonchiffon','mistyrose','linen','orange', 'c', 'mediumspringgreen', 'lemonchiffon','mistyrose','linen','orange']
fig, ax = plt.subplots(10,1, figsize=(20,40))
for i in range(len(feature_weights)):
    ax[i].barh(X.columns, feature_weights[i], color=colors[i])
    ax[i].set_title(f'Principal Component {i+1}')
# ax[1].barh(X.columns, feature_weights[1], color=colors[1])
# ax[1].set_title(f'Principal Component {2}')
# ax[2].barh(X.columns, feature_weights[2], color=colors[2])
# ax[2].set_title(f'Principal Component {3}')
    
# plt.barh(X.columns,feature_weights[0],color=['orange'], label='Principal Component 1')
# plt.barh(X.columns,feature_weights[1],color=['c'], label='Principal Component 2')
# plt.barh(X.columns,feature_weights[2],color=['mediumspringgreen'], label='Principal Component 3')
# plt.legend()
# ax1.xlabel()
plt.show()

# %%
len(X.columns)

# %%
import seaborn as sns
feature_weights = pca.components_
plt.figure(figsize=(18,8))
# fig, ax = plt.subplots(figsize=(15,8))
X_axis = np.arange(len(X.columns))
plt.bar(X_axis - 0.2,feature_weights[0],0.4,color='orange', label='Principal Component 1', alpha=0.7)
plt.bar(X_axis + 0.2,feature_weights[1],0.4,color='c', label='Principal Component 2', alpha=0.8)
plt.xticks(X_axis, X.columns)
# sns.barplot(y=X.columns,x=feature_weights[2],color='mediumspringgreen', label='Principal Component 3',alpha=0.8)
plt.legend()
plt.title('Attribute Contribution to Principal Components')
# ax1.xlabel()
plt.show()

# %%
sns.pairplot(cleveland,hue='num')

# %%
plt.figure(figsize=(18,10))
for pc in range(1):
    for att in range(len(feature_weights[pc])):
        plt.arrow(0,0, feature_weights[pc,att], feature_weights[pc+1,att])
        plt.text(feature_weights[pc,att], feature_weights[pc+1,att], X.columns[att], fontsize=12)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(visible=True)
    # Add a unit circle
plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
plt.title('Zero-mean' +'\n'+'Attribute coefficients', fontsize=20)
plt.axis('equal')

# %% [markdown]
# #### Clases

# %%
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

# %%
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

# %%
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)

# %%
from matplotlib.text import Annotation

class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

# %%
def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(Axes3D, 'annotate3D', _annotate3D)

# %% [markdown]
# #### Plot

# %%
fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')

pc=0
X_ax = []
Y_ax = []
Z_ax = []
for att in range(len(feature_weights[pc])):
    ax.arrow3D(0,1,0,
       feature_weights[pc,att],feature_weights[pc+1,att],feature_weights[pc+2,att],
       mutation_scale=20,
    #    ec='mediumspringgreen',
       arrowstyle="-|>",
       linestyle='dashed')
    ax.annotate3D(X.columns[att], (feature_weights[pc,att],feature_weights[pc+1,att]+1,feature_weights[pc+2,att]), xytext=(feature_weights[pc,att],feature_weights[pc+1,att]), textcoords='offset points')       
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('Zero-mean' +'\n'+'Attribute coefficients', fontsize=15)
plt.show()

# %%
cleveland.dtypes

# %%
# X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
principal_ca = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
print(pca.components_)
principalDf = pd.DataFrame(data = principal_ca
             , columns = ['principal component 1', 'principal component 2'])

# %%
pca_df = pd.concat([principalDf, y], axis=1)
pca_df.rename(columns={'num':'target'}, inplace=True)

# %%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,1]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pca_df['target'] == target
    ax.scatter(pca_df.loc[indicesToKeep, 'principal component 1']
               , pca_df.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

# %% [markdown]
# ## Linear Regression

# %% [markdown]
# ### Standarization

# %%
cleveland['num_bin'] = np.array([0 if x ==0 else 1 for x in cleveland.num])

# %%
clf = StandardScaler()
y = cleveland['num_bin']
X = cleveland.drop(['num','num_bin'], axis=1)
X['thal'] = X.thal.astype(float)
K = 10
# X = clf.fit_transform(X)

# %%
X_scaled = (X - X.mean()) / (X.max() - X.min())
print(f'The standard deviation is \n{X_scaled.max() - X_scaled.min()}')
print(f'The mean is \n{X_scaled.mean()}')

# %%
X_scaled = clf.fit_transform(X_scaled)
X_scaled

# %%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33)

# %%
y_train = np.array(y_train)

# %%
lambdas = np.power(10.,range(-5,9))
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas)

# %%
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
plt.xlabel('Regularization factor')
plt.ylabel('Mean Coefficient Values')
plt.grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
#legend(attributeNames[1:], loc='best')

plt.subplot(1,2,2)
plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
plt.xlabel('Regularization factor')
plt.ylabel('Squared error (crossvalidation)')
plt.legend(['Train error','Validation error'])
plt.grid()

# To inspect the used indices, use these print statements
#print('Cross validation fold {0}/{1}:'.format(k+1,K))
#print('Train indices: {0}'.format(train_index))
#print('Test indices: {0}/n'.format(test_index))

plt.show()

# %%
M = X_train.shape[1]
Error_test_rlr = np.empty((len(X_test),1))
Error_train_rlr = np.empty((len(X_train),1))
w_rlr = np.empty((M,K))

Xty = X_train.T @ y_train
XtX = X_train.T @ X_train

lambdaI = opt_lambda * np.eye(M)
lambdaI[0,0] = 0 
w_rlr[:,1] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
# Compute mean squared error with regularization with optimal lambda
Error_train_rlr[1] = np.square(y_train-X_train @ w_rlr[:,1]).sum(axis=0)/y_train.shape[0]
Error_test_rlr[1] = np.square(y_test-X_test @ w_rlr[:,1]).sum(axis=0)/y_test.shape[0]

# %%
lambda_interval = np.linspace(-8, 2, 10)
# lambda_interval = np.logspace(-8, 2, 50)

Error_train_sm = np.empty((len(X_train),1))
Error_test_sm = np.empty((len(X_test),1))

train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))
for k in range(0, len(lambda_interval)):
    # mdl = LinearRegression()
    # mdl = LogisticRegression(penalty='l2', C=1/opt_lambda)
    mdl = Ridge(alpha = opt_lambda)
    
    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T

    Error_train_sm[k] = np.square(y_train-mdl.predict(X_train)).sum()/y_train.shape[0]
    Error_test_sm[k] = np.square(y_test-mdl.predict(X_test)).sum()/y_test.shape[0]
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

# %%
seed=42

# %%
new_y = y.to_numpy()
X = X_scaled
K = 10
CV = KFold(K,shuffle=True, random_state=seed)
for k, (train_index, test_index) in enumerate(CV.split(X,new_y)):
    X_train = X[train_index,:]
    y_train = new_y[train_index]
    X_test = X[test_index,:]
    y_test = new_y[test_index]

    lambdas = np.power(10.,range(-5,9))
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas)

    lambda_interval = np.linspace(-8, 2, 10)
    # lambda_interval = np.logspace(-8, 2, 29)

    Error_train_sm = np.empty((len(X_train),1))
    Error_test_sm = np.empty((len(X_test),1))

    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    print(len(lambda_interval))
    for j in range(0, len(lambda_interval)):
        # mdl = LinearRegression()
        # mdl = LogisticRegression(penalty='l2', C=1/opt_lambda)
        mdl = Ridge(alpha = opt_lambda)
        
        mdl.fit(X_train, y_train)

        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T

        Error_train_sm[j] = np.square(y_train-mdl.predict(X_train)).sum()/y_train.shape[0]
        Error_test_sm[j] = np.square(y_test-mdl.predict(X_test)).sum()/y_test.shape[0]
        
        train_error_rate[j] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[j] = np.sum(y_test_est != y_test) / len(y_test)

        w_est = mdl.coef_[0] 
        coefficient_norm[j] = np.sqrt(np.sum(w_est**2))


# %%
Error_train_sm.shape
# Error_test_sm.shape

# %%
# plt.bar(np.arange(13),mdl.coef_)
X_columns = cleveland.drop(['num','num_bin'], axis=1)
X_columns = X_columns.columns
plt.figure(figsize=(15,5))
plt.bar(X_columns,mdl.coef_)
plt.xticks(rotation=45)

# %%
plt.figure(figsize=(20,6))
sns.heatmap(cleveland.corr()[12:13], annot=True)

# %%
plt.plot(Error_train_sm)
plt.plot(Error_test_sm)

# %%
print(f'The training MSE is {Error_train_sm.mean()}')
print(f'The test MSE is {Error_test_sm.mean()}')

# %%
mdl.score(X_test, y_test)

# %% [markdown]
# ### Baseline Model

# %%
baseline_train_error_array = []
baseline_test_error_array = []
baseline_test2_error_array = []
CV = KFold(K,shuffle=True, random_state=42)
for k, (train_index, test_index) in enumerate(CV.split(X,y)):
    baseline_train_error = np.square(y.iloc[train_index] - y.iloc[train_index].mean()).sum(axis=0)/ y.shape[0]
    baseline_test_error = np.square(y.iloc[test_index] - y.iloc[test_index].mean()).sum(axis=0)/ y.shape[0]
    baseline_test_error2 = np.square(y.iloc[test_index] - y.iloc[train_index].mean()).sum(axis=0)/ y.shape[0] # TA recommended
    print(f'The baseline training MSE is {baseline_train_error}')
    print(f'The baseline test MSE is {baseline_test_error}')
    print(f'The baseline test2 MSE is {baseline_test_error2} \n', '-'*50)
    baseline_train_error_array.append(baseline_train_error)
    baseline_test_error_array.append(baseline_test_error)
    baseline_test2_error_array.append(baseline_test_error2)

# %% [markdown]
# ### ANN

# %%
from sklearn.model_selection import KFold
import torch
K = 10
CV = KFold(K,shuffle=True)

X = X_scaled
M = X.shape[1]
# y = y.to_numpy()

# Setup figure for display of the decision boundary for the several crossvalidation folds.
decision_boundaries = plt.figure(1, figsize=(10,10))
# Determine a size of a plot grid that fits visualizations for the chosen number
# of cross-validation splits, if K=4, this is simply a 2-by-2 grid.
subplot_size_1 = int(np.floor(np.sqrt(K))) 
subplot_size_2 = int(np.ceil(K/subplot_size_1))
# Set overall title for all of the subplots
plt.suptitle('Data and model decision boundaries', fontsize=20)
# Change spacing of subplots
plt.subplots_adjust(left=0, bottom=0, right=1, top=.9, wspace=.5, hspace=0.25)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

# Define the model structure
n_hidden_units = 15 # number of hidden units in the signle hidden layer
# The lambda-syntax defines an anonymous function, which is used here to 
# make it easy to make new networks within each cross validation fold
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.ReLU(),     
                    torch.nn.Linear(n_hidden_units, n_hidden_units*2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units*2, n_hidden_units*3),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units*3, 1), # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function
                    )
# Since we're training a neural network for binary classification, we use a 
# binary cross entropy loss (see the help(train_neural_net) for more on
# the loss_fn input to the function)
loss_fn = torch.nn.BCELoss()
# Train for a maximum of 10000 steps, or until convergence (see help for the 
# function train_neural_net() for more on the tolerance/convergence))
max_iter = 10000
print('Training model of type:\n{}\n'.format(str(model())))

# Do cross-validation:
val_errors = []
errors = [] # make a list for storing generalizaition error in each loop
# Loop over each cross-validation split. The CV.split-method returns the 
# indices to be used for training and testing in each split, and calling 
# the enumerate-method with this simply returns this indices along with 
# a counter k:

# Partition_index is used to create the second level cross validation with a training and a validation set.
for k, (partition_index, test_index) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and validation set for second CV fold, 
    # and convert them to PyTorch tensors
    CV2 = KFold(K,shuffle=True)
    for k, (train_index, validation_index) in enumerate(CV2.split(partition_index)):
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y.iloc[train_index].to_numpy())
        X_validation = torch.Tensor(X[validation_index,:])
        y_validation = torch.Tensor(y.iloc[validation_index].to_numpy())

        # TODO
        # print(X_train.reshape(X_train.shape[0],1).size())
        net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=torch.squeeze(X_train),
                                                       y=y_train,
                                                       n_replicates=3,
                                                       max_iter=max_iter)

        # Determine estimated class labels for test set
        y_sigmoid = net(X_validation) # activation of final note, i.e. prediction of network
        y_val_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
        y_val_est = y_validation.type(dtype=torch.uint8)
        # Determine errors and error rate
        e = (y_val_est != y_validation)
        val_error_rate = (sum(e).type(torch.float)/len(y_validation)).data.numpy()
        val_errors.append(val_error_rate) # store error rate for current CV fold 

    # X_train = torch.Tensor(X[train_index,:] )
    # y_train = torch.Tensor(y[train_index] )
    X_test = torch.Tensor(X[test_index,:] )
    y_test = torch.Tensor(y.iloc[test_index].to_numpy())
    
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)

    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e = (y_test_est != y_test)
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    errors.append(error_rate) # store error rate for current CV fold 
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
# Display the error rate across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('Error rate');
summaries_axes[1].set_title('Test misclassification rates')
    
# Show the plots
plt.show(decision_boundaries.number) # try these lines if the following code fails (depends on package versions)
plt.show(summaries.number)
plt.show()

# Display a diagram of the best network in last fold
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,3]]
# draw_neural_net(weights, biases, tf)

# Print the average classification error rate
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))

# %% [markdown]
# 6: Generalization error/average error rate: 23.9099%
# 8: Generalization error/average error rate: 22.9009%
# 15: Generalization error/average error rate: 22.8649%
# 50: Generalization error/average error rate: 22.9099%
# 120: Generalization error/average error rate: 23.2162%

# %% [markdown]
# ### ANN easy

# %%
from sklearn.model_selection import KFold
import torch

# %%
K = 10
CV = KFold(K,shuffle=True)

X = X_scaled
M = X.shape[1]
# y = y.to_numpy()

# Setup figure for display of the decision boundary for the several crossvalidation folds.
decision_boundaries = plt.figure(1, figsize=(10,10))
# Determine a size of a plot grid that fits visualizations for the chosen number
# of cross-validation splits, if K=4, this is simply a 2-by-2 grid.
subplot_size_1 = int(np.floor(np.sqrt(K))) 
subplot_size_2 = int(np.ceil(K/subplot_size_1))
# Set overall title for all of the subplots
plt.suptitle('Data and model decision boundaries', fontsize=20)
# Change spacing of subplots
plt.subplots_adjust(left=0, bottom=0, right=1, top=.9, wspace=.5, hspace=0.25)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

# Define the model structure
n_hidden_units = 15 # number of hidden units in the signle hidden layer
# The lambda-syntax defines an anonymous function, which is used here to 
# make it easy to make new networks within each cross validation fold
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units, 1), #M features to H hiden units # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function
                    )
# Since we're training a neural network for binary classification, we use a 
# binary cross entropy loss (see the help(train_neural_net) for more on
# the loss_fn input to the function)
loss_fn = torch.nn.BCELoss()
# Train for a maximum of 10000 steps, or until convergence (see help for the 
# function train_neural_net() for more on the tolerance/convergence))
max_iter = 100 # should be 10000
print('Training model of type:\n{}\n'.format(str(model())))

# Do cross-validation:
val_errors = []
errors_easy = [] # make a list for storing generalizaition error in each loop
# Loop over each cross-validation split. The CV.split-method returns the 
# indices to be used for training and testing in each split, and calling 
# the enumerate-method with this simply returns this indices along with 
# a counter k:

# Partition_index is used to create the second level cross validation with a training and a validation set.
for k, (partition_index, test_index) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and validation set for second CV fold, 
    # and convert them to PyTorch tensors
    CV2 = KFold(K,shuffle=True)
    for k, (train_index, validation_index) in enumerate(CV2.split(partition_index)):
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y.iloc[train_index].to_numpy())
        X_validation = torch.Tensor(X[validation_index,:])
        y_validation = torch.Tensor(y.iloc[validation_index].to_numpy())

        # print(X_train.reshape(X_train.shape[0],1).size())
        net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=torch.squeeze(X_train),
                                                       y=y_train,
                                                       n_replicates=3,
                                                       max_iter=max_iter)

        # Determine estimated class labels for test set
        y_sigmoid = net(X_validation) # activation of final note, i.e. prediction of network
        y_val_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
        y_val_est = y_validation.type(dtype=torch.uint8)
        # Determine errors and error rate
        e = (y_val_est != y_validation)
        val_error_rate = (sum(e).type(torch.float)/len(y_validation)).data.numpy()
        val_errors.append(val_error_rate) # store error rate for current CV fold 

    # X_train = torch.Tensor(X[train_index,:] )
    # y_train = torch.Tensor(y[train_index] )
    X_test = torch.Tensor(X[test_index,:] )
    y_test = torch.Tensor(y.iloc[test_index].to_numpy())
    
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)

    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e = (y_test_est != y_test)
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    errors_easy.append(error_rate) # store error rate for current CV fold 
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
# Display the error rate across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors_easy)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('Error rate');
summaries_axes[1].set_title('Test misclassification rates')
    
# Show the plots
plt.show(decision_boundaries.number) # try these lines if the following code fails (depends on package versions)
plt.show(summaries.number)
plt.show()

# Display a diagram of the best network in last fold
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,3]]
# draw_neural_net(weights, biases, tf)

# Print the average classification error rate
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors_easy),4)))

# %%
print(len(errors_easy))
print(len(baseline_test2_error_array))
print(len(Error_test_sm[-11:-1]))

# %%
redige_error = Error_test_sm[-11:-1]
redige_error = redige_error.squeeze()

# %%
# redige_error = Error_test_sm[-11:-1]
table = {'ANN': errors_easy,
'Base Line Model': baseline_test2_error_array,
'Ridge Regression':redige_error}
comparison = pd.DataFrame(table)

# %%
comparison.head()

# %%
errors_easy = np.array(errors_easy)
baseline_test2_error_array = np.array(baseline_test2_error_array)
redige_error = np.array(redige_error)

# %%
##Paired T-test
alpha = 0.05

#RLR vs Baseline
z1 = redige_error.reshape(len(redige_error)) - baseline_test2_error_array.reshape(len(baseline_test2_error_array))

#ANN vs Baseline
z2 = errors_easy.reshape(len(errors_easy)) - baseline_test2_error_array.reshape(len(baseline_test2_error_array))

#RLR vs ANN
z3 = redige_error.reshape(len(redige_error)) - errors_easy.reshape(len(errors_easy))

CI1 = st.t.interval(1-alpha, len(z1)-1, loc=np.mean(z1), scale=st.sem(z1))  # Confidence interval (Will not be generated for some reason)
p1 = st.t.cdf( -np.abs( np.mean(z1) )/st.sem(z1), df=len(z1)-1)  # p-value of z1

CI2 = st.t.interval(1-alpha, len(z2)-1, loc=np.mean(z2), scale=st.sem(z2))  # Confidence interval (Will not be generated for some reason)
p2 = st.t.cdf( -np.abs( np.mean(z2) )/st.sem(z2), df=len(z2)-1)  # p-value of z2

CI3 = st.t.interval(1-alpha, len(z3)-1, loc=np.mean(z3), scale=st.sem(z3))  # Confidence interval (Will not be generated for some reason)
p3 = st.t.cdf( -np.abs( np.mean(z3) )/st.sem(z3), df=len(z3)-1)  # p-value of z3

# %%
print('CI1: ', CI1)
print('p1: ', p1)
print('CI2: ', CI2)
print('p2: ', p2)
print('CI3: ', CI3)
print('p3: ', p3)

# %% [markdown]
# ## Classification Model

# %% [markdown]
# ### ANN Classification

# %% [markdown]
# Do one hot encoding for y *maybe

# %%
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(13, 6),
                    torch.nn.ReLU(),
                    torch.nn.Linear(6, 5), #M features to H hiden units # H hidden units to 1 output neuron
                    torch.nn.Softmax(dim=1) # final tranfer function # TODO maybe change to ReLU 
                    )
net = model()
loss_fn = torch.nn.CrossEntropyLoss()
torch.nn.init.xavier_uniform_(net[0].weight)
torch.nn.init.xavier_uniform_(net[2].weight)
optimizer = torch.optim.Adam(net.parameters())
CV2 = KFold(K,shuffle=True)
for k, (train_index, validation_index) in enumerate(CV2.split(partition_index)):
    X_train = torch.Tensor(X[train_index,:])
    X_train = torch.squeeze(X_train)
    y_train = torch.tensor(y.iloc[train_index].to_numpy(), dtype=torch.long)
    # torch.tensor(y_train, dtype=torch.long)
    # print(y_train)
    # y_train = torch.Tensor(y.iloc[train_index].to_numpy())
    # print(X_train.size())
    y_Est = net(X_train)
    y_Est = y_Est.view(X_train.shape[0],-1)
    print(y_Est.size())
    print(y_train.size())
    loss_fn(y_Est, y_train)

# %%
from sklearn.model_selection import KFold
import torch
K = 10
CV = KFold(K,shuffle=True, random_state=seed)

X = X_scaled
M = X.shape[1]
y = cleveland.num
# y = y.to_numpy()
attributeNames = X_columns[11:]
classNames = cleveland.num.unique()

# Setup figure for display of the decision boundary for the several crossvalidation folds.
decision_boundaries = plt.figure(1, figsize=(10,10))
# Determine a size of a plot grid that fits visualizations for the chosen number
# of cross-validation splits, if K=4, this is simply a 2-by-2 grid.
subplot_size_1 = int(np.floor(np.sqrt(K))) 
subplot_size_2 = int(np.ceil(K/subplot_size_1))
# Set overall title for all of the subplots
plt.suptitle('Data and model decision boundaries', fontsize=20)
# Change spacing of subplots
plt.subplots_adjust(left=0, bottom=0, right=1, top=.9, wspace=.5, hspace=0.25)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

# Define the model structure
n_hidden_units = 15 # number of hidden units in the signle hidden layer
# The lambda-syntax defines an anonymous function, which is used here to 
# make it easy to make new networks within each cross validation fold
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units, 5), #M features to H hiden units # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function # TODO maybe change to ReLU 
                    )
# Since we're training a neural network for binary classification, we use a 
# binary cross entropy loss (see the help(train_neural_net) for more on
# the loss_fn input to the function)
loss_fn = torch.nn.CrossEntropyLoss()
# Train for a maximum of 10000 steps, or until convergence (see help for the 
# function train_neural_net() for more on the tolerance/convergence))
max_iter = 800 # should be 10000
print('Training model of type:\n{}\n'.format(str(model())))

# Do cross-validation:
val_errors = []
errors_easy = [] # make a list for storing generalizaition error in each loop
# Loop over each cross-validation split. The CV.split-method returns the 
# indices to be used for training and testing in each split, and calling 
# the enumerate-method with this simply returns this indices along with 
# a counter k:

# Partition_index is used to create the second level cross validation with a training and a validation set.
for k, (partition_index, test_index) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and validation set for second CV fold, 
    # and convert them to PyTorch tensors
    CV2 = KFold(K,shuffle=True, random_state=seed)
    for k, (train_index, validation_index) in enumerate(CV2.split(partition_index)):
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.tensor(y.iloc[train_index].to_numpy(), dtype=torch.long)
        # y_train = torch.Tensor(y[train_index])
        X_validation = torch.Tensor(X[validation_index,:])
        y_validation = torch.tensor(y.iloc[validation_index].to_numpy(), dtype=torch.long)
        # y_validation = torch.Tensor(y[validation_index])
        print(len(X_train))

        # print(X_train.reshape(X_train.shape[0],1).size())
        net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=torch.squeeze(X_train),
                                                       y=y_train,
                                                       n_replicates=3,
                                                       max_iter=max_iter)

        # Determine estimated class labels for test set
        y_sigmoid = net(X_validation) # activation of final note, i.e. prediction of network
        y_val_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
        y_val_est = y_validation.type(dtype=torch.uint8)
        # Determine errors and error rate
        e = (y_val_est != y_validation)
        val_error_rate = (sum(e).type(torch.float)/len(y_validation)).data.numpy()
        val_errors.append(val_error_rate) # store error rate for current CV fold 

    # X_train = torch.Tensor(X[train_index,:] )
    # y_train = torch.Tensor(y[train_index] )
    X_test = torch.Tensor(X[test_index,:] )
    y_test = torch.Tensor(y.iloc[test_index].to_numpy())
    # y_test = torch.Tensor(y[test_index] )
    
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)

    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e = (y_test_est != y_test)
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    errors_easy.append(error_rate) # store error rate for current CV fold 
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
# Display the error rate across folds
# summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors_easy)), color=color_list)
# summaries_axes[1].set_xlabel('Fold');
# summaries_axes[1].set_xticks(np.arange(1, K+1))
# summaries_axes[1].set_ylabel('Error rate');
# summaries_axes[1].set_title('Test misclassification rates')
    
# # Show the plots
# plt.show(decision_boundaries.number) # try these lines if the following code fails (depends on package versions)
# plt.show(summaries.number)
# plt.show()

# # Display a diagram of the best network in last fold
# print('Diagram of best neural net in last fold:')
# weights = [net[i].weight.data.numpy().T for i in [0,2]]
# biases = [net[i].bias.data.numpy() for i in [0,2]]
# tf =  [str(net[i]) for i in [1,3]]
# draw_neural_net(weights, biases, tf)
softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
# Get the estimated class as the class with highest probability (argmax on softmax_logits)
y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy() 
# Determine errors
y_test = y.iloc[test_index].to_numpy()
e = (y_test_est != y_test)
print('Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(sum(e),len(e)))

predict = lambda x:  (torch.max(net(torch.tensor(x, dtype=torch.float)), dim=1)[1]).data.numpy() 
# plt.figure(1,figsize=(9,9))
# visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
# plt.title('ANN decision boundaries')

# Print the average classification error rate
# print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))

# %% [markdown]
# ### Logistic Regression

# %%
new_y = y.to_numpy()
X = X_scaled
K = 10
CV = KFold(K,shuffle=True, random_state=seed)
for k, (train_index, test_index) in enumerate(CV.split(X,new_y)):
    X_train = X[train_index,:]
    y_train = new_y[train_index]
    X_test = X[test_index,:]
    y_test = new_y[test_index]

    lambdas = np.power(10.,range(-5,9))
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas)

    # lambda_interval = np.linspace(-8, 2, 10)
    lambda_interval = np.logspace(-8, 2, 29)

    Error_train_sm = np.empty((len(X_train),1))
    Error_test_sm = np.empty((len(X_test),1))

    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    print(len(lambda_interval))
    for j in range(0, len(lambda_interval)):
        # mdl = LinearRegression()
        mdl = LogisticRegression(penalty='l2', C=1/opt_lambda)
        # mdl = Ridge(alpha = opt_lambda)
        
        mdl.fit(X_train, y_train)

        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T

        Error_train_sm[j] = np.square(y_train-mdl.predict(X_train)).sum()/y_train.shape[0]
        Error_test_sm[j] = np.square(y_test-mdl.predict(X_test)).sum()/y_test.shape[0]
        
        train_error_rate[j] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[j] = np.sum(y_test_est != y_test) / len(y_test)

        w_est = mdl.coef_[0] 
        coefficient_norm[j] = np.sqrt(np.sum(w_est**2))


# %% [markdown]
# ### BaseLine model

# %%
baseline_train_error_array = []
baseline_test_error_array = []
baseline_test2_error_array = []
CV = KFold(K,shuffle=True)
for k, (train_index, test_index) in enumerate(CV.split(X,y)):
    baseline_train_error = np.square(y.iloc[train_index] - y.iloc[train_index].mean()).sum(axis=0)/ y.shape[0]
    baseline_test_error = np.square(y.iloc[test_index] - y.iloc[test_index].mean()).sum(axis=0)/ y.shape[0]
    baseline_test_error2 = np.square(y.iloc[test_index] - y.iloc[train_index].mean()).sum(axis=0)/ y.shape[0] # TA recommended
    print(f'The baseline training MSE is {baseline_train_error}')
    print(f'The baseline test MSE is {baseline_test_error}')
    print(f'The baseline test2 MSE is {baseline_test_error2} \n', '-'*50)
    baseline_train_error_array.append(baseline_train_error)
    baseline_test_error_array.append(baseline_test_error)
    baseline_test2_error_array.append(baseline_test_error2)

# %% [markdown]
# ### McNemar Test

# %% [markdown]
# ### Model

# %%
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_test, y_test)

# %%
reg.coef_

# %%
reg.intercept_

# %%
y_pred = reg.predict(X_test)
print('MAE: {:1.8f}'.format(mean_absolute_error(y_test, y_pred)))
print('MSE: {:1.8f}'.format(mean_squared_error(y_test, y_pred)))
print('RMSE: {:1.8f}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

# %%
import numpy as np

def jaccard_binary(x,y):
    """A function for finding the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    similarity = intersection.sum() / float(union.sum())
    return similarity

# Define some binary vectors
x = [1,1,1,1,1,1,1,1,0,0,0,0,0]
y = [1,0,0,1,0,0,0,0,1,1,1,1,1]

# Find similarity among the vectors
simxy = jaccard_binary(x,y)

print(' Similarity between x and y is', simxy)


