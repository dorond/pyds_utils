import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scikitplot as skplt
import tarfile
from six.moves import urllib
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor
from scipy.stats import randint
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import get_scorer

def fetch_data(base_url, file_name, remote_sub_dir=None, file_ext="csv", base_local_path = "raw", local_store_sub_dir=None):
    '''
    Downloads file_name.file_ext from base_url & optional remote_sub_dir and stores it in relative base_local_path & 
    optional local_store_sub_dir. If file is a tarball, it will be extracted as well. 
    
    Issues: doesn't work with .csv.gz files  
    '''
    if remote_sub_dir is not None:
        download_url = base_url + remote_sub_dir + file_name + "." + file_ext
    else:
        download_url = base_url + file_name + "." + file_ext
    
    if not os.path.isdir(base_local_path):
         os.makedirs(base_local_path)
            
    if local_store_sub_dir is not None:
        local_path = os.path.join(base_local_path, local_store_sub_dir)
        if not os.path.isdir(local_path):
            os.makedirs(local_path)
    else:
        local_path = os.path.join(base_local_path)
    
    
    full_file_name = file_name + "." + file_ext
    file_path = os.path.join(local_path, full_file_name)
    
    print("Downloading file:", full_file_name)
    urllib.request.urlretrieve(download_url, file_path)
    
    if file_ext in ["tgz", "tar", "gz", "tar.gz"]:
        data_tgz = tarfile.open(file_path)
        data_tgz.extractall(path=local_path)
        data_tgz.close()

def load_data(base_path, file, sub_dir=None, ext="csv", encoding=None):
    filename = file + "." + ext
    if sub_dir is not None:
        csv_path = os.path.join(base_path, sub_dir, filename)
    else:
        csv_path = os.path.join(base_path, filename)
    return pd.read_csv(csv_path)

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

def plot_scatters(df, features, target, save_image=False, image_name=None):
    
    num_plots = len(features)
    num_cols = 2
    num_rows = int(np.ceil(num_plots / num_cols))
    
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, num_rows*5))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)

    for ax, feature in zip(ax.ravel(), features):
        df.plot.scatter(x=feature, y=target, ax=ax, alpha=0.2)
    
    if save_image:
        save_fig(image_name)
           
# Used to explore a single feature on a set of subplots. 
# Visualise distribution, noise & outliers and missing values as well as correlation with target.
def explore_variable(df, feature, target, by_categorical=None):
    '''
    Numerical features will display 3 plots: histogram, correlation between source feature and a target and a box plot for feature. 
    Optional by_categorical can be provided to show box plot by levels of a categorical variable.
    Categorical features will display 2 plots: bar chart of levels, median of target by feature levels.
    '''    
    feature_type = df[feature].dtype
    missing = df.apply(lambda x: sum(x.isnull())).loc[feature]
    print("'{}' is of type {} with {} missing values".format(feature, feature_type, missing))

    if feature_type == "object":
        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        fig.subplots_adjust(wspace=0.3)

        ax1 = ax.ravel()[0]
        ax1.set_title("Distribution of {}".format(feature))
        df[feature].value_counts().plot.barh(ax=ax1)

        ax2 = ax.ravel()[1]
        ax2.set_title("Median {} by {}".format(target, feature))
#         df.groupby(feature)[[target]].median().plot.barh(ax=ax2)
        if by_categorical is not None:
            pd.pivot_table(data=df, index=feature, values=target, columns=by_categorical).plot.barh(ax=ax2)
        else:
            pd.pivot_table(data=df, index=feature, values=target).plot.barh(ax=ax2)

        plt.show()

    elif feature_type == "int64" or "float64":
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        fig.subplots_adjust(wspace=0.3)

        ax1 = ax.ravel()[0]
        ax1.set_title("Distribution of {}".format(feature))
        df[feature].hist(bins=50, ax=ax1)

        ax2 = ax.ravel()[1]
        ax2.set_title("Correlation between {} and {}".format(feature, target))
        df.plot.scatter(x=feature, y=target, ax=ax2)  

        ax3 = ax.ravel()[2]
        ax3.set_title("Box plot for {}".format(feature))
        #df[feature].plot.box(ax=ax3)
        if by_categorical is not None:
            sns.boxplot(x=by_categorical, y=feature, data=df, ax=ax3)
        else:
            sns.boxplot(y=df[feature], ax=ax3)
        

        plt.show()

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_params):
        self.features = outlier_params["features"]
        self.irq_ranges = outlier_params["range"]
        self.remove_lows = outlier_params["remove_lows"]
        self.remove_highs = outlier_params["remove_highs"]
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):     
                
        first_quartile = X.describe().loc["25%"]
        third_quartile = X.describe().loc["75%"]
        iqr = third_quartile - first_quartile
                      
        # find rows subset that should be removed for each feature and irq range
        # Perform boolean AND to decide if row is to be removed - at least one True is needed.
        
        row_filters = []
        for feature, iqr_multiplier, remove_low, remove_high in zip(self.features, self.irq_ranges, self.remove_lows, self.remove_highs):
            
            if remove_low and remove_high: 
                low_high_filter = (X[feature] > (first_quartile[feature] - iqr_multiplier * iqr[feature])) &(X[feature] < (third_quartile[feature] + iqr_multiplier * iqr[feature]))
                row_filters.append(low_high_filter)
               
            
            elif remove_low and not remove_high:
                row_filters.append(X[feature] > (first_quartile[feature] - iqr_multiplier * iqr[feature]))
                
            elif not remove_low and remove_high:
                row_filters.append(X[feature] < (third_quartile[feature] + iqr_multiplier * iqr[feature]))
                
            else:
                pass
        
        row_filter = row_filters[0]        
        for option in row_filters[1:]:
            row_filter = row_filter & option
        
        return X[row_filter]

class AddLevelImputer(BaseEstimator, TransformerMixin):
    def __init__(self, na_dict, df_out=False):
        self.na_dict = na_dict
        self.df_out = df_out
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.df_out:
            return X.fillna(value=self.na_dict)
        else:
            return np.c_[X.fillna(value=self.na_dict)]
    
# Inspired from stackoverflow.com/questions/25239958

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def __init__(self, df_out=False):
        self.df_out = df_out
        
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)

        return self
    def transform(self, X, y=None):
        if self.df_out:
            return X.fillna(self.most_frequent_)
        else:
            return np.c_[X.fillna(self.most_frequent_)]
        
class FeatureTransformerAdder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_transform_params, df_out=False):
        self.features = feature_transform_params["features"]
        self.transformations = feature_transform_params["operations"]
        self.col_prefixes = feature_transform_params["prefixes"]
        self.remove_original = feature_transform_params["remove_original"]
        self.df_out = df_out
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for feature, transform, prefix, remove in zip(self.features, self.transformations, self.col_prefixes, self.remove_original):
            col_name = prefix + "_" + feature
            X[col_name] = X[feature].apply(transform)
            if remove:
                X.drop(feature, axis=1, inplace=True)
        if self.df_out:
            return X
        else:
            return np.c_[X]

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_transform_params, df_out=True):
        self.features = feature_transform_params["features"]
        self.transforms = feature_transform_params["transforms"]
        self.is_type_change = feature_transform_params["is_type_change"]       
        self.df_out = df_out 
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if isinstance(X,(pd.core.series.Series, pd.core.frame.DataFrame)):
            for feature, transform, is_change in zip(self.features, self.transforms, self.is_type_change):
                if is_change:                    
                    X[feature] = X[feature].astype(transform)
                else:
                    #print(X[feature])
                    X[feature] = X[feature].apply(transform)
        if self.df_out:
            return X
        else:
            return np.c_[X]

class InfImputer(BaseEstimator, TransformerMixin):
    def __init__(self, df_out=False):
        self.df_out = df_out
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        X[(X == np.inf) | (X == -np.inf)] = np.nan
        if self.df_out:
            return X
            
        else:
            return np.c_[X]

class CrossFeatureTransformerAdder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_adder_params, df_out=False): # no *args or **kargs
        self.base_features_ids = feature_adder_params["base_features_ids"]
        self.operations = feature_adder_params["operations"]
        self.df_out = df_out
        if self.df_out:
            self.feature_names = feature_adder_params["feature_names"]
        
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):      
        if self.df_out:
            new_features = []        
            for base_ids, operation, name in zip(self.base_features_ids, self.operations, self.feature_names): 
                X[name] = operation(X.ix[:, base_ids[0]], X.ix[:, base_ids[1]])
        else:
            new_features = []    
            if isinstance(X,(pd.core.series.Series, pd.core.frame.DataFrame)):
                X = X.values
            for base_ids, operation in zip(self.base_features_ids, self.operations):  
                new_features.append(operation(X[:, base_ids[0]], X[:, base_ids[1]]))        

            for feature in new_features:
                X = np.c_[X, feature]
        
        return X

# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
    
class TypeSelector(BaseEstimator, TransformerMixin):
    '''
    Select features of a specific type. if choose='all' then all features of that type will be selected and features will have no affect.
    If choose='select' then features mujst contain a list of features matching dtype
    '''
    def __init__(self, dtype, choose="all", features=None):
        self.dtype = dtype
        self.choose = choose
        self.features = features
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.choose == "all":
            return X.select_dtypes(include=[self.dtype])
        elif self.choose == "select":
            if self.features is not None:
                return X.select_dtypes(include=[self.dtype])[self.features]
                
            else:
                raise ValueError('features must contain a list of features of type', self.dtype, 'if choose="select"')

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

def determine_one_hot_cat_features(cat_pipeline, original_cat_features, bool_pipeline, original_bool_features):    
    mappings = []
    for feature, levels in zip(original_cat_features, cat_pipeline.named_steps["cat_encoder"].categories_):        
        mappings.append((feature, levels))
    for feature, levels in zip(original_bool_features, bool_pipeline.named_steps["bool_encoder"].categories_):        
        mappings.append((feature, levels))
   
    features_list = []
    for item in mappings:
        new_levels = [item[0] + "_" + str(level) for level in item[1]]
        features_list.append(new_levels)
      
    return [feature for group in features_list for feature in group]

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def build_measure_predict(alg, train, test, target, pipelines, remove_identifiers=None, which_features=None, 
                          which_features_top=10, cv=10, param_search=None, scoring="neg_mean_squared_error", 
                          show_summary=True, plot_feat_importances=True, how_many_important_features=30,
                          plot_learning_curve=True, make_predictions=False, y_true=None, prediction_scorer="mean_squared_error",
                          model_name="model", save_model=False, save_measure_image=False):
        
    outlier_pipeline = pipelines["outlier_pipeline"]
    feature_add_remove_pipeline = pipelines["feature_add_remove_pipeline"]
    feature_transform_pipeline = pipelines["feature_transform_pipeline"]
    num_pipelines = pipelines["num_pipelines"]
    cat_pipelines = pipelines["cat_pipelines"]
    bool_pipelines = pipelines["bool_pipelines"]
    
    train = train.copy()
    
    if outlier_pipeline is not None:
        train = outlier_pipeline.fit_transform(train)    

    if remove_identifiers is not None:
        print("\nRemoving features:", remove_identifiers)
        train = train.drop(remove_identifiers, axis=1)
    
    labels = train[target].copy()   
    train.drop([target], axis=1, inplace=True)
    
    if feature_add_remove_pipeline is not None:
        train = feature_add_remove_pipeline.fit_transform(train)
            
    if feature_transform_pipeline is not None:
        train = feature_transform_pipeline.fit_transform(train)
    
    numeric_feature_names = train.select_dtypes(include=[np.number]).columns.values.tolist()
    cat_feature_names = train.select_dtypes(include=['object']).columns.values.tolist()   
    bool_feature_names = train.select_dtypes(include=['bool']).columns.values.tolist()
    
    num_pipeline = Pipeline([("numerics", TypeSelector(np.number, "select", numeric_feature_names))])
    cat_pipeline = Pipeline([("categoricals", TypeSelector('object', "select", cat_feature_names))])
    bool_pipeline = Pipeline([("booleans", TypeSelector('bool', "select", bool_feature_names))])
  
    for pipeline in num_pipelines:
        num_pipeline.steps.append(pipeline)
        
    for pipeline in cat_pipelines:
        cat_pipeline.steps.append(pipeline)
        
    for pipeline in bool_pipelines:
        bool_pipeline.steps.append(pipeline)
        
    preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
        ("bool_pipeline", bool_pipeline),
    ])
    
    if which_features is not None:
        full_pipeline = Pipeline([
            ('preparation', preprocess_pipeline),
            ('feature_selection', TopFeatureSelector(which_features, which_features_top))
    ])
        
    else:
        full_pipeline = preprocess_pipeline
    
            
    if param_search is not None:
        search_method = param_search["method"]
        search_cv = param_search["cv"]
        search_iterations = param_search["iterations"]
        search_scoring = param_search["scoring"]
        search_params = param_search["params"] 
        
        if search_method == "random":           
            search = RandomizedSearchCV(alg, param_distributions=search_params, n_iter=search_iterations, 
                                        cv=search_cv, scoring=search_scoring, verbose=1, n_jobs=-1)
            
        elif search_method == "grid":           
            search = GridSearchCV(alg, search_params, cv=search_cv, scoring=search_scoring, verbose=1, n_jobs=-1)
    
    print('\nPreparing data using transformation pipeline...')
    train_prepared = full_pipeline.fit_transform(train)
    print("Data prepared")   
      
    complete_feature_list = numeric_feature_names + determine_one_hot_cat_features(cat_pipeline, cat_feature_names, bool_pipeline, bool_feature_names) 
   
    print('\nFitting model to training data...')
    if param_search:
        search.fit(train_prepared, labels)
        alg = search.best_estimator_
    else:
        alg.fit(train_prepared, labels)
    
    if param_search: 
        cvres = search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
        print("\nBest Params: ")
        print(search.best_params_)
        scores = np.sqrt(-search.cv_results_["mean_test_score"])
        
    else:
        print('\nRunning Cross-Validation...')
        scores = np.sqrt(-cross_val_score(alg, train_prepared, labels, scoring=scoring, cv=cv, verbose=1))
        print('\nCV complete\n')
    
       
    if feature_add_remove_pipeline is not None and feature_transform_pipeline is not None:
        pipelines = make_pipeline(feature_add_remove_pipeline, feature_transform_pipeline, full_pipeline)
    elif feature_add_remove_pipeline is not None and feature_transform_pipeline is None:
        pipelines = make_pipeline(feature_add_remove_pipeline, full_pipeline)
    elif feature_add_remove_pipeline is None and feature_transform_pipeline is not None:
        pipelines = make_pipeline(feature_transform_pipeline, full_pipeline)
    else:
        pipelines = full_pipeline
    
    model_data = {}
    model_data["model"] = alg
    model_data["features"] = complete_feature_list
    model_data["scores"] = scores
    model_data["outlier_pipeline"] = outlier_pipeline
    model_data["prep_pipeline"] = pipelines
    model_data["train_prepared"] = train_prepared
    model_data["labels"] = labels
    model_data["target"] = target
    model_data["scoring"] = scoring
    model_data["name"] = model_name
    
    if not isinstance(alg, (LinearRegression, SVR, BaggingRegressor)):
        model_data["feature_importances"] = alg.feature_importances_
        model_data["feature_coef"] = None
        model_data["oob_score"] = None
    elif isinstance(alg, (BaggingRegressor)):
        try:
            model_data["feature_coef"] = None
            model_data["feature_importances"] = None
            model_data["oob_score"] = alg.oob_score_ 
        except AttributeError:
            model_data["oob_score"] = None
    else:
        model_data["feature_coef"] = alg.coef_        
        model_data["feature_importances"] = None
        model_data["oob_score"] = None
        
    model = model_data["model"]
    scores = model_data["scores"]
    train_prepared = model_data["train_prepared"]
    target = model_data["target"]
    labels = model_data["labels"]
    trans_feature_names = model_data["features"]
    outlier_pipeline =  model_data["outlier_pipeline"]
    scoring =  model_data["scoring"]
    
    
    ################### Show Model Validation Metrics, Feature Importances & Learning Curve #################
    
    if show_summary:
        print('\nModel Report:')
        print ("\nCV Score : Mean - {:f} | Std - {:f} | Min - {:f} | Max - {:f}".format(np.mean(scores),np.std(scores),
                                                                                  np.min(scores),np.max(scores)))
    if plot_feat_importances & plot_learning_curve:
        num_plots = 2
        fig, ax = plt.subplots(2, 1, figsize=(20,20))
        fig.subplots_adjust(hspace=0.3)
        
    elif (plot_feat_importances and not plot_learning_curve) or (not plot_feat_importances and plot_learning_curve):
        num_plots = 1
        fig, ax = plt.subplots(1, 1, figsize=(20,10))
        fig.subplots_adjust(hspace=0.3)
    else:
        num_plots = 0
    
    if num_plots > 0:
        if num_plots == 1:
            ax1 = ax
        else:
            ax1 = ax.ravel()[0]
        
        if num_plots > 1:
            ax2 = ax.ravel()[1]
        
        if plot_feat_importances and plot_learning_curve:  
            
            if not isinstance(model, (LinearRegression, MLPRegressor, BaggingRegressor)):
                skplt.estimators.plot_feature_importances(model, feature_names = trans_feature_names, 
                                                      max_num_features=how_many_important_features, 
                                                          x_tick_rotation=60, ax=ax1, title_fontsize="large")
            elif isinstance(model, (BaggingRegressor)):
                if model_data["oob_score"] is not None:
                    print("Out-of-bag Estimate:", model_data["oob_score"])

            elif isinstance(model, (LinearRegression)):
                
                num_coeffs = len(model.coef_)
                neg_coef = pd.Series(model.coef_, trans_feature_names).sort_values().nsmallest(int(num_coeffs*0.05))
                pos_coef = pd.Series(model.coef_, trans_feature_names).sort_values().nlargest(int(num_coeffs*0.05))
                coef = neg_coef.append(pos_coef)      
                coef.plot(kind='bar', title='Feature Coefficients', ax=ax1)   
            else:
                num_coeffs = len(model.coefs_)
                neg_coef = pd.Series(model.coefs_, trans_feature_names).sort_values().nsmallest(int(num_coeffs*0.05))
                pos_coef = pd.Series(model.coefs_, trans_feature_names).sort_values().nlargest(int(num_coeffs*0.05))
                coef = neg_coef.append(pos_coef)      
                coef.plot(kind='bar', title='Feature Coefficients', ax=ax1)      
                     
            skplt.estimators.plot_learning_curve(model, train_prepared, labels, scoring=scoring, ax=ax2, 
                                                 shuffle=True, n_jobs=-1, title_fontsize="large")
            
        elif plot_feat_importances and not plot_learning_curve:
            if not isinstance(model, (LinearRegression, MLPRegressor, BaggingRegressor)):
                skplt.estimators.plot_feature_importances(model, feature_names = trans_feature_names, 
                                                      max_num_features=how_many_important_features, 
                                                          x_tick_rotation=60, ax=ax1, title_fontsize="large")
            elif isinstance(model, (BaggingRegressor)):
                if model_data["oob_score"] is not None:
                    print("Out-of-bag Estimate:", model_data["oob_score"])

            elif isinstance(model, (LinearRegression)):
                num_coeffs = len(model.coef_)
                neg_coef = pd.Series(model.coef_, trans_feature_names).sort_values().nsmallest(int(num_coeffs*0.05))
                pos_coef = pd.Series(model.coef_, trans_feature_names).sort_values().nlargest(int(num_coeffs*0.05))
                coef = neg_coef.append(pos_coef)      
                coef.plot(kind='bar', title='Feature Coefficients', ax=ax1)   
            else:
                num_coeffs = len(model.coefs_)
                neg_coef = pd.Series(model.coefs_, trans_feature_names).sort_values().nsmallest(int(num_coeffs*0.05))
                pos_coef = pd.Series(model.coefs_, trans_feature_names).sort_values().nlargest(int(num_coeffs*0.05))
                coef = neg_coef.append(pos_coef)      
                coef.plot(kind='bar', title='Feature Coefficients', ax=ax1)
           
        elif not plot_feat_importances and plot_learning_curve:
            skplt.estimators.plot_learning_curve(model, train_prepared, labels, scoring=scoring, ax=ax1, 
                                                 shuffle=True, n_jobs=-1, title_fontsize="large")
        
        if save_measure_image:
            save_fig(model_name)
     
    ################# Make predictions on a test set. Compare to y_true if it's available #########  
    if make_predictions:
        test = test.copy()
        model = model_data["model"]
        labels = model_data["labels"]
        fitted_pipeline = model_data["prep_pipeline"]
        
        if y_true is not None:
            scorer = get_scorer(prediction_scorer)

        if remove_identifiers is not None:
            test = test.drop(remove_identifiers, axis=1)

        test_prepared = fitted_pipeline.transform(test)
        final_predictions = model.predict(test_prepared)
        model_data["final_predictions"] = final_predictions
        model_data["test_prepared"] = test_prepared

        if y_true is not None:
            errors = np.sqrt(scorer(y_true, final_predictions))
            model_data["test_errors"] = error 
    
    if save_model:
        store_model(model_data, model_name)
        
    return model_data
        
class SpecificFeaturesAdderRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feature_adder_params, df_out=False):
        self.new_features_names = feature_adder_params["new_features_names"]
        self.features_to_remove = feature_adder_params["features_to_remove"]
        self.df_out = df_out
    def fit(self, X, y=None):
        self.new_features_names_ = self.new_features_names
        self.features_to_remove_ = self.features_to_remove
        self.original_df_columns_ = X.columns
        self.original_df_ = X
        return self  # nothing else to do
    def transform(self, X, y=None):
        X["HasRemod"] = X["YearBuilt"] != X["YearRemodAdd"] 
        X["YearSinceRemod"] = X["YearRemodAdd"] - X["YearBuilt"]
        X["YearSinceBuilt"] = X["YrSold"] -  X["YearBuilt"]
        X["YearBtwSoldRemod"] = X["YrSold"] - X["YearRemodAdd"]
        
        X["HasEnclosedPorch"] = X["EnclosedPorch"] != 0
        
        X["GrLivAreaBucket"] = X["GrLivArea"] // 1000 * 1000
        X["TotalBsmtSFBucket"] = X["TotalBsmtSF"] // 1000 * 1000
        X["GarageAreaBucket"] = X["GarageArea"] // 350 * 350
        X["BsmtFinSF1Bucket"] = X["BsmtFinSF1"] // 400 * 400
        X["1stFlrSFBucket"] = X["1stFlrSF"] // 800 * 800
        X["LotAreaBucket"] = X["LotArea"] // 20000 * 20000
        X["MasVnrAreaBucket"] = X["MasVnrArea"] // 500 * 500
        
        if self.features_to_remove is not None:           
            X = X.drop(self.features_to_remove_, axis=1)            
            
        if self.df_out:
            return X
        else:
            return np.c_[X]

def compare_scores(models, show_plot=True):
        df = pd.DataFrame.from_records([(model["name"], np.mean(model["scores"])) for model in models], columns=["Model", "Score"],
                                       index="Model")
        df = df.sort_values(by="Score", ascending=True)
               
        if show_plot:
            df.plot.barh(figsize=(12,8))
        
        return df

def compare_scores_distribution(scores, titles, y_label):
    plt.figure(figsize=(20, 9))
    num_scores = len(scores)
    for i in range(num_scores):
        plt.plot([i+1]*len(scores[i]), scores[i], ".")      
        plt.boxplot(scores, labels=titles)
        plt.ylabel(y_label, fontsize=14)
        #plt.ylim(0.1, 0.25)

