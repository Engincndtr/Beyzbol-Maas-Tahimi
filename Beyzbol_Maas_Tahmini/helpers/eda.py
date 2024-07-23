from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve


def check_df(dataframe):
    """
    You can learn general information about a dataframe
    Shape, type, head, tail, describe, info, sum of null values, how much are there any unique values and quantiles
    Parameters
    ----------
    dataframe: dataframe

    Returns
    -------
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        check_df(df)
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### Describe #####################")
    print(dataframe.describe().T)
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### N unique #####################")
    print(dataframe.nunique())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat


def num_summary(dataframe, numerical_col, plot=False):
    """
    You can learn quantile values of numerical column
    Parameters
    ----------
    dataframe: dataframe
        The dataframe which quantile values are to be retrieved
    numerical_col: list
        Numerical variable list
    plot: bool
        if you want to see histogram graph, it should be true

    Returns
    -------

    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def cat_summary(dataframe, col_name, plot=False):
    """

    Parameters
    ----------
    dataframe: dataframe
        The dataframe which value counts of categorical variables are to be retrieved
    col_name: list
        Categorical variable list
    plot: bool
        if you want to see histogram graph, it should be true

    Returns
    -------

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


###################################
# Missing
###################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def missing_vs_target(dataframe, target):
    temp_df = dataframe.copy()

    for col in missing_values_table(dataframe, True):
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flasg = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


###################################
# Outliers
###################################
def outlier_thresholds(dataframe, col_name, q1=0.5, q3=0.95):
    """
    Find lower and upper limit values for outlier of a dataframe column
    Parameters
    ----------
    dataframe: dataframe
    col_name: string
        column name of dataframe
    q1: float
    q3: float
    Returns
    -------
    low_limit : float
        low threshold limit
    up_limit : float
        high threshold limit
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """
    Check outlier a column of dataframe
    Parameters
    ----------
    dataframe: dataframe
    col_name: string
        column name of dataframe

    Returns
    -------
        if there are any outlier value returns true, otherwise false
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())

    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


###################################
# Encoding
###################################
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

"""
Encoding : Değişkenlerin temsil değişkenlerinin değiştirilmesi

Amaç: Algoritmaların bizden beklediği yapı var ona dönüştürmek. 
    Örneğin one hot encoder işlemlerinde Bir değişkenin önemli olabilecek sınıfını değişkenleştirerek
    ona bir değer atfetmek
"""


def label_encoder(dataframe, binary_col):
    """
        Label encoding de string şeklindeki ifadeleri numerik şeklinde ifade edilecek şekle değiştirilmesidir.
    Parameters
    ----------
    dataframe: dataframe
    binary_col: columns
        Ordinal bir şekilde sıralanamayacak değişkenleri burada değerlendirmemek gerekir.
        Örneğin futbol takımları arasında fark olmadığı için Labelencoder edildiğinde ölçüm
        problemleri ortaya çıkaracaktır
    Returns
    -------
    dataframe: dataframe
        Labelencoder den transform edilmiş, yeni değişken oluşturulmuş dataframe
    """
    dataframe["NEW_" + binary_col + "_num"] = LabelEncoder().fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True, dummy_na = False):
    """
    One hot encoder ile sınıfları değişkenlere dönüştürüyoruz
    Parameters
    ----------
    dataframe: dataframe
    categorical_cols: columns
    drop_first: boolean
        Oluşturduğumuz değişkenlere dummy(kukla) değişken denir.
        Eğer kukla değişkenler birbiri üzerinden oluşturulabilir olursa
        bu durumda ortaya ölçüm problemleri çıkmaktadır.
        Birbirleri üzerinden oluşturulabilen değişkenler yüksek bir korelasyona sebep olacaktır.
        Bundan dolayı dummy değişken oluşturulurken ilk değişken drop edilebilir

        Alfabetik sıraya göre ilk sınıfı seçer ve siler.

    Returns
    -------
    dataframe : Sınıfları değişkene dönüşmüş dataframe
    -------
    Not: Label encoder de sınıflar arası farklılık yokken varmış gibi davranması
    doğru olmayacağından dolayı onehot encoder kullanılır.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dummy_na=dummy_na)
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


###################################
# Encoding
###################################

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """
              0         1         2         3
    0  1.000000  0.117570  0.871754  0.817941
    1  0.117570  1.000000  0.428440  0.366126
    2  0.871754  0.428440  1.000000  0.962865
    3  0.817941  0.366126  0.962865  1.000000


        0        1         2         3
    0 NaN  0.11757  0.871754  0.817941
    1 NaN      NaN  0.428440  0.366126
    2 NaN      NaN       NaN  0.962865
    3 NaN      NaN       NaN       NaN


    Bir kare matriste  köşegen elemanlarının altında kalan bütün elemanlar sıfıra eşitse üst üçgen matris ()
    ve eğer köşegen elamanların üstünde kalan her eleman sıfıra eşitse buna da alt üçgen matris () denir.
    Aşağıda üst üçgen ve alt üçgen matris tanımına uyan örnekler verilmektedir.

    k: Üzerinde öğelerin sıfırlanacağı köşegen. k = 0 (varsayılan) ana köşegendir, k < 0 onun altındadır ve k > 0 üsttedir.
    Triu: Bir dizinin üst üçgeni. k'inci diyagonalin altındaki öğeleri sıfırlanmış bir dizinin bir kopyasını döndür.

    Parameters
    ----------
    dataframe
    plot
    corr_th

    Returns
    -------

    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


#####################
# Cross validate
#####################

def grab_cross_validate_results(model, X, y, cv=10):
    cv_results = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print("**************** Test Accuracy ****************")
    print(cv_results['test_accuracy'].mean())
    print("**************** Test F1 ****************")
    print(cv_results['test_f1'].mean())
    print("**************** Test ROC AUC ****************")
    print(cv_results['test_roc_auc'].mean())
    print("**************** Test Precision ****************")
    print(cv_results['test_precision'].mean())
    print("**************** Test recall ****************")
    print(cv_results['test_recall'].mean())



def plot_importance(model, features, num, save=False):
    num = len(model)
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)
