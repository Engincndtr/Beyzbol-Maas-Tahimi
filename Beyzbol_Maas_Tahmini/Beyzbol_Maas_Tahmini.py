import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from datetime import date

import target
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor
from sklearn.preprocessing import (
    MinMaxScaler,
    LabelEncoder,
    StandardScaler,
    RobustScaler,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import helpers.eda as eda


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)
warnings.simplefilter(action="ignore")


####### ADIM 1#######  : Keşifçi veri analizi

df = pd.read_csv("datasets/hitters.csv")
#df.info() # veri hakkında genel bilgi verir

##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

cat_cols, num_cols, cat_but_car, num_but_cat = eda.grab_col_names(df)


# Kategorik Değişken Analizi (Analysis of Categorical Variables)
######################################

for col in cat_cols:
    eda.cat_summary(df, col)

######################################

# 3. Sayısal Değişken Analizi
#############################

for col in num_cols:
    eda.num_summary(df, col)



######################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
######################################

for col in num_cols:
    eda.target_summary_with_cat(df, "Salary", col)


# Bağımlı değişkenin incelenmesi
df["Salary"].hist(bins=100)
# plt.show()

# Bağımlı değişkenin logaritmasının incelenmesi
np.log1p(df['Salary']).hist(bins=50)
# plt.show()

###kartillerin grafiği aykırılık
for col in num_cols:
  sns.boxplot(x = df[col])
  plt.show(block=True)
  print(" ")

def outlier_thresholds(df, q1=0.72, q3=0.83):
    quartile1 = df["Salary"].quantile(q1)
    quartile3 = df["Salary"].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# # Örnek kullanım
# # df = pd.read_csv('veri_dosyasi.csv') # Veri setini yükleme
low_limit, up_limit = outlier_thresholds(df)
print(f"Aykırı Değer Alt Sınırı: {low_limit}")
print(f"Aykırı Değer Üst Sınırı: {up_limit}")


# Aykırı Değer Analizi
######################################
# Salary sütununa göre sıralama yap ve indexleri sıfırla
df = df.sort_values(by='Salary').reset_index(drop=True)
# IQR hesaplaması
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
# Alt ve üst sınırları belirle, alt sınırı sıfıra ayarla
lower_bound = max(Q1 - 1.5 * IQR, 0)
upper_bound = Q3 + 1.5 * IQR
# Aykırı değerleri belirle
outliers = df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]
# Aykırı değerleri say
num_outliers = outliers.shape[0]
# Aykırı değerleri yazdır
# print("Toplam aykırı değer sayısı:", num_outliers)
# Alt ve üst limitleri yazdır
# print('Alt limit=', lower_bound, 'Üst limit=', upper_bound)

# Aykırı değerleri baskılama (Winsorization)
df['Salary'] = df['Salary'].clip(lower=lower_bound, upper=upper_bound)

# Sonuçları kontrol etmek için
# print(df['Salary'])


####eksik veri analizi
df.isna().sum()
df.isnull().sum()
# print(df.isna().sum())

def missing_values_table(df, na_name=False):
    # Eksik değere sahip sütunları belirleme
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]

    # Eksik değerlerin sayısını ve oranlarını hesaplama
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    # Eksik değer tablosunu yazdırma
    print(missing_df, end="\n")

    # Eğer na_name True ise, eksik değere sahip sütun isimlerini döndür
    if na_name:
        return na_columns
    print(missing_df, end="\n")
print(df.isna().sum())

def fill_missing_with_mean(df, columns):
    for col in columns:
        if df[col].isnull().any():  # Eğer sütunda eksik değer varsa
            mean_value = df[col].mean()  # Sütunun ortalamasını hesapla
            df[col].fillna(mean_value, inplace=True)  # Eksik değerleri ortalamayla doldur

# Örnek kullanım
missing_columns = missing_values_table(df, na_name=True)
fill_missing_with_mean(df, missing_columns)
# print(df['Salary'])

# Korelasyonların gösterilmesi
def find_correlation(dataframe, numeric_cols, target, corr_limit=0.60):
    high_correlations = []
    low_correlations = []

    for col in numeric_cols:
        if col == target:
            continue

        correlation = dataframe[[col, target]].corr().iloc[0, 1]

        if abs(correlation) > corr_limit:
            high_correlations.append(f"{col}: {correlation}")
        else:
            low_correlations.append(f"{col}: {correlation}")

    return low_correlations, high_correlations
# Örnek kullanım
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
low_corrs, high_corrs = find_correlation(df, num_cols, "Salary")
print("Yüksek Korelasyonlu Değişkenler:")
print(high_corrs)
print("\nDüşük Korelasyonlu Değişkenler:")
print(low_corrs)

####### Degisken yaratma
df["İsabet_orani"] = df["HmRun"] / df["AtBat"]

#2.degisken
df["Oyuncu_maasi"] = df["HmRun"] * df["Walks"] * df["CHmRun"] * df["PutOuts"]
df["Oyuncu_basarisi_sirasi"] = pd.qcut(df["Oyuncu_maasi"], 5,
                                          labels = ["kötü_oyuncu", "vasat_oyuncu", "ortalama_oyuncu", "iyi_oyuncu", "mükemmel_oyuncu"])
#print(df.head())
cat_cols, num_cols, cat_but_car, num_but_cat = eda.grab_col_names(df)

#print(df.isnull().sum())
###### encoding islemleri

# Label Encoder
df["Oyuncu_basarisi_sirasi"] = df["Oyuncu_basarisi_sirasi"].astype("object")
# print(df.info())

df.loc[(df["Oyuncu_basarisi_sirasi"] == "kötü_oyuncu"), "Oyuncu_basarisi_sirasi"] = 0
df.loc[(df["Oyuncu_basarisi_sirasi"] == "vasat_oyuncu"), "Oyuncu_basarisi_sirasi"] = 1
df.loc[(df["Oyuncu_basarisi_sirasi"] == "ortalama_oyuncu"), "Oyuncu_basarisi_sirasi"] = 2
df.loc[(df["Oyuncu_basarisi_sirasi"] == "iyi_oyuncu"), "Oyuncu_basarisi_sirasi"] = 3
df.loc[(df["Oyuncu_basarisi_sirasi"] == "mükemmel_oyuncu"), "Oyuncu_basarisi_sirasi"] = 4
# print(df[["Oyuncu_basarisi_sirasi"]].head(12))
# print(df.head())

#standarlastirma

df["Oyuncu_basarisi_sirasi"] = df["Oyuncu_basarisi_sirasi"].astype("int")
cat_cols, num_cols, cat_but_car, num_but_cat = eda.grab_col_names(df)


scaler = StandardScaler()
num_cols = [col for col in num_cols if col not in "Salary"]
df[num_cols] = scaler.fit_transform(df[num_cols])
print(df.head())

###makine ogrenmesi
X = df.drop("Salary", axis = 1)  # bagimsiz degisken
#print(X.head().T)
y = df[["Salary"]]   # bagimli degisken
#print(y.head().T)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 17)
print("\n X_train=",X_train, "\n X_test=",X_test, "\n y_train=",y_train, " \n y_test=",y_test )

#uzunluklari gosterelim
print(X_train.shape, X_test.shape, y_train.shape , y_test.shape)





# Doğrusal regresyon modelini oluşturma ve eğitme
# Sadece sayısal değişkenleri seçme
numeric_columns_train = X_train.select_dtypes(include=[np.number]).columns
X_train = X_train[numeric_columns_train]

numeric_columns_test = X_test.select_dtypes(include=[np.number]).columns
X_test = X_test[numeric_columns_test]

# Doğrusal regresyon modelini oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Modelin eğim (slope) ve kesişim (intercept) katsayılarını yazdırma
print("Eğim (slope):", model.coef_)
print("Kesişim (intercept):", model.intercept_)

#Tahmin
#Yeni bir veri noktası için tahmin yapma (örneğin, X_new = np.array([[6]]))
X_new = np.array(X_test)  #  AtBat değeri için tahmin yapma
y_pred = model.predict(X_new)
print("Yeni veri için tahmin: \n", y_pred)

# Modelin görselleştirilmesi
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'color': 'b', 's': 9}, ci=False, color="r")
plt.title("Gerçek ve Tahmin Edilen Değerler")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.xlim(0, None)
plt.ylim(0, None)
plt.show()

#####  Mean Squared Error (MSE) - Hata Karelerin Ortalaması
# Mean Squared Error (MSE) hesaplama
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Root Mean Square Error (RMSE) hesaplama
rmse = np.sqrt(mse)
print("Root Mean Square Error (RMSE):", rmse)

# Mean Absolute Error (MAE) hesaplama
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

# R-kare (R²) hesaplama
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R-kare (R²):", r2)


# RandomForestRegressor ile model eğitimi
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Modelin tahmin edilmesi ve değerlendirilmes
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

# Değişken Önemini Görselleştirme
def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
plot_importance(rf_model, X_train, 20)
