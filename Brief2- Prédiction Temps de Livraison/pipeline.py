# import labriries
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, mean_absolute_error ,r2_score
from test_Pipeline import test_dim ,test_format , test_mae
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder




# Cleaning dataset
def cleaning_data(df):
    """
    Nettoie le DataFrame en effectuant les opérations suivantes :
    - Suppression des doublons
    - Conversion de 'TotalCharges' en numérique
    - Remplissage des valeurs manquantes (numériques et catégorielles)
    """
    df_cleaned = df.copy()
    
    # Remove the dupilcate row
    df_cleaned = df_cleaned.drop_duplicates()
    # Check missing value 
    df.isna().sum()
    # fill missing value for numérique variable 
    df_cleaned["Courier_Experience_yrs"] = df_cleaned["Courier_Experience_yrs"] .fillna(df_cleaned["Courier_Experience_yrs"] .mean())
    # fill missing value for catégoriale variable 
    catg_cols = df_cleaned.select_dtypes(include='object').columns
    for col in catg_cols :
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0]) # la valeur le plus fréquentes 
    
    
    return df_cleaned

def split_data (df,target) :
   
   X = df.drop(columns=[target]) 
   y = df[target]
   
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

   return X_train, X_test, y_train, y_test

def choix_model(model_name) :
   
    if model_name == "SVR":
        return SVR()   
       
    elif model_name == 'RFR' :
        return RandomForestRegressor(random_state=42)
    else:
        raise ValueError("Modèle non reconnu")
    

def get_param_grid(model_name):

    if model_name == "RFR":
        return { 
            'select__k': [5, 10, 'all'],
            'regressor__n_estimators': [100, 300, 400],
            'regressor__max_depth': [None, 5, 10]
            
                 }

    elif model_name == "SVR":
        return {
          'select__k': [5, 10, 'all'],           
          'regressor__C': [0.1, 1, 10],              
          'regressor__kernel': ['linear', 'rbf']
          
                     }


    


def Entrainer_Evaluer_pipeline (X_train, X_test, y_train, y_test,model_name) :
   num_cols=['Distance_km','Preparation_Time_min']
   cat_cols=['Weather','Traffic_Level','Time_of_Day','Vehicle_Type']

   preprocessor  = ColumnTransformer(
     transformers= [ 
         ('num',StandardScaler(),num_cols),
         ('cat', OneHotEncoder(handle_unknown='ignore'),cat_cols)
     ] )
   
   # Modèle choisi
   model = choix_model(model_name) 
   pipe = Pipeline(
    steps=[
        ('preprocessor' , preprocessor),
        ('select', SelectKBest(score_func=f_regression)),
        ('regressor',model)
        ]
        )
   
   param_grid= get_param_grid(model_name)
   
   scoring  = { 'MAE':  make_scorer(mean_absolute_error,greater_is_better=False),
                'R2' : make_scorer(r2_score) 
             }
   
   #  GridSearchCV

   grid_search = GridSearchCV(
        estimator =pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit= "MAE",
        cv=5,
        n_jobs=-1,
        verbose=2
    )
   
    # 🔹 Entraînement
   grid_search.fit(X_train, y_train)



   # Évaluation 
   y_pred = grid_search.predict(X_test)
   MAE = mean_absolute_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)


   print(f" Modèle : {model_name}")
   print(f" Meilleurs paramètres : {grid_search.best_params_}")
   print(f"Meilleur score:{-grid_search.best_score_}")
   print(f" MAE : {MAE:.3f}")
   print(f"R² score : {r2:.3f}")

   resultats = {
        
        'Modèle': model_name,
        'Best_model': grid_search.best_estimator_,
        'Best_Params': grid_search.best_params_,
        'Best_Score': -grid_search.best_score_,
        'MAE': MAE,
        'R² score': r2
    }

   
   return resultats

# **************** Main ************************

if __name__ == "__main__":

 df = pd.read_csv(r"C:\Users\khadija\Desktop\simplon project\Data-IA-Simplon-Maghreb\Brief2- Prédiction Temps de Livraison\Prédiction-Temps-Livraison.csv")
 
# Netoyée le dataset 
 df_cleaned = cleaning_data(df)
# On fait un test pour les dimenssions de dataset aprés netoyage 
 test_dim(df,df_cleaned)
# on fait un test pour les formats des collones numérique mais declaré de type object en df 
 test_format(df)

 # suprimmer les colonnes inutiles ('Order_ID')
 df = df_cleaned.drop(columns= 'Order_ID') 

X_train, X_test, y_train, y_test = split_data(df,target='Delivery_Time_min')
modeles = ["SVR","RFR"]
best_mae = float('inf')
best_model = None
resultats = []
for model_name in modeles :
     print(f"\n Entrainnement et Évaluation de {model_name} ...")
     res = Entrainer_Evaluer_pipeline (X_train, X_test, y_train, y_test,model_name)
     # Test unitaire : MAE ne doit pas dépasser le seuil
     test_mae(res['MAE'] )
     resultats.append(res)
     for res in resultats:
       if res['MAE'] < best_mae :
        best_mae = res['MAE']
        best_model = res['Best_model']
        best_name = model_name
print(f"Meilleur modèle : {best_name} avec MAE = {best_mae:.3f}")       
    


    
    



