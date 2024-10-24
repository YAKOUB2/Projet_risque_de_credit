import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from category_encoders import CountEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFECV
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from jenkspy import JenksNaturalBreaks
from sklearn.utils import resample


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.dataframe = None

    def load_data(self):
        self.dataframe = pd.read_csv(self.filepath, delimiter=",", index_col=0)
        return self.dataframe
    
class Imputation:
    def __init__(self, df):
        self.df = df

    def remove_columns(self, columns):
        self.df.drop(columns=columns, inplace=True)

    def impute_columns(self, strategy, columns):
        imputer = SimpleImputer(strategy=strategy)
        self.df[columns] = imputer.fit_transform(self.df[columns])

    def custom_imputation(self, imputations):
        for column, value in imputations.items():
            self.df[column].fillna(value, inplace=True)

    def impute_own_car_age(self):
        self.df.loc[(self.df['FLAG_OWN_CAR'] == 'N') & (self.df['OWN_CAR_AGE'].isnull()), 'OWN_CAR_AGE'] = -1
        median_own_car_age = self.df[self.df['FLAG_OWN_CAR'] == 'Y']['OWN_CAR_AGE'].median()
        self.df.loc[(self.df['FLAG_OWN_CAR'] == 'Y') & (self.df['OWN_CAR_AGE'].isnull()), 'OWN_CAR_AGE'] = median_own_car_age

    def create_obs_cnt_social_circle(self):
        self.df['OBS_CNT_SOCIAL_CIRCLE'] = 0
        condition = (self.df['OBS_30_CNT_SOCIAL_CIRCLE'] > 0) | (self.df['OBS_60_CNT_SOCIAL_CIRCLE'] > 0)
        self.df.loc[condition, 'OBS_CNT_SOCIAL_CIRCLE'] = 1

    def impute_with_random_forest(self, column, features_columns):
        # Séparez le DataFrame en ensembles d'entraînement et de test basés sur la présence de valeurs manquantes dans la colonne cible
        train_df = self.df.dropna(subset=[column])
        test_df = self.df[self.df[column].isnull()]

        X_train = train_df[features_columns]
        y_train = train_df[column]
        X_test = test_df[features_columns]

        # Entraînement du modèle Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Prédiction des valeurs manquantes
        predictions = rf.predict(X_test)

        # Mise à jour de la colonne cible dans le DataFrame original
        self.df.loc[self.df[column].isnull(), column] = predictions

    def remove_rows_and_reset_index(self, condition_column, condition_value):
        # Suppression des lignes qui satisfont la condition
        self.df = self.df[self.df[condition_column] != condition_value]
        # Réinitialisation de l'index
        self.df.reset_index(drop=True, inplace=True)
        
    def execute_all_imputations(self, imputations_info):
        for method, details in imputations_info.items():
            if method == 'remove':
                self.remove_columns(details)
            elif method == 'simple':
                for strategy, columns in details.items():
                    self.impute_columns(strategy, columns)
            elif method == 'custom':
                self.custom_imputation(details)
            elif method == 'impute_own_car_age':
                self.impute_own_car_age()
            elif method == 'random_forest':
                # Ici, nous attendons que details soit un dictionnaire avec une clé 'column' pour la colonne cible
                # et une clé 'features_columns' pour les colonnes à utiliser comme features
                self.impute_with_random_forest(details['column'], details['features_columns'])
        self.create_obs_cnt_social_circle()
        self.remove_rows_and_reset_index('CODE_GENDER', 'XNA')

class Encodage:
    def __init__(self, df):
        self.df = df

    def encode(self, mappings):
        for column, mapping in mappings.items():
            self.df[column] = self.df[column].replace(mapping)

class TrainTestSplit:
    def __init__(self, df):
        self.df = df

    def split_data(self, feature_columns, target_column, date_column, validation_year, test_size=0.2, random_state=42):
        # Conversion de la colonne date_column en datetime si ce n'est pas déjà le cas
        self.df[date_column] = pd.to_datetime(self.df[date_column], errors='coerce')
        # Sélection des lignes pour l'ensemble de validation basé sur l'année
        df_validation = self.df[self.df[date_column].dt.year == validation_year]
        X_validation = df_validation[feature_columns]
        y_validation = df_validation[target_column]
        
        # Sélection des données pour les ensembles d'entraînement et de test
        df_train_test = self.df[self.df[date_column].dt.year != validation_year]
        X = df_train_test[feature_columns]
        y = df_train_test[target_column]

        # Division en ensemble d'entraînement et ensemble de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        return X_train, X_test, X_validation, y_train, y_test, y_validation


class ChiMergeDiscretizer:
    def __init__(self, max_intervals=5):
        self.max_intervals = max_intervals
        self.intervals_dict = {}
        self.mapping_dict = {}

    def chimerge(self, data, attr, label):
        distinct_vals = sorted(set(data[attr]))
        labels = sorted(set(data[label]))
        empty_count = {l: 0 for l in labels}

        deciles = np.arange(0, 110, 10)
        decile_values = np.percentile(distinct_vals, deciles)
        intervals = [[decile_values[i], decile_values[i + 1]] for i in range(len(decile_values) - 1)]

        while len(intervals) > self.max_intervals:
            chi = []
            for i in range(len(intervals) - 1):
                obs0 = data[data[attr].between(intervals[i][0], intervals[i][1])]
                obs1 = data[data[attr].between(intervals[i + 1][0], intervals[i + 1][1])]
                total = len(obs0) + len(obs1)
                count_0 = np.array([v for i, v in {**empty_count, **Counter(obs0[label])}.items()])
                count_1 = np.array([v for i, v in {**empty_count, **Counter(obs1[label])}.items()])
                count_total = count_0 + count_1
                expected_0 = count_total * sum(count_0) / total
                expected_1 = count_total * sum(count_1) / total
                chi_ = (count_0 - expected_0) ** 2 / expected_0 + (count_1 - expected_1) ** 2 / expected_1
                chi_ = np.nan_to_num(chi_)
                chi.append(sum(chi_))
            min_chi = min(chi)
            for i, v in enumerate(chi):
                if v == min_chi:
                    min_chi_index = i
                    break
            new_intervals = []
            skip = False
            done = False
            for i in range(len(intervals)):
                if skip:
                    skip = False
                    continue
                if i == min_chi_index and not done:
                    t = intervals[i] + intervals[i + 1]
                    new_intervals.append([min(t), max(t)])
                    skip = True
                    done = True
                else:
                    new_intervals.append(intervals[i])
            intervals = new_intervals

        return intervals

    def fit(self, data, cols_to_discretize, label):
        for col in cols_to_discretize:
            intervals = self.chimerge(data, col, label)
            self.intervals_dict[col] = intervals

            mapping = {i: f'{interval[0]} - {interval[1]}' for i, interval in enumerate(intervals)}
            self.mapping_dict[col] = mapping

    def transform(self, data):
        for col, intervals in self.intervals_dict.items():
            data[col+'_chimerge'] = 0
            for i, interval in enumerate(intervals):
                mask = (data[col] >= interval[0]) & (data[col] <= interval[1])
                data.loc[mask, col+'_chimerge'] = i

        return data

    def get_mapping_dict(self):
        return self.mapping_dict


class Regroupement:
    def __init__(self, df):
        self.df = df

    def apply_regroupements(self):
        self.df['NAME_INCOME_TYPE'] = self.df['NAME_INCOME_TYPE'].apply(self.regroup_NAME_INCOME_TYPE)
        self.df['NAME_EDUCATION_TYPE'] = self.df['NAME_EDUCATION_TYPE'].apply(self.regroup_NAME_EDUCATION_TYPE)
        self.df['ORGANIZATION_TYPE'] = self.df['ORGANIZATION_TYPE'].apply(self.regroup_ORGANIZATION_TYPE)
        self.df['NAME_HOUSING_TYPE'] = self.df['NAME_HOUSING_TYPE'].apply(self.regroup_NAME_HOUSING_TYPE)
        self.df['NAME_FAMILY_STATUS'] = self.df['NAME_FAMILY_STATUS'].apply(self.regroup_NAME_FAMILY_STATUS)
        self.df['OCCUPATION_TYPE'] = self.df['OCCUPATION_TYPE'].apply(self.regroup_OCCUPATION_TYPE)

    def regroup_NAME_INCOME_TYPE(self, mod):
        if mod in ['Working', 'Commercial associate', 'Maternity leave', 'Businessman']:
            return 'Actif'
        elif mod in ['Pensioner', 'State servant', 'Student', 'Unemployed']:
            return 'Helped/state rel'
        else:
            return mod

    def regroup_NAME_EDUCATION_TYPE(self, mod):
        if mod in ['Academic degree', 'Higher education']:
            return 'Études sup'
        elif mod in ['Incomplete higher', 'Lower secondary', 'Secondary / secondary special']:
            return 'Éducation secondaire'
        else:
            return mod

    def regroup_ORGANIZATION_TYPE(self, mod):
        if mod in ['University', 'Industry: type 12', 'Trade: type 4', 'Bank', 'Electricity', 'Insurance', 'Military', 'Police', 'Security Ministries', 'Transport: type 1', 'Culture', 'Emergency', 'Government', 'Hotel', 'Legal Services', 'Medicine', 'Religion', 'School', 'Telecom', 'Kindergarten', 'Industry: type 11', 'Postal', 'Services', 'Other']:
            return 'Faible Risque'
        else:
            return 'Risque Élevé'

    def regroup_NAME_HOUSING_TYPE(self, mod):
        if mod == 'House / apartment':
            return 'Propriétaire'
        elif mod in ['With parents', 'Rented apartment','Co-op apartment', 'Municipal apartment', 'Office apartment']:
            return 'Hébergé et Louer (?)'
        else:
            return 'Non Classifié'

    def regroup_NAME_FAMILY_STATUS(self, mod):
        if mod in ['Married', 'Widow', 'Unknown']:
            return 'Marié ou veuf'
        elif mod in ['Civil marriage', 'Single / not married', 'Separated']:
            return 'Mariage civil et célibataire'
        else:
            return mod

    def regroup_OCCUPATION_TYPE(self, occupation):
        if occupation in ['Cleaning staff', 'Cooking staff', 'Drivers', 'Laborers', 'Low-skill Laborers', 'Security staff', 'Waiters/barmen staff']:
            return 'Métier peu qualifié'
        elif occupation in ['Accountants', 'Core staff', 'HR staff', 'High skill tech staff', 'IT staff', 'Managers', 'Medicine staff', 'Private service staff', 'Realty agents', 'Sales staff', 'Secretaries', 'non_dentifie']:
            return 'Corporate'
        else:
            return occupation


class ModeleLogistique:
    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train):
        # Ajouter une colonne constante pour l'intercept
        X_train_sm = sm.add_constant(X_train)
        # Ajuster le modèle
        self.model = sm.Logit(y_train, X_train_sm).fit(disp=0)  # disp=0 désactive la sortie pendant le fitting

    def predict_and_evaluate(self, X, y):
        # Préparation des données pour la prédiction
        X_sm = sm.add_constant(X, has_constant='add')
        # Prédictions de probabilité
        y_pred_probs = self.model.predict(X_sm)
        # Calcul de l'AUC
        auc = roc_auc_score(y, y_pred_probs)
        return y_pred_probs, auc

    def summary(self):
        # Afficher un résumé du modèle
        return self.model.summary()




class GrilleDeScore:
    def __init__(self, df, model):
        self.df = df
        self.model = model

    def liste_coefs(self, variable):
        modalites = self.model.params.index.drop("const")
        modalites_de_la_variable = [var for var in modalites if var.startswith(variable)]
        return self.model.params[modalites_de_la_variable]

    def calcul_note_modalite(self, variable, modalite):
        modalites = self.model.params.index.drop("const")
        liste_variable = list(set("_".join(mod.split("_")[:-1]) for mod in modalites))
        coefs_variable = self.liste_coefs(variable)
        coef_modalite = self.model.params[modalite]
        
        max_coef = max(coefs_variable)    
        
        numerateur = abs(max_coef - coef_modalite)

        denominateur = 0
        for var in liste_variable:
            coefs_var = self.liste_coefs(var)
            min_cf = min(coefs_var)
            max_cf = max(coefs_var)
            denominateur += max_cf - min_cf

        note = (numerateur / denominateur) * 1000
        return note

    def liste_modalite_variable(self, variable):
        modalites = self.model.params.index.drop("const")
        return [var for var in modalites if var.startswith(variable)]

    def part_population_mod(self, modalite):
        total_population = len(self.df)
        
        if modalite in self.df.columns:
            counts = self.df[modalite].sum()
            part = counts / total_population
        else:
            part = 0  # Si la modalité n'est pas trouvée, renvoyer 0
        
        return part

    def calcul_contribution_variable(self, variable):
        modalites = self.liste_modalite_variable(variable)
        note_moyenne = np.mean([self.calcul_note_modalite(variable, modalite) for modalite in modalites])

        numerateur = 0
        for modalite in modalites:
            r = self.part_population_mod(modalite)
            numerateur += r * ((self.calcul_note_modalite(variable, modalite) - note_moyenne) ** 2)
        numerateur = np.sqrt(numerateur)

        denominateur = 0
        terme = 0
        for var in list(set("_".join(modal.split("_")[:-1]) for modal in self.model.params.index.drop("const"))):
            note_moyenne_var = np.mean([self.calcul_note_modalite(var, modal2) for modal2 in self.liste_modalite_variable(var)])

            for modalite2 in self.liste_modalite_variable(var):
                r = self.part_population_mod(modalite2)
                terme += r * ((self.calcul_note_modalite(var, modalite2) - note_moyenne_var) ** 2)
            terme = np.sqrt(terme)
            
            denominateur += terme

        contribution = numerateur / denominateur
        
        return contribution

    def pvalue_modalite(self, modalite):
        return self.model.pvalues[modalite]

    def taux_de_defaut_modalite(self, modalite):
        return self.df[self.df[modalite] == 1]["TARGET"].mean()

    def effectif_modalite(self, modalite):
        return self.df[modalite].mean()

    def generer_grille(self):
        liste_variables = list(set("_".join(mod.split("_")[:-1]) for mod in self.model.params.index.drop("const")))
        liste_dico = []
        for variable in liste_variables:
            liste_modalites_var = self.liste_modalite_variable(variable)
            for modalite in liste_modalites_var:
                dico = {
                    "Variable": variable,
                    "Classe": modalite.rsplit('_', 1)[1] if len(modalite.rsplit('_', 1)) > 1 else '',
                    "p-value/Significativité": self.pvalue_modalite(modalite),
                    "Note": self.calcul_note_modalite(variable, modalite),
                    "Contribution": self.calcul_contribution_variable(variable),
                    "Taux de défaut en %": self.taux_de_defaut_modalite(modalite),
                    "Effectif de chaque classe en %": self.effectif_modalite(modalite)
                }
                liste_dico.append(dico)

        return pd.DataFrame(liste_dico)


class LRA_MOC:
    def __init__(self, df, segment_col='Segment', target_col='TARGET', year_col='annee', n_bootstrap=1000, seed=42):
        self.df = df
        self.segment_col = segment_col
        self.target_col = target_col
        self.year_col = year_col
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def calculate_lra(self):
        sum_by_segment_year = self.df.groupby([self.segment_col, self.year_col])[self.target_col].mean().reset_index().groupby(self.segment_col)[self.target_col].sum()
        LRA = sum_by_segment_year / 7
        self.df['LRA'] = self.df[self.segment_col].map(LRA)

    def calculate_moc(self):
        np.random.seed(self.seed)
        lra_bootstrap = {segment: [] for segment in self.df[self.segment_col].unique()}
        
        for _ in range(self.n_bootstrap):
            bootstrap_sample = resample(self.df, replace=True)
            lra_sample = bootstrap_sample.groupby([self.segment_col, self.year_col])[self.target_col].mean().reset_index().groupby(self.segment_col)[self.target_col].sum() / 7
            for segment, lra in lra_sample.items():
                lra_bootstrap[segment].append(lra)
                
        moc = {}
        for segment, lras in lra_bootstrap.items():
            percentile_90 = np.percentile(lras, 90)
            mean_lra = np.mean(lras)
            moc[segment] = percentile_90 - mean_lra
            
        moc_series = pd.Series(moc)
        self.df['MoC C'] = self.df[self.segment_col].map(moc_series)

    def calculate_pd(self):
        taux_defaut_total = self.df.groupby(self.segment_col)[self.target_col].mean()
        df_avant_2019 = self.df[self.df[self.year_col] < 2019]
        taux_defaut_avant_2019 = df_avant_2019.groupby(self.segment_col)[self.target_col].mean()
        taux_variation = (taux_defaut_avant_2019 / taux_defaut_total) - 1
        self.df['MoC A'] = self.df[self.segment_col].map(taux_variation)
        self.df['PD'] = self.df['LRA'] + self.df['MoC C'] + self.df['MoC A']

    def apply_all(self):
        self.calculate_lra()
        self.calculate_moc()
        self.calculate_pd()
        return self.df[['note_totale', 'LRA', 'MoC C', 'MoC A', 'PD']]



def main():

    # Chemin vers le fichier de données
    filepath = 'application_train_vf.csv'
    print("Chargement des données...")

    # Initialisation et chargement des données
    data_loader = DataLoader(filepath)
    df = data_loader.load_data()

    print("Données chargées avec succès.")
    ###########################
    #########Imputation########
    ###########################

    columns_to_delete = [
    'COMMONAREA_MEDI', 'COMMONAREA_AVG', 'COMMONAREA_MODE',
    'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_MODE',
    'FONDKAPREMONT_MODE',
    'LIVINGAPARTMENTS_MODE', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAPARTMENTS_AVG',
    'FLOORSMIN_MODE', 'FLOORSMIN_MEDI', 'FLOORSMIN_AVG',
    'YEARS_BUILD_MODE', 'YEARS_BUILD_MEDI', 'YEARS_BUILD_AVG',
    'LANDAREA_AVG', 'LANDAREA_MEDI', 'LANDAREA_MODE',
    'BASEMENTAREA_MEDI', 'BASEMENTAREA_AVG', 'BASEMENTAREA_MODE',
    'EXT_SOURCE_1',
    'NONLIVINGAREA_MEDI',
    'ELEVATORS_MODE', 'ELEVATORS_AVG',
    'APARTMENTS_MODE', 'APARTMENTS_MEDI', 'APARTMENTS_AVG',
    'ENTRANCES_MODE', 'ENTRANCES_AVG', 'ENTRANCES_MEDI',
    'HOUSETYPE_MODE',
    'FLOORSMAX_MEDI', 'FLOORSMAX_AVG', 'FLOORSMAX_MODE',
    'EXT_SOURCE_3',
    'NAME_TYPE_SUITE','NONLIVINGAREA_MODE', 'LIVINGAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MODE', 
    'LIVINGAREA_MODE', 'YEARS_BEGINEXPLUATATION_MEDI',
    "FLAG_DOCUMENT_2","FLAG_DOCUMENT_3","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_6","FLAG_DOCUMENT_7","FLAG_DOCUMENT_8",
    "FLAG_DOCUMENT_9","FLAG_DOCUMENT_10","FLAG_DOCUMENT_11","FLAG_DOCUMENT_12","FLAG_DOCUMENT_13","FLAG_DOCUMENT_14","FLAG_DOCUMENT_15",
    "FLAG_DOCUMENT_16","FLAG_DOCUMENT_17","FLAG_DOCUMENT_18","FLAG_DOCUMENT_19","FLAG_DOCUMENT_20","FLAG_DOCUMENT_21",
    'YEARS_BEGINEXPLUATATION_AVG', 'NONLIVINGAPARTMENTS_AVG', 'ELEVATORS_MEDI', 'WALLSMATERIAL_MODE','FLAG_MOBIL', 'FLAG_EMP_PHONE', 
    'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL','DAYS_LAST_PHONE_CHANGE'
    ]

    colonnes_a_imputer_par_mediane = ['NONLIVINGAREA_AVG', 'LIVINGAREA_AVG', 
                                      'TOTALAREA_MODE','EXT_SOURCE_2',
                                      'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 
                                      'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 
                                      'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
    
    colonnes_a_imputer_mode = ['EMERGENCYSTATE_MODE','OBS_30_CNT_SOCIAL_CIRCLE', 
                               'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']

    imputations_info = {
    'remove': columns_to_delete,
    'simple': {
        'median': colonnes_a_imputer_par_mediane,
        'most_frequent': colonnes_a_imputer_mode
    },
    'custom': {
        'OCCUPATION_TYPE': 'non_dentifie',
        'AMT_GOODS_PRICE': 0,
        'CNT_FAM_MEMBERS': 0
    },
    'random_forest': {
        'column': 'AMT_ANNUITY',
        'features_columns': ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'EXT_SOURCE_2', 'REGION_RATING_CLIENT_W_CITY', 'OWN_CAR_AGE', 'REGION_RATING_CLIENT']  # Ajoutez les colonnes pertinentes
    },                      
    'impute_own_car_age': True  # Ajoutez simplement un indicateur pour cette imputation spécifique
    }

    imputer = Imputation(df)
    imputer.execute_all_imputations(imputations_info)

    ###########################
    #########Encodage##########
    ###########################

    mappings = {
        'CODE_GENDER': {'F': 0, 'M': 1},
        'FLAG_OWN_CAR': {'N': 0, 'Y': 1},
        'FLAG_OWN_REALTY': {'N': 0, 'Y': 1},
        'EMERGENCYSTATE_MODE': {'No': 0, 'Yes': 1},
        'WEEKDAY_APPR_PROCESS_START': {
            'MONDAY': 1,
            'TUESDAY': 2,
            'WEDNESDAY': 3,
            'THURSDAY': 4,
            'FRIDAY': 5,
            'SATURDAY': 6,
            'SUNDAY': 7
        }
    }

    encoder = Encodage(df)
    encoder.encode(mappings)

    
    ###########################
    #####Train Test Split######
    ###########################

    # Variables catégorielles à conserver
    categorical_vars = ['REGION_RATING_CLIENT', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY'
                    ,'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'OBS_CNT_SOCIAL_CIRCLE', 'CODE_GENDER', 'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                    'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE','ORGANIZATION_TYPE']

    # Variables quantitatives à conserver
    quantitative_vars = [
        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OWN_CAR_AGE', 'CNT_FAM_MEMBERS',
        'LIVINGAREA_AVG', 'NONLIVINGAREA_AVG', 'TOTALAREA_MODE',
        'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR','EXT_SOURCE_2'
    ]

    splitter = TrainTestSplit(df)

    # Sélection des colonnes de features et de la colonne cible
    feature_columns = categorical_vars + quantitative_vars  # Supposons que ces listes contiennent les noms de vos colonnes
    target_column = 'TARGET'
    date_column = 'date_mensuelle'
    validation_year = 2020

    # Appel de la méthode pour diviser les données
    X_train, X_test, X_validation, y_train, y_test, y_validation = splitter.split_data(feature_columns, target_column, date_column, validation_year)

    ###########################
    #########Chi Merge#########
    ###########################

    df_train_y = X_train.merge(y_train, left_index=True, right_index=True)
    df_test_y = X_test.merge(y_test, left_index=True, right_index=True)
    df_validation_y = X_validation.merge(y_validation, left_index=True, right_index=True)

    df_train_y['DAYS_BIRTH'] = round(-X_train['DAYS_BIRTH']/365.25)
    df_test_y['DAYS_BIRTH'] = round(-X_train['DAYS_BIRTH']/365.25)
    X_validation['DAYS_BIRTH'] = round(-X_train['DAYS_BIRTH']/365.25)

    #DAYS_EMPLOYED

    df_train_y['DAYS_EMPLOYED'] = round(-X_train['DAYS_EMPLOYED']/365.25)
    df_test_y['DAYS_EMPLOYED'] = round(-X_train['DAYS_EMPLOYED']/365.25)
    X_validation['DAYS_EMPLOYED'] = round(-X_train['DAYS_EMPLOYED']/365.25)


    quanti_vars_2 = ['CNT_CHILDREN', 'OWN_CAR_AGE', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED']
    quanti_vars_3 = ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    quanti_vars_4 = ['DAYS_BIRTH', 'EXT_SOURCE_2']

    # Créer un dictionnaire pour stocker les résultats
    df_discretized_dict = {}

    for max_intervals, quanti_vars in [(2, quanti_vars_2), (3, quanti_vars_3), (4, quanti_vars_4)]:

        # Create an instance of the ChiMergeDiscretizer
        discretizer = ChiMergeDiscretizer(max_intervals=max_intervals)

        # Fit on df_train_y
        discretizer.fit(df_train_y, quanti_vars, 'TARGET')

        # Transform df_train_y
        df_train_y_discretized = discretizer.transform(df_train_y)

        # Transform df_test_y
        df_test_y_discretized = discretizer.transform(df_test_y)

        # Transform df_validation_y
        df_validation_y_discretized = discretizer.transform(df_validation_y)

        # Utiliser get_mapping_dict pour obtenir le dictionnaire de correspondance
        mapping_dict = discretizer.get_mapping_dict()

        for column in quanti_vars :
            # Accéder au mapping pour une colonne donnée
            df_discretized_dict[column] = mapping_dict[column]

    regroup_train = Regroupement(df_train_y_discretized)
    regroup_test = Regroupement(df_test_y_discretized)
    regroup_validation = Regroupement(df_validation_y_discretized)

    # Application des regroupements
    regroup_train.apply_regroupements()
    regroup_test.apply_regroupements()
    regroup_validation.apply_regroupements()




    variable_quali_list_new = ['REGION_RATING_CLIENT', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'OBS_CNT_SOCIAL_CIRCLE', 'CODE_GENDER', 'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']

    target = ['TARGET']

    variable_quanti_list_new = ['CNT_CHILDREN_chimerge',
        'OWN_CAR_AGE_chimerge', 'AMT_INCOME_TOTAL_chimerge',
        'AMT_CREDIT_chimerge', 'AMT_ANNUITY_chimerge',
        'AMT_GOODS_PRICE_chimerge', 'DAYS_BIRTH_chimerge', 'DAYS_EMPLOYED_chimerge', 'EXT_SOURCE_2_chimerge']

    df_train_y_discretized = df_train_y_discretized[variable_quanti_list_new + variable_quali_list_new + target]    


    combined_variable_list = variable_quali_list_new + variable_quanti_list_new

    def one_hot_encode_with_reference(df, columns, target_column):
        encoded_df = df.copy()
        reference_categories = {}

        for column in columns:
            # Calculer le pourcentage de défaut pour chaque modalité
            target_rate_by_category = df.groupby(column)[target_column].mean()
            # Identifier la modalité avec le taux de défaut le plus faible
            reference_category = target_rate_by_category.idxmin()
            reference_categories[column] = reference_category
            
            # Réaliser un encodage One-Hot
            dummies = pd.get_dummies(df[column], prefix=column, drop_first=False)
            
            # Optionnel : Supprimer la colonne de la modalité de référence si souhaité
            dummies.drop(f"{column}_{reference_category}", axis=1, inplace=True)
            
            # Concaténer avec le DataFrame encodé
            encoded_df = pd.concat([encoded_df.drop(column, axis=1), dummies], axis=1)

        return encoded_df, reference_categories

    # Ajouter une colonne pour distinguer les ensembles
    df_train_y_discretized['is_train'] = 1
    df_test_y_discretized['is_train'] = 0
    df_validation_y_discretized['is_train'] = 2

    # Fusionner les dataframes
    df_combined = pd.concat([df_train_y_discretized, df_test_y_discretized, df_validation_y_discretized])

    # Appliquer l'encodage
    df_combined_encoded, reference_categories = one_hot_encode_with_reference(df_combined, combined_variable_list, 'TARGET')

    # Séparer les dataframes
    df_train_encoded = df_combined_encoded[df_combined_encoded['is_train'] == 1].drop('is_train', axis=1)
    df_test_encoded = df_combined_encoded[df_combined_encoded['is_train'] == 0].drop('is_train', axis=1)
    df_validation_encoded = df_combined_encoded[df_combined_encoded['is_train'] == 2].drop('is_train', axis=1)


    # Remplacer True par 1 et False par 0 dans tout le DataFrame
    df_train_encoded.replace({True: 1, False: 0}, inplace=True)
    df_test_encoded.replace({True: 1, False: 0}, inplace=True)
    df_validation_encoded.replace({True: 1, False: 0}, inplace=True)

    # Supprimer toutes les colonnes où il y a des NaN dans df_train_encoded
    df_train_encoded.dropna(axis=1, inplace=True)

    # Supprimer toutes les colonnes où il y a des NaN dans df_test_encoded
    df_test_encoded.dropna(axis=1, inplace=True)

    df_validation_encoded.dropna(axis=1, inplace=True)

    y_train = df_train_encoded['TARGET']
    X_train = df_train_encoded.drop(columns=['TARGET'])
    y_test = df_test_encoded['TARGET']
    X_test = df_test_encoded.drop(columns=['TARGET'])

    y_validation = df_validation_encoded['TARGET']
    X_validation = df_validation_encoded.drop(columns=['TARGET'])

    selected_variables2=['DAYS_BIRTH_chimerge_0','DAYS_BIRTH_chimerge_1','DAYS_BIRTH_chimerge_2','AMT_INCOME_TOTAL_chimerge_0' , 'AMT_ANNUITY_chimerge_0', 'AMT_ANNUITY_chimerge_1','CODE_GENDER_1', 'EXT_SOURCE_2_chimerge_0', 'EXT_SOURCE_2_chimerge_1', 'EXT_SOURCE_2_chimerge_2', 'NAME_CONTRACT_TYPE_Cash loans']


    X_train = X_train[selected_variables2]
    X_test = X_test[selected_variables2]
    X_validation = X_validation[selected_variables2]  

    ###########################
    ##########Modele###########
    ###########################

    # Initialisation du modèle
    modele = ModeleLogistique()

    # Entraînement du modèle
    modele.fit(X_train, y_train)

    # Prédiction et évaluation sur l'ensemble de test
    y_pred_test, auc_test = modele.predict_and_evaluate(X_test, y_test)
    print(f"AUC sur l'ensemble de test: {auc_test}")

    # Prédiction et évaluation sur l'ensemble de validation
    y_pred_validation, auc_validation = modele.predict_and_evaluate(X_validation, y_validation)
    print(f"AUC sur l'ensemble de validation: {auc_validation}")

    # Affichage du résumé du modèle
    print(modele.summary())


    ###########################
    ######Grille de score######
    ###########################

    df_grille = X_train.copy()
    df_grille['TARGET'] = y_train.copy()
    grille_score = GrilleDeScore(df_grille,modele.model)
    grille_de_score_df = grille_score.generer_grille()

    #On calcul les notes et les autres colonnes pour la modalités de réference qui n'est pas dispo dans les sorties du modèle logit !

    new_rows = pd.DataFrame({
        'Variable': ['NAME_CONTRACT_TYPE', 'AMT_INCOME_TOTAL_chimerge', 'CODE_GENDER', 'DAYS_BIRTH_chimerge', 'AMT_ANNUITY_chimerge', 'EXT_SOURCE_2_chimerge'],
        'Classe': ['Revolving loans', '1', '0', '3', '2', '3'],
        'p-value/Significativité': [0, 0, 0, 0, 0, 0],
        'Note': [(0.4127/1.7127132352245684)*1000, (0.1472/1.7127132352245684)*1000, (0.3482/1.7127132352245684)*1000, (0.7040/1.7127132352245684)*1000, (0.3057/1.7127132352245684)*1000, (1.7127132352245684/1.7127132352245684)*1000],
        'Contribution': [0, 0, 0, 0.192348, 0.160389, 0.645135],
        'Taux de défaut en %': [
            df_train_y['TARGET'][df_train_y['NAME_CONTRACT_TYPE'] == 'Revolving loans'].mean(),
            df_train_y['TARGET'][df_train_y['AMT_INCOME_TOTAL_chimerge'] == 1].mean(),
            df_train_y['TARGET'][df_train_y['CODE_GENDER'] == 'F'].mean(),
            df_train_y['TARGET'][df_train_y['DAYS_BIRTH_chimerge'] == 3].mean(),
            df_train_y['TARGET'][df_train_y['AMT_ANNUITY_chimerge'] == 2].mean(),
            df_train_y['TARGET'][df_train_y['EXT_SOURCE_2_chimerge'] == 3].mean()
        ],
        'Effectif de chaque classe en %': [1-0.904721, 1-0.770046, 1-0.341779, 1-(0.159194+0.268503+0.336550), 1-(0.230159+0.673843), 1-(0.053630+0.241014+ 0.354632)]
    })

    grille_de_score_df = pd.concat([grille_de_score_df, new_rows])

    # Tri par ordre alphabétique de 'Variable'
    grille_de_score_df.sort_values(by='Variable', inplace=True)

    print('\n\n Grille de Score: \n')
    print(grille_de_score_df)


    variables = grille_de_score_df['Variable'].unique().tolist()



    notes_dict = {}
    for index, row in grille_de_score_df.iterrows():
        notes_dict[(row['Variable'], row['Classe'])] = row['Note'] 

    def calcul_note(row):
        note_totale = 0
        for var in variables:
            
            modalite_active = [col.split('_')[-1] for col in row.index if col.startswith(var + "_") and row[col] == 1]
            if modalite_active:
                
                modalite = modalite_active[0]
                if (var, modalite) in notes_dict:
                    note_totale += notes_dict.get((var, modalite), 0)
        return note_totale



    X_train['note_totale'] = X_train.apply(calcul_note, axis=1)


    

    breaks = JenksNaturalBreaks(n_classes=6)
    breaks.fit(X_train['note_totale'].values)
    X_train['Segment']=breaks.predict(X_train['note_totale'].values)


    ###########################
    ########LRA/MoC/PD#########
    ###########################

    X_train['annee'] = df['date_mensuelle'].dt.year  
    X_train["TARGET"] = y_train
    
    lra_moc_calculator = LRA_MOC(X_train)
    Grille_LRA_MOC = lra_moc_calculator.apply_all()

    print(Grille_LRA_MOC.head())



if __name__ == "__main__":
    main()
















