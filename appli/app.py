from flask import Flask, request, render_template, redirect, url_for, jsonify
from datetime import datetime
import pickle

# Importation de votre classe Preprocessing
from modele.Preprocess_Infos_Flask.Classe_Preprocessing import Preprocessing

app = Flask(__name__)

# Charger le modèle
def load_model():
    print("Chargement du modèle")
    with open('appli/modele/modele_logistique_risque_de_credit.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Modèle chargé")
    return model

model = load_model()  # Charger le modèle lors du démarrage de l'app

@app.route('/')
@app.route('/form.html')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Configuration des intervalles pour chaque variable
    config = {
        'AMT_INCOME_TOTAL': [(0, 265572.9),(265572.9, 18000090.0)],
        'AMT_ANNUITY': [(0, 27618),(27618, 46450),(46450, 258025.5)],
        'age': [(0, 25.8),(25.8, 30.6),(30.6, 64.2),(64.2, 90)],
        'EXT_SOURCE_2': [(0, 0.1424),(0.1424, 0.6812),(0.6812, 0.7322),(0.7322, 1)]
    }

    # Instanciation du préprocesseur
    preprocessor = Preprocessing(config)

    # Récupération des données du formulaire
    birthdate = request.form['birthdate']
    income = float(request.form['AMT_INCOME_TOTAL'])
    annuity = float(request.form['AMT_ANNUITY'])
    ext_source_2 = float(request.form['EXT_SOURCE_2'])
    sexe = request.form['sexe']
    type_contrat = request.form['type_contrat']

    # Application des mappings
    sexe_mapping = {'Homme': 1,'Femme': 0}
    type_contrat_mapping = {'Cash loans': 0,'Revolving loans': 1}
    sexe = sexe_mapping[sexe]
    type_contrat = type_contrat_mapping[type_contrat]

    # Calcul de l'âge
    birthdate = datetime.strptime(birthdate, '%Y-%m-%d')
    today = datetime.today()
    age = (today - birthdate).days / 365.25

    # Préparer les données pour le prétraitement
    input_data = {
        'age': age,
        'AMT_INCOME_TOTAL': income,
        'AMT_ANNUITY': annuity,
        'EXT_SOURCE_2': ext_source_2
    }

    # Utilisation de la classe Preprocessing
    processed_data = preprocessor.preprocess(input_data)
    processed_data["sexe"] = sexe
    processed_data["type_contrat"] = type_contrat

    # Mapper les données pour le modèle
    model_input = {
        'const': 1,  # Généralement, une constante pour l'intercept
        'DAYS_BIRTH_chimerge_0': 1 if processed_data['age'] == 0 else 0,
        'DAYS_BIRTH_chimerge_1': 1 if processed_data['age'] == 1 else 0,
        'DAYS_BIRTH_chimerge_2': 1 if processed_data['age'] == 2 else 0,
        'AMT_INCOME_TOTAL_chimerge_0': 1 if processed_data['AMT_INCOME_TOTAL'] == 0 else 0,
        'AMT_ANNUITY_chimerge_0': 1 if processed_data['AMT_ANNUITY'] == 0 else 0,
        'AMT_ANNUITY_chimerge_1': 1 if processed_data['AMT_ANNUITY'] == 1 else 0,
        'CODE_GENDER_1': 1 if processed_data['sexe'] == 1 else 0,
        'EXT_SOURCE_2_chimerge_0': 1 if processed_data['EXT_SOURCE_2'] == 0 else 0,
        'EXT_SOURCE_2_chimerge_1': 1 if processed_data['EXT_SOURCE_2'] == 1 else 0,
        'EXT_SOURCE_2_chimerge_2': 1 if processed_data['EXT_SOURCE_2'] == 2 else 0,
        'NAME_CONTRACT_TYPE_Cash loans': 1 if processed_data['type_contrat'] == 0 else 0
    }
    
    # Convertir les données en format attendu par le modèle
    model_input_list = [model_input[key] for key in sorted(model_input.keys())]
    
    # Faire la prédiction (obtenir directement les probabilités)
    probability = model.predict(model_input_list)

    # Appliquer un seuil pour convertir la probabilité en prédiction de classe
    threshold = 0.5
    prediction = 1 if probability >= threshold else 0

    # Retourner la prédiction et la probabilité
    return jsonify({'probability': float(1 - probability), 'prediction': 1 - prediction})

@app.route('/bouton1') 
def bouton1():
    return redirect(url_for('page2'))

@app.route('/page2')
@app.route('/page2.html')
def page2():
    return render_template('page2.html')

@app.route('/bouton2') 
def bouton2():
    return redirect(url_for('score_grid'))

@app.route('/score_grid')
@app.route('/score_grid.html')
def page3():
    return render_template('score_grid.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
