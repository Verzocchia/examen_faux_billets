import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import io

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

df_prep = pd.read_csv('./ressources/billets.csv', sep=';')
df_prep = df_prep.rename(columns={
    'is_genuine': 'type_billet',
    'diagonal': 'diagonale',
    'height_left': 'h_gauche',
    'height_right': 'h_droite',
    'margin_low': 'marge_bas',
    'margin_up': 'marge_haut',
    'length': 'longueur'
})

def complete_marge_bas(df):
    df_train = df[df['marge_bas'].notnull()]
    df_missing = df[df['marge_bas'].isnull()]
    if len(df_missing) > 0:
        model = RandomForestRegressor()
        model.fit(df_train.drop(columns=['marge_bas']), df_train['marge_bas'])
        df.loc[df['marge_bas'].isnull(), 'marge_bas'] = model.predict(df_missing.drop(columns=['marge_bas']))
    return df

df_prep = complete_marge_bas(df_prep)

features = ['diagonale', 'h_gauche', 'h_droite', 'marge_bas', 'marge_haut', 'longueur']
X = df_prep[features]
y = df_prep['type_billet']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)

#Section streamlit 

st.image("./ressources/oncfm.png")
st.markdown("<h1 style='text-align: center;'>Détection automatique de faux billets</h1>", unsafe_allow_html=True)
st.markdown("""
Veuillez entre un CSV contenant les caractéristiques de billets. Nous pourrons ainsi détecter
(ou plutôt tenter) les **vrais** et **faux** billets grâce à un modèle RandomForest entraîné.
""")

csv_dl = st.file_uploader("""📤 Uploadez un fichier CSV de billets ici. 
Il doit contenir, en anglais, les colonnes suivantes : 
|diagonal|height_left|height_right|margin_low|margin_up|length|is_genuine(facultatif)|
|--------|-----------|------------|----------|---------|------|----------------------|""", type=['csv'])


if csv_dl:
    ##Permet d'être agnostique sur le type de csv 
    content = csv_dl.read().decode("utf-8")
    sample = content[:1024]  

    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(sample)
    separator = dialect.delimiter
    ## 

    df = pd.read_csv(io.StringIO(content), sep=separator)
    df = df.rename(columns={
        'diagonal': 'diagonale',
        'height_left': 'h_gauche',
        'height_right': 'h_droite',
        'margin_low': 'marge_bas',
        'margin_up': 'marge_haut',
        'length': 'longueur',
        'is_genuine': 'type_billet'  
    })

    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        st.error(f"Le fichier est invalide : il manque les colonnes suivantes : {missing_cols}")
    else:
        df = complete_marge_bas(df)
        
        X_run = df[features]
        predictions = model.predict(X_run)
        proba = model.predict_proba(X_run)

        df['prediction'] = predictions
        df['proba_faux'] = proba[:, 0]
        df['proba_vrai'] = proba[:, 1]

        nb_total = len(df)
        nb_vrais = (df['prediction'] == True).sum()
        nb_faux = nb_total - nb_vrais
        pct_vrais = round(nb_vrais / nb_total * 100, 2)
        pct_faux = round(nb_faux / nb_total * 100, 2)

        st.subheader("Résultats sur les billets")
        st.markdown(f"- Nombre total de billets : **{nb_total}**")
        st.markdown(f"- ✅ Billets **vrais** détectés : **{nb_vrais}** ({pct_vrais}%)")
        st.markdown(f"- ❌ Billets **faux** détectés : **{nb_faux}** ({pct_faux}%)")

        # Si colonne "type_billet" existe, on calcule la précision
        if 'type_billet' in df.columns:
            acc = accuracy_score(df['type_billet'], df['prediction'])
            st.markdown(f"Précision du modèle : **{round(acc * 100, 2)}%**")

            cm = confusion_matrix(df['type_billet'], df['prediction'])
            matrice_conf = pd.DataFrame(cm, index=["Faux (vrai)", "Vrai (vrai)"], 
                                            columns=["Prédit Faux", "Prédit Vrai"])
            st.write(matrice_conf)

            st.title("Anomalies")
            df_anomalies = df[df['type_billet'] != df['prediction']]
            csv_anomalies = df_anomalies.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger les anomalies", 
                               data=csv_anomalies, 
                               file_name='erreurs_prediction.csv', 
                               mime='text/csv')

        st.title("Détail des prédictions")
        st.dataframe(df.head(5))

        st.markdown("""Si vous souhaitez avoir les résultats pour l'ensemble du jeu de données, 
                    vous pouvez le télécharger ici. """)
        csv_result = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger les résultats", 
                           data=csv_result, 
                           file_name='resultats_predictions.csv', 
                           mime='text/csv')

else:
    st.info("Veuillez téléverser un fichier CSV pour commencer.")