import streamlit as st
import pickle
import numpy as np
import pandas as pd


#### Modelo SVM ####

# Cargar el modelo y el vectorizador guardados
@st.cache_resource
def cargar_modelo():
    with open('svm-model.pck', 'rb') as f:
        dv, modelo_svm = pickle.load(f)  # Desempaquetar la tupla
    return dv, modelo_svm

# Cargar el modelo y vectorizador
dv, modelo_svm = cargar_modelo()


# Crear la aplicación Streamlit
st.title("Predicción de notas de estudiantes")

# Formulario para introducir datos 
st.header("Introduce los datos del estudiante:")

school = st.selectbox("Escuela del estudiante", ["GP", "MS"])
sex = st.selectbox("Sexo del estudiante", ["F", "M"])
age = st.number_input("Edad del estudiante", min_value=15, max_value=22, step=1)
address = st.selectbox("Tipo de dirección", ["U", "R"])
famsize = st.selectbox("Tamaño de la familia", ["LE3", "GT3"])
Pstatus = st.selectbox("Estado de convivencia de los padres", ["T", "A"])
Medu = st.number_input("Educación de la madre", min_value=0, max_value=4, step=1)
Fedu = st.number_input("Educación del padre", min_value=0, max_value=4, step=1)
Mjob = st.selectbox("Trabajo de la madre", ["teacher", "health", "services", "at_home", "other"])
Fjob = st.selectbox("Trabajo del padre", ["teacher", "health", "services", "at_home", "other"])
reason = st.selectbox("Razón para elegir esta escuela", ["home", "reputation", "course", "other"])
guardian = st.selectbox("Guardián del estudiante", ["mother", "father", "other"])
traveltime = st.number_input("Tiempo de viaje a la escuela (minutos)", min_value=1, max_value=4, step=1)
studytime = st.number_input("Tiempo de estudio semanal (horas)", min_value=1, max_value=4, step=1)
failures = st.number_input("Número de fallos pasados", min_value=1, max_value=4, step=1)
schoolsup = st.selectbox("Apoyo educativo adicional", ["yes", "no"])
famsup = st.selectbox("Apoyo familiar adicional", ["yes", "no"])
paid = st.selectbox("Clases extra pagadas", ["yes", "no"])
activities = st.selectbox("Actividades extra-curriculares", ["yes", "no"])
nursery = st.selectbox("Asistió a la escuela preescolar", ["yes", "no"])
higher = st.selectbox("Desea continuar con educación superior", ["yes", "no"])
internet = st.selectbox("Acceso a Internet en casa", ["yes", "no"])
romantic = st.selectbox("Está en una relación romántica", ["yes", "no"])
famrel = st.number_input("Relación familiar", min_value=1, max_value=5, step=1)
freetime = st.number_input("Tiempo libre después de la escuela", min_value=1, max_value=5, step=1)
goout = st.number_input("Salir con amigos", min_value=1, max_value=5, step=1)
Dalc = st.number_input("Consumo de alcohol en días laborales", min_value=1, max_value=5, step=1)
Walc = st.number_input("Consumo de alcohol en fines de semana", min_value=1, max_value=5, step=1)
health = st.number_input("Estado de salud actual", min_value=1, max_value=5, step=1)
absences = st.number_input("Número de ausencias escolares", min_value=0, max_value=93, step=1)
G1 = st.number_input("Calificación del primer período", min_value=0, max_value=20, step=1)
G2 = st.number_input("Calificación del segundo período", min_value=0, max_value=20, step=1)

# Crear un dataframe con los datos introducidos
nuevos_datos = pd.DataFrame({
    'school': [school],
    'sex': [sex],
    'age': [age],
    'address': [address],
    'famsize': [famsize],
    'Pstatus': [Pstatus],
    'Medu': [Medu],
    'Fedu': [Fedu],
    'Mjob': [Mjob],
    'Fjob': [Fjob],
    'reason': [reason],
    'guardian': [guardian],
    'traveltime': [traveltime],
    'studytime': [studytime],
    'failures': [failures],
    'schoolsup': [schoolsup],
    'famsup': [famsup],
    'paid': [paid],
    'activities': [activities],
    'nursery': [nursery],
    'higher': [higher],
    'internet': [internet],
    'romantic': [romantic],
    'famrel': [famrel],
    'freetime': [freetime],
    'goout': [goout],
    'Dalc': [Dalc],
    'Walc': [Walc],
    'health': [health],
    'absences': [absences],
    'G1': [G1],
    'G2': [G2]
})


# Aplicar el vectorizador a los datos procesados
nuevos_datos_vectorizados = dv.transform(nuevos_datos.to_dict(orient='records'))

# Realizar la predicción con el modelo cargado
if st.sidebar.button('Predecir'):
    prediccion = modelo_svm.predict_proba(nuevos_datos_vectorizados)[:, 1][0]
    
    # Mostrar el resultado
    aprueba = prediccion >= 0.5
    resultado_texto = "Aprobará" if aprueba else "No aprobará"
    st.write(f"**Resultado:** {resultado_texto}")
    st.write(f"**Probabilidad de aprobar:** {prediccion:.2f}")