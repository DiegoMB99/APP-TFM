import streamlit as st
import pandas as pd
import pickle
import catboost 

# Cargar el modelo desde el archivo pickle
with open('palladium_model_catboost_reducido.pkl', 'rb') as archivo:
    loaded_model = pickle.load(archivo)


def predeccion(DIA_ANIO_LLEGADA, MES_LLEGADA, TOP_PAIS_RESV, NUM_MES_FECHA_MOD):
    # Construir el DataFrame con las características
    datos = {
        'DIA_ANIO_LLEGADA': [DIA_ANIO_LLEGADA],
        'NUM_MES_LLEGADA': [MES_LLEGADA],
        'NUM_MES_FECHA_MOD': [NUM_MES_FECHA_MOD],
        'TOP_PAIS_RESV': [TOP_PAIS_RESV]
    }
    
    df = pd.DataFrame(datos)

    # Realizar la predicción con el modelo cargado
    probabilidad_cancelacion = loaded_model.predict_proba(df)[:, 1]
    return probabilidad_cancelacion[0]

    # Definir opciones para las características categóricas
TOP_PAIS_RESV = ['México', 'Estados Unidos', 'África', 'Canadá', 'América del Sur',
       'España', 'Oceanía', 'Brasil', 'Argentina', 'Perú', 'Uruguay',
       'Chile', 'Europa', 'Alemania', 'Colombia', 'Reino Unido', 'Rusia',
       'Asia', 'Sin País', 'Afganistán', 'América Central',
       'República Dominicana', 'Antártida', 'América del Norte']

# Setting layout page
st.set_page_config(layout='centered')

# Streamlit App setting and inputs
with st.container():
    column_1, column_2 = st.columns([0.8,1])
    with column_2:
        st.title('Cancelación de reservas hoteleras')
        st.write("Modelo de predicción de cancelaciones")

        DIA_ANIO_LLEGADA = st.slider("Días de estancia", min_value=1, max_value=20)
        
        TOP_PAIS_RESV = st.selectbox("Zona de procedencia del cliente", options=[''] + TOP_PAIS_RESV)

        MESES_LLEGADA =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 
        NUM_MES_LLEGADA = st.selectbox("Mes de llegada", MESES_LLEGADA)

        MESES_MOD = [i for i in range(1, NUM_MES_LLEGADA + 1)]  # Solo permite seleccionar meses hasta NUM_MES_LLEGADA
        NUM_MES_FECHA_MOD = st.selectbox("Mes de  modificacion", MESES_MOD)


 # Output
        if st.button('Predecir'):
            if not DIA_ANIO_LLEGADA or not NUM_MES_LLEGADA or not TOP_PAIS_RESV or not NUM_MES_FECHA_MOD:
                st.warning('Es neceario completar todos los campos para realizar la predicción.')
            else:
                probabilidad = predeccion(DIA_ANIO_LLEGADA, NUM_MES_LLEGADA, TOP_PAIS_RESV, NUM_MES_FECHA_MOD)
                st.write(f'**Esta reserva tiene una probabilidad del {probabilidad:.2%} de ser cancelada.**')
                if probabilidad < 0.3:
                    mensaje = "Recomendamos añadir al cliente un descuento en excursiones."
                elif probabilidad < 0.6:
                    mensaje = "Recomendamos añadir al cliente un '10%' de descuento en la reserva."
                else:
                    mensaje = "Alta probabilidad de cancelación, añadiendo automáticamente a la lista de churn."
                st.write(f'**{mensaje}**')
