# Importación de librerías necesarias
import yfinance as yf  # Para obtener datos financieros
import numpy as np  # Para cálculos numéricos
import pandas as pd  # Para manipulación de datos
from sklearn.preprocessing import MinMaxScaler  # Para escalar datos entre 0 y 1
from tensorflow.keras.models import Sequential  # Para construir modelos de redes neuronales
from tensorflow.keras.layers import LSTM, Dense, Dropout  # Capas para redes neuronales LSTM
import matplotlib.pyplot as plt  # Para generar gráficos
import io  # Para manejar gráficos en memoria
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile  # Para manejar interacciones con Telegram
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters  # Para manejar comandos y eventos de Telegram

# Diccionario que contiene las acciones disponibles y sus símbolos en el mercado
acciones = {
    "Bancolombia": "CIB",
    "Ecopetrol": "EC",
    "Grupo Aval": "AVAL",
    "Grupo Nutresa": "NCH"
}

# Variables globales para almacenar datos temporales
dias_a_predecir = {}  # Almacena el número de días a predecir por usuario
predicciones_cache = {}  # Cache para almacenar predicciones y evitar cálculos repetidos

# Función para predecir el precio de una acción para el día siguiente
def predecir_precio_para_manana(simbolo, model, scaler, df_scaled, time_step):
    """
    Predice el precio de una acción para el día siguiente utilizando un modelo LSTM.
    Si ya existe una predicción en el cache, la devuelve directamente.
    """
    if simbolo in predicciones_cache:
        return predicciones_cache[simbolo]

    # Usar los últimos datos escalados para realizar la predicción
    last_data = df_scaled[-time_step:].reshape(1, time_step, 1)
    prediction = model.predict(last_data)
    prediction_rescaled = scaler.inverse_transform(prediction)  # Desescalar la predicción

    # Guardar la predicción en el cache
    predicciones_cache[simbolo] = {
        "manana": prediction_rescaled[0][0],
        "futuro": None
    }
    return predicciones_cache[simbolo]

# Función para predecir precios futuros (hasta 30 días)
def predecir_futuro(simbolo, model, scaler, df_scaled, time_step, precio_predicho):
    """
    Genera predicciones para los próximos 30 días utilizando un modelo LSTM.
    Si ya existen predicciones futuras en el cache, las reutiliza.
    """
    if simbolo in predicciones_cache and predicciones_cache[simbolo]["futuro"] is not None:
        predicciones_cache[simbolo]["futuro"][0] = precio_predicho
        return predicciones_cache[simbolo]["futuro"]

    # Usar los últimos datos escalados para generar predicciones futuras
    last_data = df_scaled[-time_step:].reshape(1, time_step, 1)
    predictions = []

    for i in range(30):  # Generar predicciones para 30 días
        prediction = model.predict(last_data)
        prediction_rescaled = scaler.inverse_transform(prediction)
        predictions.append(prediction_rescaled[0][0])

        # Actualizar la ventana de datos con la nueva predicción
        last_data = np.append(last_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

    # Reemplazar el primer día con el precio predicho
    predictions[0] = precio_predicho

    # Guardar las predicciones futuras en el cache
    predicciones_cache[simbolo]["futuro"] = predictions
    return predictions

# Función para procesar los datos históricos de una acción
def procesar_datos(simbolo):
    """
    Descarga los datos históricos de una acción, los escala y los prepara para entrenar un modelo LSTM.
    """
    ticker = yf.Ticker(simbolo)
    df = ticker.history(period="max")['Close'].dropna()  # Obtener precios de cierre

    if df.empty:
        raise ValueError(f"No se encontraron datos para la compañía: {simbolo}")

    # Escalar los datos entre 0 y 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df.values.reshape(-1, 1))

    # Crear secuencias de datos para entrenar el modelo
    def create_sequences(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60  # Número de días usados para predecir
    X, y = create_sequences(df_scaled, time_step)
    if len(X) == 0 or len(y) == 0:
        raise ValueError(f"No se pudieron generar secuencias para la compañía: {simbolo}")

    # Redimensionar los datos para que sean compatibles con LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler, df_scaled, time_step, df

# Función para construir y entrenar un modelo LSTM
def construir_modelo(X_train, y_train, time_step):
    """
    Construye y entrena un modelo LSTM para predecir precios de acciones.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),  # Primera capa LSTM
        Dropout(0.2),  # Regularización para evitar sobreajuste
        LSTM(50, return_sequences=False),  # Segunda capa LSTM
        Dropout(0.2),
        Dense(1)  # Capa de salida
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')  # Configurar el modelo
    model.fit(X_train, y_train, epochs=10, batch_size=32)  # Entrenar el modelo
    return model

# Función para generar gráficos
def generar_grafico(df, predicciones_totales=None, dias=1, opcion=1, y_test_rescaled=None, y_pred_rescaled=None):
    """
    Genera gráficos basados en los datos históricos, predicciones o comparaciones.
    """
    if df.empty:
        raise ValueError("El DataFrame de datos históricos está vacío. No se puede generar el gráfico.")

    # Asegurarse de que el índice del DataFrame sea de tipo datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    plt.figure(figsize=(10, 6))
    if opcion == 1:  # Gráfico de datos históricos
        plt.plot(df.index, df.values, label="Datos históricos", color="blue")
        plt.title("Datos históricos")
    elif opcion == 2:  # Gráfico de predicciones vs. datos reales
        if y_test_rescaled is None or y_pred_rescaled is None:
            raise ValueError("Se requieren datos reales y predicciones para generar este gráfico.")
        plt.plot(y_test_rescaled, label="Datos reales", color="blue")
        plt.plot(y_pred_rescaled, label="Predicciones", color="orange")
        plt.title("Predicciones vs. Datos reales")
    elif opcion == 3:  # Gráfico de predicción futura
        if predicciones_totales is None or len(predicciones_totales) == 0:
            raise ValueError("No hay predicciones disponibles para generar el gráfico.")
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=dias, freq='D')
        plt.plot(df.index, df.values, label="Datos históricos", color="blue")
        plt.plot(future_dates, predicciones_totales[:dias], label="Predicción futura", color="green")
        plt.title(f"Predicción futura ({dias} días)")

    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return buffer

def generar_grafico_comparativo(predicciones_dict):
    if not predicciones_dict:
        raise ValueError("No hay predicciones disponibles para generar el gráfico comparativo.")

    nombres = list(predicciones_dict.keys())
    valores = np.array([predicciones_dict[nombre] for nombre in nombres])

    # Normalización para aplicar colores según el valor
    norm = (valores - valores.min()) / (valores.max() - valores.min() + 1e-6)
    colores = plt.cm.Greens(norm)

    # Mejor compañía (valor máximo)
    mejor_indice = np.argmax(valores)
    mejor_empresa = nombres[mejor_indice]
    mejor_valor = valores[mejor_indice]

    # Color dorado para resaltar la mejor
    colores[mejor_indice] = (0.0, 0.392, 0.0, 1.0)  # Verde oscuro

    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    barras = plt.bar(nombres, valores, color=colores)
    plt.xlabel("Compañía")
    plt.ylabel("Precio predicho ($)")
    plt.title("Predicción de precios para mañana")
    plt.grid(axis='y')

    # Añadir etiquetas de valor sobre cada barra
    for i, barra in enumerate(barras):
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width() / 2, altura + 0.5, f"${valores[i]:.2f}", ha='center')

    # Ajustar posición de la anotación para evitar superposición con el texto
    offset = max(valores) * 0.2  # mayor separación
    plt.annotate(
        f"🏆 Mejor proyección: {mejor_empresa} (${mejor_valor:.2f})",
        xy=(mejor_indice, mejor_valor),
        xytext=(mejor_indice, mejor_valor + offset),
        ha='center',
        fontsize=12,
        color='darkgreen',
        fontweight='bold'
    )

    # Guardar gráfico como imagen en buffer
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer

# Llamar la función de bienvenida
bienvenida_enviada = {}

# Función para mostrar la lista de acciones
async def mostrar_lista(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    user_id = update.effective_user.id

    # Solo mostrar la bienvenida si no ha sido enviada antes
    if user_id not in bienvenida_enviada:
        await bienvenida(update, context)
        bienvenida_enviada[user_id] = True

    # Mostrar las opciones de acciones disponibles
    keyboard = [[InlineKeyboardButton(nombre, callback_data=nombre)] for nombre in acciones.keys()]
    keyboard.append([InlineKeyboardButton("Todas", callback_data="Todas")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.message:
        await update.message.reply_text("Selecciona la compañia de la cual deseas predecir el comportamiento de su acción para mañana", reply_markup=reply_markup)
    elif update.callback_query:
        await update.callback_query.message.reply_text("Selecciona la compañia de la cual deseas predecir el comportamiento de su acción para mañana", reply_markup=reply_markup)


# Función para manejar la selección de una acción
# Función para manejar la selección de una acción
async def manejar_seleccion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    accion = query.data

    mensaje_procesando = await query.edit_message_text("Estamos trabajando en tu solicitud. Esto puede tardar unos segundos...")

    try:
        if accion == "Todas":
            predicciones = []
            predicciones_dict = {}
            for nombre, simbolo in acciones.items():
                try:
                    X_train, X_test, y_train, y_test, scaler, df_scaled, time_step, df = procesar_datos(simbolo)
                    model = construir_modelo(X_train, y_train, time_step)
                    precio_predicho = predecir_precio_para_manana(simbolo, model, scaler, df_scaled, time_step)["manana"]
                    predicciones.append(f"- {nombre}: ${precio_predicho:.2f}")
                    predicciones_dict[nombre] = precio_predicho
                except ValueError as e:
                    predicciones.append(f"- {nombre}: {str(e)}")

            mensaje_predicciones = "📈 **Predicciones para mañana:**\n" + "\n".join(predicciones)
            await mensaje_procesando.edit_text(mensaje_predicciones)

            if predicciones_dict:
                grafico_comparativo = generar_grafico_comparativo(predicciones_dict)
                await context.bot.send_photo(chat_id=query.message.chat_id, photo=InputFile(grafico_comparativo, filename="comparativo.png"))

            keyboard = [
                [InlineKeyboardButton("Seleccionar otra compañía", callback_data="menu")],
                [InlineKeyboardButton("Salir", callback_data="salir")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.message.reply_text("¿Qué deseas hacer ahora?", reply_markup=reply_markup)

        else:
            simbolo = acciones[accion]
            X_train, X_test, y_train, y_test, scaler, df_scaled, time_step, df = procesar_datos(simbolo)

            if df.empty:
                raise ValueError(f"No se encontraron datos para la compañía {accion}.")

            model = construir_modelo(X_train, y_train, time_step)
            precio_predicho = predecir_precio_para_manana(simbolo, model, scaler, df_scaled, time_step)["manana"]

            keyboard = [
                [InlineKeyboardButton("Datos históricos", callback_data=f"{accion}_1")],
                [InlineKeyboardButton("Predicciones vs. Datos reales", callback_data=f"{accion}_2")],
                [InlineKeyboardButton("Predicción futura", callback_data=f"{accion}_3")],
                [InlineKeyboardButton("Regresar al menú", callback_data="menu")],
                [InlineKeyboardButton("Salir", callback_data="salir")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            mensaje_predicho = await query.edit_message_text(
                f"El precio predicho de {accion} para el siguiente día es: ${precio_predicho:.2f}\n\n"
            )
            await mensaje_predicho.reply_text("Selecciona el tipo de gráfico que deseas ver o elige una opción:", reply_markup=reply_markup)

    except ValueError as e:
        await mensaje_procesando.edit_text(f"Error: {str(e)}")

# Nueva función para mostrar menú de gráficas
async def mostrar_menu_graficas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data.split("_")
    accion = data[0]

    keyboard = [
        [InlineKeyboardButton("Datos históricos", callback_data=f"{accion}_1")],
        [InlineKeyboardButton("Predicciones vs. Datos reales", callback_data=f"{accion}_2")],
        [InlineKeyboardButton("Predicción futura", callback_data=f"{accion}_3")],
        [InlineKeyboardButton("Seleccionar otra compañía", callback_data="menu")],
        [InlineKeyboardButton("Salir", callback_data="salir")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        f"Selecciona el tipo de gráfico que deseas ver para {accion}:",
        reply_markup=reply_markup
    )

# Función para manejar la selección de gráficos
async def manejar_grafico(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data.split("_")
    accion = data[0]
    opcion = int(data[1])

    # Solo mostrar el mensaje si la opción NO es 3 (es decir, no es predicción futura)
    if opcion != 3:
        # Mostrar mensaje de procesamiento mientras se genera el gráfico
        mensaje_procesando = await query.edit_message_text("Estamos generando el gráfico. Esto puede tardar unos segundos...") 

    if opcion == 3:  # Predicción futura
        dias_a_predecir[query.message.chat_id] = accion  # Guardar la acción seleccionada
        await query.message.reply_text("Por favor, ingresa el número de días a predecir (entre 1 y 30):")
        return

    simbolo = acciones[accion]
    X_train, X_test, y_train, y_test, scaler, df_scaled, time_step, df = procesar_datos(simbolo)
    model = construir_modelo(X_train, y_train, time_step)

    # Obtener el precio predicho para mañana
    precio_predicho = predecir_precio_para_manana(simbolo, model, scaler, df_scaled, time_step)["manana"]

    # Generar el gráfico seleccionado
    if opcion == 1:  # Datos históricos
        grafico = generar_grafico(df, opcion=opcion)
        analisis = generar_analisis(df, opcion=1, precio_predicho=precio_predicho)
    elif opcion == 2:  # Predicciones vs. datos reales
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_rescaled = scaler.inverse_transform(model.predict(X_test))
        grafico = generar_grafico(df, y_test_rescaled=y_test_rescaled, y_pred_rescaled=y_pred_rescaled, opcion=opcion)
        analisis = generar_analisis(df, y_test_rescaled=y_test_rescaled, y_pred_rescaled=y_pred_rescaled, opcion=2)

    # Enviar el gráfico al usuario
    await context.bot.send_photo(chat_id=query.message.chat_id, photo=InputFile(grafico, filename="grafico.png"))

    # Enviar el análisis textual
    await query.message.reply_text(analisis)

    # Mostrar opciones para seguir
    keyboard = [
        [InlineKeyboardButton("Generar otra gráfica", callback_data=f"{accion}_menu_graficas")],
        [InlineKeyboardButton("Seleccionar otra compañía", callback_data="menu")],
        [InlineKeyboardButton("Salir", callback_data="salir")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.reply_text("¿Qué deseas hacer ahora?", reply_markup=reply_markup)
    
# Nueva función para manejar el número de días ingresado por el usuario
async def manejar_dias(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    
    # Verifica que el usuario haya seleccionado una acción antes de continuar
    if chat_id not in dias_a_predecir:
        await update.message.reply_text("Por favor, selecciona una acción primero.")
        return

    # Tomamos el texto ingresado por el usuario y reemplazamos las comas por puntos
    texto_usuario = update.message.text.replace(',', '.')

    try:
        # Intentamos convertir el texto a un número decimal (float) y luego tomamos la parte entera
        dias = int(float(texto_usuario))  # Convierte el texto a float y luego a int para tomar solo la parte entera

        # Verificar que el número esté entre 1 y 30
        if dias < 1 or dias > 30:
            await update.message.reply_text("El número de días debe estar entre 1 y 30. Por favor, ingresa un número válido.")
            return
        
    except ValueError:
        # Si el valor no es un número válido (ni entero ni flotante), mostrar mensaje de error
        await update.message.reply_text("Por favor, ingresa un número válido entre 1 y 30.")
        return

    # Mostrar mensaje de procesamiento mientras se genera el gráfico
    await update.message.reply_text("Estamos generando la predicción para los próximos días... Esto puede tardar unos segundos.")

    # Si el número es válido, continúa con el proceso
    accion = dias_a_predecir.pop(chat_id)
    simbolo = acciones[accion]
    X_train, X_test, y_train, y_test, scaler, df_scaled, time_step, df = procesar_datos(simbolo)
    model = construir_modelo(X_train, y_train, time_step)

    # Obtener el precio predicho para mañana
    precio_predicho = predecir_precio_para_manana(simbolo, model, scaler, df_scaled, time_step)["manana"]

    # Obtener las predicciones futuras (30 días) y reemplazar el primer día con precio_predicho
    predicciones_totales = predecir_futuro(simbolo, model, scaler, df_scaled, time_step, precio_predicho)

    # Generar el gráfico de predicción futura
    grafico = generar_grafico(df, predicciones_totales=predicciones_totales, dias=dias, opcion=3)

    # Enviar el gráfico al usuario
    await context.bot.send_photo(chat_id=chat_id, photo=InputFile(grafico, filename="grafico.png"))

    # Enviar análisis de la predicción futura
    analisis = generar_analisis(df, prediction_rescaled=np.array(predicciones_totales[:dias]).reshape(-1, 1), opcion=3, dias=dias)
    await update.message.reply_text(analisis)

    # Mostrar opciones para seguir
    keyboard = [
        [InlineKeyboardButton("Generar otra gráfica", callback_data=f"{accion}_menu_graficas")],
        [InlineKeyboardButton("Seleccionar otra compañía", callback_data="menu")],
        [InlineKeyboardButton("Salir", callback_data="salir")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("¿Qué deseas hacer ahora?", reply_markup=reply_markup)

# Función para manejar el menú principal o salir
async def manejar_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "menu":
        await mostrar_lista(update, context)
    elif query.data == "salir":
        await query.edit_message_text("Gracias por usar el bot. ¡Hasta luego!")

# Configurar el bot
def main():
    TOKEN = "8122809488:AAGBMH00XUCw4KUy4KCJeqlty_XnSiOYT7o"
    app = ApplicationBuilder().token(TOKEN).build() 

     # Manejadores
    app.add_handler(CommandHandler("predecir", mostrar_lista))  # Manejo de /predecir
    app.add_handler(CallbackQueryHandler(mostrar_menu_graficas, pattern="^[^_]+_menu_graficas$"))
    app.add_handler(CallbackQueryHandler(manejar_grafico, pattern="^[^_]+_[123]$"))
    app.add_handler(CallbackQueryHandler(manejar_menu, pattern="^(menu|salir)$"))
    app.add_handler(CallbackQueryHandler(manejar_seleccion, pattern="^[^_]+$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manejar_dias))  # Manejar texto para días

    app.run_polling()

def generar_analisis(df, y_test_rescaled=None, y_pred_rescaled=None, prediction_rescaled=None, opcion=1, dias=1, precio_predicho=None):
    mensaje = ""
    if opcion == 1:  # Análisis de datos históricos
        mensaje = (
            f"📈 **Análisis de Datos Históricos**:\n"
            f"- Fecha inicial: {df.index[0].date()}\n"
            f"- Fecha final: {df.index[-1].date()}\n"
            f"- Precio mínimo: ${df.min():.2f}\n"
            f"- Precio máximo: ${df.max():.2f}\n"
            f"- Precio actual: ${precio_predicho:.2f}"  
        )
    elif opcion == 2 and y_test_rescaled is not None and y_pred_rescaled is not None:  # Análisis de predicciones vs. datos reales
        error = np.mean(np.abs(y_pred_rescaled - y_test_rescaled))
        mensaje = (
            f"📊 **Comparación Predicción vs. Real**:\n"
            f"- Promedio de precios reales: ${np.mean(y_test_rescaled):.2f}\n"
            f"- Promedio de predicciones: ${np.mean(y_pred_rescaled):.2f}\n"
            f"- Error absoluto medio (MAE): ${error:.2f}"
        )
    elif opcion == 3 and prediction_rescaled is not None:  # Análisis de predicción futura
        last_date = df.index[-1]
        mensaje = f"🔮 **Predicción Futura ({dias} días)**:\n"
        for i, p in enumerate(prediction_rescaled.flatten(), start=1):
            fecha_prediccion = (last_date + pd.Timedelta(days=i)).date()
            mensaje += f"- Día {i} ({fecha_prediccion}): ${p:.2f}\n"
    return mensaje
    
# Función para mostrar un mensaje introductorio con título y descripción
async def bienvenida(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensaje_bienvenida = (
        "🌟 **¡Bienvenido al Bot de Predicción de Precios de Acciones!** 🌟\n\n"
        "Este bot utiliza modelos avanzados de *Machine Learning* para predecir los precios futuros de acciones "
        "de las principales empresas. Solo selecciona una acción y obtén gráficos, análisis y predicciones precisas.\n\n"
        "🔍 **¿Cómo Funciona?**\n"
        "1. Elige una acción que deseas analizar.\n"
        "2. Visualiza los gráficos históricos, predicciones y más.\n"
        "3. ¡Obtén tu predicción para los próximos días!\n\n"
        "🔹 **Acciones Disponibles:** Bancolombia, Ecopetrol, Grupo Aval, Grupo Nutresa.\n\n"
        "¡Vamos a empezar! 🚀"
    )
    
    if update.message:
        await update.message.reply_text(mensaje_bienvenida)
    elif update.callback_query:
        await update.callback_query.message.reply_text(mensaje_bienvenida)

if __name__ == "__main__":
    main()