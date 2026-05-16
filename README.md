# RRHH Predictivo en Python

Aplicativo funcional y responsive para PC y celular que demuestra métodos predictivos aplicados a Recursos Humanos. La app permite ajustar datos de un colaborador y obtener predicciones en tiempo real para apoyar decisiones de talento.

## Funcionalidades

- Predicción de riesgo de rotación.
- Estimación de desempeño esperado.
- Probabilidad de promoción.
- Salario recomendado como referencia interna.
- Recomendaciones accionables de retención, bienestar y capacitación.
- Tablero analítico con visualizaciones y métricas de validación.
- Diseño responsive para escritorio y pantallas móviles mediante Streamlit y CSS.

## Métodos predictivos incluidos

La aplicación entrena modelos demostrativos con datos sintéticos:

- **Random Forest Classifier** para rotación y promoción.
- **Random Forest Regressor** para desempeño y salario recomendado.
- Separación de datos en entrenamiento y prueba para reportar exactitud, R² y MAE cuando corresponde.

> Importante: los datos son sintéticos. Para uso real, se deben conectar datos internos autorizados, anonimizar información sensible, validar sesgos y mantener supervisión humana.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución en PC

```bash
streamlit run app.py
```

Luego abre la URL local que muestra Streamlit, normalmente `http://localhost:8501`.

## Uso desde celular

1. Conecta el celular a la misma red Wi-Fi del PC.
2. Ejecuta la app en el PC con acceso de red:

```bash
streamlit run app.py --server.address 0.0.0.0
```

3. En el celular abre `http://IP_DEL_PC:8501`.

## Pruebas

```bash
python -m pytest
```
