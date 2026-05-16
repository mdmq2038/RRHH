# RRHH Predictivo en Python

Aplicativo funcional y responsive para PC y celular que demuestra metodos predictivos aplicados a Recursos Humanos. La app permite ajustar datos de un colaborador y obtener predicciones en tiempo real para apoyar decisiones de talento.

## Funcionalidades

- Prediccion de riesgo de rotacion.
- Estimacion de desempeno esperado.
- Probabilidad de promocion.
- Salario recomendado como referencia interna.
- Recomendaciones accionables de retencion, bienestar y capacitacion.
- Tablero analitico con visualizaciones y metricas de validacion.
- Diseno responsive para escritorio y pantallas moviles mediante Streamlit y CSS.

## Metodos predictivos incluidos

La aplicacion entrena modelos demostrativos con datos sinteticos:

- Random Forest Classifier para rotacion y promocion.
- Random Forest Regressor para desempeno y salario recomendado.
- Separacion de datos en entrenamiento y prueba para reportar exactitud, R2 y MAE cuando corresponde.

Importante: los datos son sinteticos. Para uso real, se deben conectar datos internos autorizados, anonimizar informacion sensible, validar sesgos y mantener supervision humana.

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecucion local

```bash
streamlit run app.py
```

Luego abre la URL local que muestra Streamlit, normalmente `http://localhost:8501`.

## Despliegue en Render

El repositorio ya incluye lo necesario para Render:

- `render.yaml` con el servicio web `RRHH`.
- `Procfile` para arrancar Streamlit en el puerto asignado por Render.
- `.streamlit/config.toml` con modo headless y estadisticas desactivadas.

Si Render esta conectado al repositorio de GitHub, cada push a `main` dispara un redeploy automatico.

## Pruebas

```bash
python -m pytest
```
