# Asesor Multiagente de Vivienda — ZMVM (CrewAI + Ollama)

Prototipo de la Fase 3 (Inteligencia Artificial, Universidad Anáhuac). Es un sistema multiagente que recomienda municipios, alcaldías y AGEB para buscar vivienda en la **Zona Metropolitana del Valle de México**. Toma en cuenta el perfil, las preferencias y la capacidad financiera del usuario.

## Agentes
| Agente | Herramienta | Delegación |
|---|---|---|
| Analista de Perfil del Usuario | `registrar_perfil_usuario` | No |
| Evaluador Territorial de Zonas | `puntuar_zonas` | No |
| Analista Financiero Hipotecario | `evaluar_viabilidad_financiera` | No |
| Coordinador de Recomendación | `integrar_ranking` | Sí |

Proceso: `Process.sequential` (perfil → zona → finanzas → coordinador). Los cálculos los hacen herramientas de Python, que comparten sus resultados en una "pizarra" (blackboard).

## Datos
- `datos/agebs_zmvm.csv`: 5,875 AGEB urbanas, 76 municipios y alcaldías, 6.0 millones de viviendas, con viviendas por nivel socioeconómico (AB a E) y centroide en lat/lon.
- Fuente: AMAI, niveles socioeconómicos por AGEB (shapefile `VALLE_DE_MEXICO.shp` + Excel `NSE_por_AGEB_AMAI.xlsx`).
- Para regenerarlo:
  ```bash
  pip install pandas openpyxl pyshp pyproj shapely
  python preparar_datos.py --shp VALLE_DE_MEXICO.shp --xlsx NSE_por_AGEB_AMAI.xlsx
  ```
- **Supuestos** (en `crew_vivienda.py`): precio de referencia por nivel socioeconómico, tasa hipotecaria de 11 % y coordenadas aproximadas de los centros de trabajo.

## Cómo correrlo

### Opción A: Google Colab
1. Sube `crew_vivienda.ipynb` a Colab y cambia el entorno a **GPU T4**.
2. Ejecuta las celdas en orden. La segunda instala Ollama y descarga `qwen2.5:7b` (~4.7 GB). La celda de datos te pide subir `agebs_zmvm.csv`.

### Opción B: tu computadora
```bash
ollama pull qwen2.5:7b
OLLAMA_CONTEXT_LENGTH=8192 ollama serve      # en otra terminal
pip install -r requirements.txt
python crew_vivienda.py
```
Para usar otro modelo: `MODELO_OLLAMA=llama3.1:8b python crew_vivienda.py`.

En Jupyter usa el notebook, que llama a `kickoff_async`; el `.py` usa `kickoff()`.

## Salidas (carpeta `salidas/`)
- `1_perfil.md`, `2_zonas.md`, `3_finanzas.md`, `4_recomendacion_final.md`: el reporte de cada tarea.
- `log_corrida.txt`: log del razonamiento y de la interacción entre agentes (para el Bloque D).
- `pizarra.json`: estado compartido con todos los cálculos.
