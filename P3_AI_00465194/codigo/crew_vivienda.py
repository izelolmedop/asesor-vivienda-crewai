"""
Asesor Multiagente de Vivienda — Fase 3 (Sistemas Multiagente con CrewAI)
=========================================================================

Sistema de apoyo a decisiones para elegir zona de vivienda en la Zona
Metropolitana del Valle de México (ZMVM). Un equipo (Crew) de 4 agentes
especializados colabora de forma secuencial:

    1. Agente de Perfil del Usuario  -> interpreta necesidades y restricciones
    2. Agente de Evaluación de Zona  -> puntúa municipios/alcaldías
    3. Agente Financiero             -> evalúa la viabilidad de pago
    4. Agente Coordinador            -> integra, resuelve conflictos y rankea

Datos del entorno (reales, AMAI):
    datos/agebs_zmvm.csv — 5,875 AGEB urbanas de la ZMVM con viviendas por
    nivel socioeconómico (AB a E) y centroide geográfico. Se genera con
    preparar_datos.py a partir del shapefile y del Excel NSE por AGEB de AMAI.

Supuestos (no vienen en los datos y están marcados como tales):
    precio de referencia por nivel socioeconómico, tasa hipotecaria,
    coordenadas aproximadas de los centros de trabajo.

El modelo de lenguaje corre LOCALMENTE con Ollama (sin API de pago).

Arquitectura de "pizarra" (blackboard):
    Los agentes razonan en lenguaje natural, pero los CÁLCULOS se hacen en
    herramientas deterministas de Python. Sus resultados se guardan en un
    estado compartido (PIZARRA) que leen los demás agentes, así un modelo
    pequeño (7-8B) no copia tablas de números entre agentes.

Requisitos:
    pip install "crewai[litellm]" pandas
    ollama pull qwen2.5:7b
    OLLAMA_CONTEXT_LENGTH=8192 ollama serve     # contexto amplio para el Crew

Uso:
    python crew_vivienda.py
"""

import json
import math
import os
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Type

import pandas as pd
from pydantic import BaseModel, Field

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import BaseTool

# ---------------------------------------------------------------------------
# 0. Configuración general
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
os.chdir(BASE_DIR)                       # CrewAI guarda output_file relativo al directorio actual
RUTA_AGEBS = Path("datos") / "agebs_zmvm.csv"
DIR_SALIDAS = Path("salidas")
DIR_SALIDAS.mkdir(exist_ok=True)

MODELO_OLLAMA = os.getenv("MODELO_OLLAMA", "qwen2.5:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

NIVELES = ["AB", "C+", "C", "C-", "D+", "D", "E"]

# Escala ordinal del nivel socioeconómico (AMAI): AB = 7 ... E = 1
VALOR_NSE = {"AB": 7, "C+": 6, "C": 5, "C-": 4, "D+": 3, "D": 2, "E": 1}

# SUPUESTO: precio de referencia de una vivienda de 2 recámaras según el nivel
# socioeconómico de la AGEB (MXN). Sustituible por datos de portales inmobiliarios.
PRECIO_POR_NSE = {"AB": 7_500_000, "C+": 4_200_000, "C": 2_600_000, "C-": 1_700_000,
                  "D+": 1_150_000, "D": 800_000, "E": 550_000}
FACTOR_RECAMARAS = {1: 0.85, 2: 1.00, 3: 1.20, 4: 1.40}   # ajuste por tamaño

# Supuestos financieros
TASA_ANUAL_BASE = 0.11        # tasa hipotecaria anual de referencia (11 %)
PLAZO_BASE_ANIOS = 20         # plazo de referencia
RATIO_MAX_PAGO = 0.30         # el pago mensual no debe superar 30 % del ingreso
ENGANCHE_MIN = 0.10           # los bancos piden al menos 10 % de enganche
GASTOS_ESCRITURACION = 0.05   # ~5 % del valor se va en notaría e impuestos
OFERTA_ACCESIBLE_MIN = 0.15   # al menos 15 % de las viviendas deben ser costeables

# SUPUESTO: coordenadas aproximadas (lat, lon) de centros de trabajo de la ZMVM
LUGARES_TRABAJO = {
    "santa fe": (19.3594, -99.2595),
    "polanco": (19.4330, -99.1950),
    "reforma": (19.4270, -99.1677),
    "centro historico": (19.4326, -99.1332),
    "ciudad universitaria": (19.3326, -99.1870),
    "coyoacan": (19.3467, -99.1617),
    "insurgentes sur": (19.3700, -99.1790),
    "ciudad satelite": (19.5094, -99.2370),
    "tlalnepantla industrial": (19.5400, -99.1950),
    "vallejo industrial": (19.4900, -99.1600),
    "naucalpan industrial": (19.4790, -99.2390),
    "cuautitlan izcalli": (19.6460, -99.2150),
    "ecatepec": (19.6018, -99.0507),
    "aeropuerto aicm": (19.4361, -99.0719),
    "aeropuerto aifa": (19.7456, -99.0108),
}

# LLM local servido por Ollama. Temperatura baja = respuestas más consistentes.
llm_local = LLM(
    model=f"ollama/{MODELO_OLLAMA}",
    base_url=OLLAMA_URL,
    temperature=0.1,
)

# ---------------------------------------------------------------------------
# 1. Estado compartido (pizarra) y funciones de apoyo
# ---------------------------------------------------------------------------
PIZARRA: dict = {}
_CACHE: dict = {}


def cargar_agebs() -> pd.DataFrame:
    """AGEB urbanas de la ZMVM con viviendas por nivel socioeconómico (AMAI)."""
    if "agebs" not in _CACHE:
        df = pd.read_csv(RUTA_AGEBS, dtype={"cve_ageb": str, "cve_mun": str})
        _CACHE["agebs"] = df[df["viviendas"] > 0].copy()
    return _CACHE["agebs"]


def resumen_municipios() -> pd.DataFrame:
    """Agrega las AGEB por municipio/alcaldía: viviendas, mezcla de NSE y centroide."""
    if "municipios" not in _CACHE:
        df = cargar_agebs()
        mun = df.groupby(["cve_mun", "municipio", "entidad"])[NIVELES + ["viviendas"]].sum().reset_index()
        # centroide ponderado por viviendas (donde vive la gente, no el centro del polígono)
        df = df.assign(wlat=df["lat"] * df["viviendas"], wlon=df["lon"] * df["viviendas"])
        cen = df.groupby("cve_mun")[["wlat", "wlon", "viviendas"]].sum()
        mun["lat"] = mun["cve_mun"].map(cen["wlat"] / cen["viviendas"])
        mun["lon"] = mun["cve_mun"].map(cen["wlon"] / cen["viviendas"])
        mun["n_agebs"] = mun["cve_mun"].map(df.groupby("cve_mun").size())
        total_nse = mun[NIVELES].sum(axis=1).clip(lower=1)
        # índice NSE 0-10: promedio de la escala 1-7 ponderado por viviendas
        prom = sum(mun[n] * VALOR_NSE[n] for n in NIVELES) / total_nse
        mun["indice_nse"] = ((prom - 1) / 6 * 10).round(2)
        for n in NIVELES:
            mun[f"pct_{n}"] = mun[n] / total_nse
        mun["nse_mayoritario"] = mun[NIVELES].idxmax(axis=1)
        _CACHE["municipios"] = mun
    return _CACHE["municipios"]


def distancia_km(lat1, lon1, lat2, lon2) -> float:
    """Distancia en línea recta (fórmula del haversine)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def normalizar(texto: str) -> str:
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_acentos.lower().strip()


def pago_mensual(principal: float, tasa_anual: float, plazo_anios: int) -> float:
    """Fórmula de amortización francesa (pagos fijos)."""
    r, n = tasa_anual / 12, plazo_anios * 12
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def evaluar_precio(precio, perfil, tasa_anual, plazo_anios) -> dict:
    """Viabilidad de comprar una vivienda de cierto precio con el perfil dado."""
    enganche_neto = max(perfil["enganche_disponible"] - precio * GASTOS_ESCRITURACION, 0)
    credito = max(precio - enganche_neto, 0)
    pago = pago_mensual(credito, tasa_anual, plazo_anios)
    ratio = pago / perfil["ingreso_mensual"]
    pct_eng = enganche_neto / precio
    viable = pct_eng >= ENGANCHE_MIN and ratio <= RATIO_MAX_PAGO and precio <= perfil["presupuesto_max"]
    return {"pago": pago, "ratio": ratio, "pct_enganche": pct_eng, "viable": viable}


def precio_nivel(nivel: str, recamaras: int) -> float:
    return PRECIO_POR_NSE[nivel] * FACTOR_RECAMARAS[min(max(int(recamaras), 1), 4)]


# ---------------------------------------------------------------------------
# 2. Herramientas personalizadas (tools)
# ---------------------------------------------------------------------------
# 2.1 Agente de Perfil -------------------------------------------------------
class PerfilInput(BaseModel):
    ingreso_mensual: float = Field(..., description="Ingreso mensual neto del hogar en MXN")
    enganche_disponible: float = Field(..., description="Dinero disponible para enganche en MXN")
    presupuesto_max: float = Field(..., description="Precio máximo de vivienda que acepta, MXN")
    recamaras_min: int = Field(2, description="Número mínimo de recámaras (1 a 4)")
    lugar_trabajo: str = Field(..., description="Centro de trabajo; uno de: " + ", ".join(LUGARES_TRABAJO))
    distancia_max_km: float = Field(20, description="Distancia máxima aceptable al trabajo en km")
    prioridad_nivel_entorno: int = Field(3, description="Importancia 1-5 del nivel socioeconómico del entorno")
    prioridad_cercania_trabajo: int = Field(3, description="Importancia 1-5 de vivir cerca del trabajo")
    prioridad_oferta_vivienda: int = Field(3, description="Importancia 1-5 de que haya mucha oferta de vivienda")


class RegistrarPerfilTool(BaseTool):
    name: str = "registrar_perfil_usuario"
    description: str = (
        "Registra en la pizarra compartida el perfil del usuario (ingresos, enganche, "
        "presupuesto, recámaras, lugar de trabajo, distancia máxima y prioridades 1-5) "
        "y devuelve los pesos normalizados. lugar_trabajo debe ser uno de: "
        + ", ".join(LUGARES_TRABAJO) + "."
    )
    args_schema: Type[BaseModel] = PerfilInput

    def _run(self, **kw) -> str:
        perfil = PerfilInput(**kw).model_dump()
        clave = normalizar(perfil["lugar_trabajo"])
        coincidencias = [k for k in LUGARES_TRABAJO if k in clave or clave in k]
        if not coincidencias:
            return "ERROR: lugar_trabajo no reconocido. Usa uno de: " + ", ".join(LUGARES_TRABAJO)
        perfil["lugar_trabajo"] = coincidencias[0]
        prior = {
            "nivel_entorno": perfil["prioridad_nivel_entorno"],
            "cercania_trabajo": perfil["prioridad_cercania_trabajo"],
            "oferta_vivienda": perfil["prioridad_oferta_vivienda"],
        }
        prior = {k: min(max(int(v), 1), 5) for k, v in prior.items()}
        pesos = {k: round(v / sum(prior.values()), 3) for k, v in prior.items()}
        PIZARRA["perfil"], PIZARRA["pesos"] = perfil, pesos
        return json.dumps({"perfil": perfil, "pesos_normalizados": pesos}, ensure_ascii=False)


# 2.2 Agente de Evaluación de Zona -------------------------------------------
class SinArgumentos(BaseModel):
    confirmar: bool = Field(True, description="Siempre True; la herramienta usa el perfil de la pizarra")


class PuntuarZonasTool(BaseTool):
    name: str = "puntuar_zonas"
    description: str = (
        "Puntúa (0-10) los 76 municipios y alcaldías de la ZMVM con datos AMAI por AGEB: "
        "nivel socioeconómico del entorno, cercanía al trabajo y oferta de vivienda, "
        "ponderados con los pesos del perfil de la pizarra. Devuelve los 12 mejores "
        "dentro de la distancia máxima del usuario."
    )
    args_schema: Type[BaseModel] = SinArgumentos

    def _run(self, confirmar: bool = True) -> str:
        if "pesos" not in PIZARRA:
            return "ERROR: primero debe registrarse el perfil del usuario."
        p, perfil = PIZARRA["pesos"], PIZARRA["perfil"]
        lat_t, lon_t = LUGARES_TRABAJO[perfil["lugar_trabajo"]]
        mun = resumen_municipios().copy()
        mun["dist_km"] = [round(distancia_km(a, b, lat_t, lon_t), 1) for a, b in zip(mun["lat"], mun["lon"])]
        # cercanía: 10 en el lugar de trabajo, 0 a partir de 50 km
        mun["cercania"] = (10 * (1 - mun["dist_km"] / 50)).clip(lower=0).round(2)
        # oferta: escala logarítmica respecto al municipio con más viviendas
        mun["oferta"] = (10 * mun["viviendas"].apply(math.log) / math.log(mun["viviendas"].max())).round(2)
        mun["puntaje_zona"] = (p["nivel_entorno"] * mun["indice_nse"]
                               + p["cercania_trabajo"] * mun["cercania"]
                               + p["oferta_vivienda"] * mun["oferta"]).round(2)
        mun["dentro_radio"] = mun["dist_km"] <= perfil["distancia_max_km"]
        PIZARRA["zonas"] = mun.set_index("municipio")[
            ["puntaje_zona", "indice_nse", "dist_km", "dentro_radio", "nse_mayoritario", "viviendas"]
        ].to_dict(orient="index")

        dentro = mun[mun["dentro_radio"]].sort_values("puntaje_zona", ascending=False)
        top = dentro.head(12)[["municipio", "entidad", "puntaje_zona", "indice_nse",
                               "dist_km", "nse_mayoritario", "viviendas"]]
        return json.dumps({
            "municipios_dentro_radio": int(len(dentro)),
            "municipios_fuera_radio": int((~mun["dentro_radio"]).sum()),
            "top": top.to_dict(orient="records"),
        }, ensure_ascii=False)


# 2.3 Agente Financiero -----------------------------------------------------
class FinancieroInput(BaseModel):
    tasa_anual: float = Field(TASA_ANUAL_BASE, description="Tasa hipotecaria anual en decimal, ej. 0.11")
    plazo_anios: int = Field(PLAZO_BASE_ANIOS, description="Plazo del crédito en años: 15 o 20")


class EvaluarFinanzasTool(BaseTool):
    name: str = "evaluar_viabilidad_financiera"
    description: str = (
        "Para cada municipio dentro de la distancia máxima del usuario calcula: el precio "
        "típico (vivienda mediana según su mezcla de NSE), el pago mensual y su % del "
        "ingreso, y el % de viviendas del municipio que el usuario puede costear. "
        "Devuelve un puntaje financiero 0-10. Úsala con el escenario base (0.11, 20) "
        "y con un escenario alternativo."
    )
    args_schema: Type[BaseModel] = FinancieroInput

    def _run(self, tasa_anual: float = TASA_ANUAL_BASE, plazo_anios: int = PLAZO_BASE_ANIOS) -> str:
        if "perfil" not in PIZARRA:
            return "ERROR: primero debe registrarse el perfil del usuario."
        perfil = PIZARRA["perfil"]
        rec = perfil["recamaras_min"]
        lat_t, lon_t = LUGARES_TRABAJO[perfil["lugar_trabajo"]]
        # qué niveles socioeconómicos son costeables para este hogar
        costeable = {n: evaluar_precio(precio_nivel(n, rec), perfil, tasa_anual, plazo_anios)["viable"]
                     for n in NIVELES}

        filas = []
        for _, m in resumen_municipios().iterrows():
            if distancia_km(m["lat"], m["lon"], lat_t, lon_t) > perfil["distancia_max_km"]:
                continue                                    # restricción dura del perfil
            # precio típico = nivel de la vivienda mediana (acumulando de E hacia AB)
            acumulado, nivel_mediano = 0.0, "AB"
            for n in reversed(NIVELES):
                acumulado += m[f"pct_{n}"]
                if acumulado >= 0.5:
                    nivel_mediano = n
                    break
            precio = precio_nivel(nivel_mediano, rec)
            ev = evaluar_precio(precio, perfil, tasa_anual, plazo_anios)
            pct_accesible = float(sum(m[f"pct_{n}"] for n in NIVELES if costeable[n]))
            # 10 si el pago típico <= 20 % del ingreso, 0 si >= 40 %
            puntaje_pago = max(0.0, min(10.0, 10 * (0.40 - ev["ratio"]) / 0.20))
            if ev["pct_enganche"] < ENGANCHE_MIN:
                puntaje_pago = 0.0
            puntaje = 0.6 * 10 * pct_accesible + 0.4 * puntaje_pago
            filas.append({
                "municipio": m["municipio"],
                "nivel_mediano": nivel_mediano,
                "precio_tipico": round(precio),
                "pago_tipico": round(ev["pago"]),
                "ratio_pago_ingreso": round(ev["ratio"], 3),
                "tipico_viable": bool(ev["viable"]),
                "pct_oferta_accesible": round(pct_accesible, 3),
                "puntaje_financiero": round(puntaje, 2),
            })
        filas.sort(key=lambda f: f["puntaje_financiero"], reverse=True)
        escenario = f"tasa {tasa_anual:.1%}, {plazo_anios} años"
        PIZARRA.setdefault("escenarios", {})[escenario] = filas
        niveles_ok = [n for n in NIVELES if costeable[n]]
        if abs(tasa_anual - TASA_ANUAL_BASE) < 1e-9 and plazo_anios == PLAZO_BASE_ANIOS:
            PIZARRA["finanzas_base"] = {f["municipio"]: f for f in filas}
            PIZARRA["niveles_costeables"] = niveles_ok
        return json.dumps({
            "escenario": escenario,
            "niveles_nse_costeables": niveles_ok,
            "precio_referencia_por_nivel": {n: round(precio_nivel(n, rec)) for n in NIVELES},
            "municipios_evaluados": len(filas),
            "top": filas[:12],
        }, ensure_ascii=False)


# 2.4 Agente Coordinador ----------------------------------------------------
class IntegrarInput(BaseModel):
    peso_zona: float = Field(0.5, description="Peso de la evaluación territorial (0-1)")
    peso_financiero: float = Field(0.5, description="Peso de la evaluación financiera (0-1)")


class IntegrarRankingTool(BaseTool):
    name: str = "integrar_ranking"
    description: str = (
        "Combina el puntaje territorial y el financiero de la pizarra con los pesos "
        "indicados, descarta municipios que violan restricciones duras (fuera de la "
        "distancia máxima o con menos de 15 % de oferta costeable) y devuelve el ranking "
        "final con las 3 mejores AGEB costeables de cada uno de los 3 primeros."
    )
    args_schema: Type[BaseModel] = IntegrarInput

    def _run(self, peso_zona: float = 0.5, peso_financiero: float = 0.5) -> str:
        faltan = [k for k in ("zonas", "finanzas_base") if k not in PIZARRA]
        if faltan:
            return f"ERROR: faltan resultados previos en la pizarra: {faltan}"
        s = peso_zona + peso_financiero
        wz, wf = peso_zona / s, peso_financiero / s
        lat_t, lon_t = LUGARES_TRABAJO[PIZARRA["perfil"]["lugar_trabajo"]]

        ranking, descartados = [], []
        for mun, z in PIZARRA["zonas"].items():
            if not z["dentro_radio"]:
                continue                       # fuera de la distancia máxima del usuario
            f = PIZARRA["finanzas_base"][mun]
            reg = {"municipio": mun,
                   "puntaje_zona": z["puntaje_zona"],
                   "puntaje_financiero": f["puntaje_financiero"],
                   "puntaje_final": round(wz * z["puntaje_zona"] + wf * f["puntaje_financiero"], 2),
                   "dist_km": z["dist_km"],
                   "precio_tipico": f["precio_tipico"],
                   "pct_oferta_accesible": f["pct_oferta_accesible"]}
            if f["pct_oferta_accesible"] < OFERTA_ACCESIBLE_MIN:
                reg["motivo"] = f"solo {f['pct_oferta_accesible']:.0%} de la oferta es costeable"
                descartados.append(reg)
            else:
                ranking.append(reg)
        ranking.sort(key=lambda r: r["puntaje_final"], reverse=True)
        descartados.sort(key=lambda r: r["puntaje_zona"], reverse=True)

        # AGEB concretas: costeables (NSE predominante dentro de lo costeable),
        # ordenadas por mejor nivel socioeconómico y luego por cercanía al trabajo
        costeables = set(PIZARRA["niveles_costeables"])
        agebs = cargar_agebs().copy()
        agebs["nivel"] = agebs["nse_predominante"].replace({"A/B": "AB"})
        for i, r in enumerate(ranking, 1):
            r["posicion"] = i
            if i <= 3:
                a = agebs[(agebs["municipio"] == r["municipio"]) & agebs["nivel"].isin(costeables)].copy()
                a["dist_km"] = [round(distancia_km(x, y, lat_t, lon_t), 1) for x, y in zip(a["lat"], a["lon"])]
                a["valor"] = a["nivel"].map(VALOR_NSE)
                a = a.sort_values(["valor", "dist_km"], ascending=[False, True]).head(3)
                r["agebs_sugeridas"] = a[["cve_ageb", "nivel", "dist_km", "viviendas"]].to_dict(orient="records")
        PIZARRA["ranking"] = ranking
        return json.dumps({"pesos_usados": {"zona": round(wz, 2), "financiero": round(wf, 2)},
                           "ranking_top5": ranking[:5],
                           "descartados_top5": descartados[:5],
                           "total_descartados": len(descartados)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 3. Agentes especializados
# ---------------------------------------------------------------------------
agente_perfil = Agent(
    role="Analista de Perfil del Usuario",
    goal=(
        "Traducir la descripción en lenguaje natural del usuario a un perfil "
        "estructurado (ingresos, enganche, presupuesto, recámaras, lugar de trabajo, "
        "distancia máxima y prioridades de 1 a 5) sin inventar datos que no dio."
    ),
    backstory=(
        "Eres asesora inmobiliaria con 10 años de experiencia entrevistando familias "
        "en la Ciudad de México y el Estado de México. Distingues entre lo que el "
        "cliente NECESITA (restricciones duras) y lo que PREFIERE (prioridades). "
        "Cuando un dato no se menciona, usas el valor por defecto y lo declaras "
        "como supuesto."
    ),
    tools=[RegistrarPerfilTool()],
    llm=llm_local,
    allow_delegation=False,   # primer eslabón: no depende de nadie
    max_iter=5,
    verbose=True,
)

agente_zona = Agent(
    role="Evaluador Territorial de Zonas",
    goal=(
        "Puntuar objetivamente los municipios y alcaldías de la Zona Metropolitana "
        "del Valle de México según el nivel socioeconómico de su entorno, la "
        "cercanía al trabajo y la oferta de vivienda, explicando las mejores opciones."
    ),
    backstory=(
        "Eres urbanista especializado en la estructura metropolitana del Valle de "
        "México y conoces la regionalización socioeconómica de AMAI por AGEB. "
        "No opinas sobre dinero: tu trabajo es solo la calidad y ubicación del "
        "territorio, y siempre basas tus conclusiones en la herramienta de puntuación."
    ),
    tools=[PuntuarZonasTool()],
    llm=llm_local,
    allow_delegation=False,   # evaluación independiente de la financiera
    max_iter=5,
    verbose=True,
)

agente_financiero = Agent(
    role="Analista Financiero Hipotecario",
    goal=(
        "Determinar en qué municipios la compra es financieramente viable para el "
        "usuario y bajo qué escenario de crédito, sin aprobar opciones cuyo pago "
        "supere el 30 % del ingreso."
    ),
    backstory=(
        "Eres analista de crédito hipotecario con experiencia en banca mexicana e "
        "INFONAVIT. Eres conservador: prefieres advertir un riesgo que dejar a una "
        "familia sobreendeudada. Siempre comparas al menos un escenario alternativo "
        "(otra tasa o plazo) contra el escenario base."
    ),
    tools=[EvaluarFinanzasTool()],
    llm=llm_local,
    allow_delegation=False,
    max_iter=6,
    verbose=True,
)

agente_coordinador = Agent(
    role="Coordinador de Recomendación de Vivienda",
    goal=(
        "Integrar las evaluaciones territorial y financiera, resolver los conflictos "
        "entre criterios y entregar al usuario un ranking final claro y justificado, "
        "con AGEB concretas donde buscar vivienda."
    ),
    backstory=(
        "Diriges un despacho de asesoría patrimonial. Tu equipo te entrega análisis "
        "parciales y tú decides. Cuando la mejor zona en calidad no es la más "
        "asequible, explicas el intercambio (trade-off) con números. Si un dato de tu "
        "equipo te parece incoherente, puedes pedir aclaración al especialista "
        "correspondiente antes de decidir."
    ),
    tools=[IntegrarRankingTool()],
    llm=llm_local,
    allow_delegation=True,    # puede consultar a los especialistas ante conflictos
    max_iter=6,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 4. Tareas y flujo de información
# ---------------------------------------------------------------------------
tarea_perfil = Task(
    description=(
        "Lee la solicitud del usuario:\n\n\"{solicitud}\"\n\n"
        "1) Extrae ingreso mensual neto, enganche disponible, presupuesto máximo, "
        "recámaras mínimas, lugar de trabajo y distancia máxima al trabajo.\n"
        "2) Convierte sus preferencias a prioridades de 1 a 5 para: nivel "
        "socioeconómico del entorno, cercanía al trabajo y oferta de vivienda. "
        "Lo no mencionado vale 3.\n"
        "3) Llama a la herramienta registrar_perfil_usuario con esos valores."
    ),
    expected_output=(
        "Reporte breve en Markdown con: (a) tabla de restricciones duras, "
        "(b) tabla de prioridades 1-5 con su peso normalizado, (c) lista de "
        "supuestos hechos por falta de información."
    ),
    agent=agente_perfil,
    output_file="salidas/1_perfil.md",
)

tarea_zonas = Task(
    description=(
        "Usa la herramienta puntuar_zonas para evaluar los municipios y alcaldías "
        "con el perfil registrado. Analiza los 5 mejor puntuados: puntaje, índice "
        "NSE, distancia al trabajo, su principal fortaleza y su principal debilidad. "
        "Indica cuántos municipios quedan fuera por distancia."
    ),
    expected_output=(
        "Reporte en Markdown con una tabla de los municipios mejor puntuados "
        "(municipio, entidad, puntaje 0-10, índice NSE, distancia km) seguida de un "
        "análisis cualitativo de los 5 mejores."
    ),
    agent=agente_zona,
    context=[tarea_perfil],                      # depende del perfil
    output_file="salidas/2_zonas.md",
)

tarea_finanzas = Task(
    description=(
        "Usa la herramienta evaluar_viabilidad_financiera dos veces: primero con el "
        "escenario base (tasa_anual=0.11, plazo_anios=20) y después con un escenario "
        "alternativo (tasa_anual=0.11, plazo_anios=15). Indica qué niveles "
        "socioeconómicos puede costear el usuario, en qué municipios el precio típico "
        "es viable y qué porcentaje de la oferta es costeable en cada uno. Compara "
        "ambos plazos."
    ),
    expected_output=(
        "Reporte en Markdown con: niveles NSE costeables, una tabla por municipio "
        "(precio típico, pago mensual a 20 años, % del ingreso, % de oferta costeable) "
        "y un párrafo que compare el escenario de 20 años contra el de 15."
    ),
    agent=agente_financiero,
    context=[tarea_perfil],                      # independiente de la evaluación territorial
    output_file="salidas/3_finanzas.md",
)

tarea_coordinacion = Task(
    description=(
        "Integra los reportes del perfil, de zonas y de finanzas.\n"
        "1) Decide los pesos zona/financiero según el perfil: si el usuario prioriza "
        "mucho la calidad del entorno usa algo como 0.6/0.4; si su margen financiero "
        "es estrecho usa 0.4/0.6; en otro caso 0.5/0.5. Justifica tu elección.\n"
        "2) Llama a integrar_ranking con esos pesos.\n"
        "3) Explica los conflictos detectados (ej. el municipio mejor evaluado en "
        "territorio que queda fuera por precio) y cómo se resolvieron.\n"
        "4) Emite la recomendación final para el usuario, incluyendo las AGEB sugeridas."
    ),
    expected_output=(
        "Recomendación final en Markdown con: (a) pesos usados y por qué, (b) tabla "
        "del ranking top 5 (posición, municipio, puntaje final, distancia, precio "
        "típico, % oferta costeable), (c) AGEB sugeridas en los 3 primeros, "
        "(d) municipios descartados con su motivo, (e) conflictos resueltos, "
        "(f) recomendación en 3-4 frases dirigida al usuario."
    ),
    agent=agente_coordinador,
    context=[tarea_perfil, tarea_zonas, tarea_finanzas],   # integra todo
    output_file="salidas/4_recomendacion_final.md",
)

# ---------------------------------------------------------------------------
# 5. Tripulación (Crew)
# ---------------------------------------------------------------------------
crew_vivienda = Crew(
    agents=[agente_perfil, agente_zona, agente_financiero, agente_coordinador],
    tasks=[tarea_perfil, tarea_zonas, tarea_finanzas, tarea_coordinacion],
    process=Process.sequential,     # perfil -> zona -> finanzas -> coordinador
    verbose=True,
    output_log_file="salidas/log_corrida.txt",   # log de razonamiento
)

# ---------------------------------------------------------------------------
# 6. Corrida de demostración
# ---------------------------------------------------------------------------
SOLICITUD_DEMO = (
    "Somos una pareja y los dos trabajamos en Santa Fe. Juntos ganamos 55,000 "
    "pesos netos al mes y tenemos 500,000 pesos ahorrados para el enganche. No "
    "queremos pagar más de 2.5 millones y necesitamos mínimo 2 recámaras. Lo más "
    "importante es no vivir a más de 20 km del trabajo y que la zona sea de buen "
    "nivel; que haya mucha oferta de vivienda nos importa menos."
)

if __name__ == "__main__":
    print(f"Modelo: {MODELO_OLLAMA} vía Ollama en {OLLAMA_URL}")
    inicio = datetime.now()
    resultado = crew_vivienda.kickoff(inputs={"solicitud": SOLICITUD_DEMO})
    print("\n" + "=" * 70 + "\nRESPUESTA FINAL CONSOLIDADA DEL CREW\n" + "=" * 70)
    print(resultado.raw)
    print(f"\nTiempo total: {datetime.now() - inicio}")
    (DIR_SALIDAS / "pizarra.json").write_text(
        json.dumps(PIZARRA, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
