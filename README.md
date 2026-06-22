# 🤖 Asistente Experto en IA Generativa — RAG + LangGraph

Asistente conversacional basado en **Retrieval-Augmented Generation (RAG)** que responde preguntas sobre conceptos de IA Generativa (LLMs, embeddings, RAG, LangGraph, prompt engineering y despliegue de agentes), fundamentando cada respuesta en su propia base de conocimiento documental y citando la fuente.

Implementa un agente **ReAct** sobre LangGraph con memoria conversacional, e incluye una interfaz web en Streamlit.

> Proyecto desarrollado en el módulo de IA Generativa del Máster en Data Science & Inteligencia Artificial.

---

## 📋 Dominio y base de conocimiento

El agente es experto en el temario de IA Generativa del curso. Su base de conocimiento se construye a partir de 4 documentos técnicos:

| Documento | Contenido |
|---|---|
| `Introduccion_LLMs.pdf` | LLMs, arquitectura Transformer, tokens, embeddings, integración con APIs |
| `Ingenieria_de_Prompts.pdf` | Prompt templates, few-shot, Chain of Thought, ReAct, salida estructurada |
| `Orquestacion_Despliegue_Agentes.pdf` | Orquestación de agentes, memoria, despliegue y observabilidad |
| `RAG_langgraph.pdf` | Arquitectura RAG, chunking, grafos de estado y evaluación |

---

## 🏗️ Arquitectura

```
Usuario
  │
  ▼
Agente LangGraph (patrón ReAct — StateGraph manual)
  │         │
  │         ▼  (si el LLM decide invocar la herramienta)
  │    buscar_en_base_conocimiento (@tool)
  │         │
  │         ▼
  │    ChromaDB + HuggingFace Embeddings
  │    (recupera los 3 fragmentos más relevantes)
  │         │
  │         ▼
  └── LLM  →  Respuesta final con fuentes citadas
```

El flujo ReAct se controla con un router condicional: el agente razona, decide si necesita
recuperar contexto, invoca la herramienta de búsqueda si procede, y genera la respuesta final.
La **memoria de conversación** se gestiona con `MemorySaver`, identificando cada sesión por un `thread_id`.

---

## 🛠️ Stack tecnológico

| Componente | Tecnología | Justificación |
|---|---|---|
| Framework del agente | LangGraph `StateGraph` | Permite controlar explícitamente el flujo ReAct y las bifurcaciones |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Modelo local, gratuito y sin dependencia de API externa para vectorizar |
| Base vectorial | ChromaDB | Ligera, local y persistente, sin servicios externos |
| Memoria | `MemorySaver` + `thread_id` | Memoria conversacional por sesión |
| Loader | `PyPDFDirectoryLoader` | Carga masiva de PDFs conservando metadatos de página |
| Chunking | `RecursiveCharacterTextSplitter` | Respeta separadores naturales (párrafos antes que líneas) |
| Interfaz | Streamlit | Chat web con historial y fuentes consultadas desplegables |

---

## ⚙️ Instalación y ejecución

### 1. Clonar e instalar dependencias

```bash
git clone https://github.com/Davidgpicazo/[NOMBRE-DEL-REPO].git
cd [NOMBRE-DEL-REPO]

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configurar la API key

Crea un archivo `API.env` en la raíz del proyecto (a partir de `API.env.example`):

```
API_KEY1=tu_clave_aqui
```

> ⚠️ El archivo `API.env` está excluido del repositorio mediante `.gitignore`. Nunca subas tu clave real.

### 3. Ejecutar

**Notebook** (entregable principal, con ejemplos documentados):

```bash
jupyter notebook DavidGomez_Proyecto_AI.ipynb
```

La primera ejecución del notebook indexa los PDFs en ChromaDB y persiste la base vectorial en `./chroma_db/`; las siguientes la reutilizan.

**Interfaz web** (requiere haber ejecutado antes el notebook para generar `./chroma_db/`):

```bash
streamlit run app.py
```

---

## 🎯 System prompt y decisiones de diseño

```
Eres un asistente experto en IA Generativa. Tienes acceso a los PDFs del curso que cubren:
LLMs, RAG, LangGraph, embeddings, prompt engineering y agentes.

Reglas:
1. Antes de responder cualquier pregunta técnica, usa SIEMPRE la herramienta
   `buscar_en_base_conocimiento` para buscar en los PDFs.
2. Cita siempre el documento y la página de donde sacas la información.
3. Responde en español, de forma clara y directa.
4. Si no encuentras la información en los PDFs, dilo y responde con lo que sabes,
   pero indicando que es conocimiento general.
5. Recuerda lo que hemos hablado antes en la conversación y aprovéchalo.
```

- **Uso obligatorio de la herramienta:** sin esta instrucción el modelo respondería desde su conocimiento de entrenamiento sin consultar la base vectorial; así se garantiza que cada respuesta técnica se fundamenta en el material indexado.
- **Citar fuentes:** aporta transparencia y permite verificar la información en el documento original.
- **Comportamiento ante falta de información:** reduce alucinaciones; el modelo admite el límite en lugar de inventar.
- **Coherencia conversacional:** activa el uso del historial guardado por `MemorySaver` para resolver preguntas de seguimiento.

---

## 📁 Estructura del proyecto

```
.
├── DavidGomez_Proyecto_AI.ipynb   # Notebook principal (entregable)
├── app.py                         # Interfaz web Streamlit
├── requirements.txt               # Dependencias
├── API.env.example                # Plantilla de la API key (sin la clave real)
├── .gitignore
├── README.md
└── PDFs/                          # Base documental
    ├── Introduccion_LLMs.pdf
    ├── Ingenieria_de_Prompts.pdf
    ├── Orquestacion_Despliegue_Agentes.pdf
    └── RAG_langgraph.pdf
```

---

<sub>Desarrollado por David Gómez · Madrid, España</sub>
