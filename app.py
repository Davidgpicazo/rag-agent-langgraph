"""
Asistente Experto con Gemini, RAG y Agentes - Interfaz Streamlit
Autor: David Gómez Picazo
Proyecto Final - IA Generativa

Ejecutar con:
    streamlit run app.py
"""
import os
from typing import Annotated, TypedDict, Literal

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# =============================================================================
# Configuración general de la página
# =============================================================================
st.set_page_config(
    page_title="Asistente IA Generativa",
    page_icon="📘",
    layout="centered",
)


# =============================================================================
# Carga de la API key
# =============================================================================
load_dotenv("./API.env")
GEMINI_API_KEY = os.getenv("API_KEY1")
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
else:
    st.error("No se ha podido cargar la API Key. Revisa el fichero API.env")
    st.stop()


# =============================================================================
# Inicialización del agente (cacheada para no reconstruirla en cada interacción)
# =============================================================================
@st.cache_resource(show_spinner="Cargando base de conocimiento y agente...")
def construir_agente():
    """Construye una sola vez el vectorstore, la herramienta y el grafo."""

    # ---- Base vectorial persistente ----
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        collection_name="proyecto_ia_master",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # ---- Herramienta RAG ----
    @tool
    def buscar_en_base_conocimiento(consulta: str) -> str:
        """Busca información relevante en los PDFs del curso de IA Generativa.

        Úsala cuando el usuario pregunte sobre:
        - RAG, Retrieval-Augmented Generation, recuperación de documentos
        - LangGraph, grafos de estado, agentes, patrón ReAct
        - Embeddings, vectores, similitud coseno, ChromaDB
        - LLMs, modelos de lenguaje, tokens
        - Prompt Engineering, few-shot, output parsers
        - Memoria en agentes, MemorySaver, checkpoints
        """
        docs = retriever.invoke(consulta)
        partes = []
        for doc in docs:
            fichero = os.path.basename(doc.metadata.get("source", "?"))
            pagina = doc.metadata.get("page", "?")
            partes.append(f"[{fichero} | pág. {pagina}]\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(partes)

    # ---- System prompt ----
    SYSTEM_PROMPT = (
        "Eres un asistente experto en IA Generativa. Tienes acceso a los PDFs del curso "
        "que cubren: LLMs, RAG, LangGraph, embeddings, prompt engineering y agentes. "
        "Reglas: "
        "1. Antes de responder cualquier pregunta técnica, usa SIEMPRE la herramienta "
        "`buscar_en_base_conocimiento` para buscar en los PDFs. "
        "2. Cita siempre el documento y la página de donde sacas la información. "
        "3. Responde en español, de forma clara y directa. "
        "4. Si no encuentras la información en los PDFs, dilo y responde con lo que sabes, "
        "pero indicando que es conocimiento general. "
        "5. Recuerda lo que hemos hablado antes en la conversación y aprovéchalo."
    )

    # ---- Modelo ----
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
    tools = [buscar_en_base_conocimiento]
    llm_con_tools = llm.bind_tools(tools)

    # ---- Grafo ----
    class EstadoAgente(TypedDict):
        mensajes: Annotated[list[BaseMessage], add_messages]

    def nodo_agente(estado: EstadoAgente) -> dict:
        mensajes = [SystemMessage(content=SYSTEM_PROMPT)] + estado["mensajes"]
        return {"mensajes": [llm_con_tools.invoke(mensajes)]}

    def router_react(estado: EstadoAgente) -> Literal["tools", "__end__"]:
        ultimo = estado["mensajes"][-1]
        if hasattr(ultimo, "tool_calls") and ultimo.tool_calls:
            return "tools"
        return "__end__"

    nodo_tools = ToolNode(tools, messages_key="mensajes")

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("agente", nodo_agente)
    grafo.add_node("tools", nodo_tools)
    grafo.add_edge(START, "agente")
    grafo.add_conditional_edges(
        "agente", router_react, {"tools": "tools", "__end__": END}
    )
    grafo.add_edge("tools", "agente")

    checkpointer = MemorySaver()
    return grafo.compile(checkpointer=checkpointer), retriever


agente, retriever = construir_agente()


# =============================================================================
# Estado de la sesión
# =============================================================================
if "historial" not in st.session_state:
    st.session_state.historial = []  # lista de tuples (rol, texto, fuentes)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session"


# =============================================================================
# Función auxiliar para extraer el content del último mensaje
# =============================================================================
def extraer_texto(mensaje) -> str:
    """Devuelve el texto de un mensaje, aplanando el formato lista de Gemini."""
    content = mensaje.content
    if isinstance(content, list):
        partes = []
        for p in content:
            if isinstance(p, dict):
                partes.append(p.get("text", ""))
            else:
                partes.append(str(p))
        return "\n".join(t for t in partes if t)
    return content


# =============================================================================
# Cabecera
# =============================================================================
st.title("Asistente Experto en IA Generativa")
st.caption(
    "Agente con RAG, LangGraph y memoria sobre los PDFs del módulo. "
    "Pregúntame sobre LLMs, RAG, embeddings, prompts, agentes…"
)

with st.sidebar:
    st.subheader("Sesión")
    st.write(f"**Thread ID:** `{st.session_state.thread_id}`")
    if st.button("Nueva conversación", use_container_width=True):
        st.session_state.historial = []
        st.session_state.thread_id = f"streamlit-{os.urandom(3).hex()}"
        st.rerun()

    st.divider()
    st.subheader("Stack")
    st.write(
        "- LLM: Gemini 2.5 Flash Lite\n"
        "- Embeddings: MiniLM-L6-v2\n"
        "- Vector DB: ChromaDB\n"
        "- Agente: LangGraph (ReAct)"
    )


# =============================================================================
# Pintado del historial
# =============================================================================
for rol, texto, fuentes in st.session_state.historial:
    with st.chat_message(rol):
        st.markdown(texto)
        if fuentes:
            with st.expander("Ver fuentes consultadas"):
                for i, frag in enumerate(fuentes, 1):
                    st.markdown(f"**Fragmento {i}** — `{frag['fuente']}` · pág. {frag['pagina']}")
                    st.text(frag["texto"])


# =============================================================================
# Caja de chat
# =============================================================================
pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:
    # Pinta la pregunta del usuario
    st.session_state.historial.append(("user", pregunta, None))
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Llama al agente y muestra "escribiendo"
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los PDFs y elaborando la respuesta..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            resultado = agente.invoke(
                {"mensajes": [HumanMessage(content=pregunta)]},
                config=config,
            )
            respuesta = extraer_texto(resultado["mensajes"][-1])

            # Recuperamos también las fuentes consultadas en este turno
            fuentes = retriever.invoke(pregunta)
            fuentes_data = [
                {
                    "fuente": os.path.basename(d.metadata.get("source", "?")),
                    "pagina": d.metadata.get("page", "?"),
                    "texto": d.page_content.strip(),
                }
                for d in fuentes
            ]

        st.markdown(respuesta)
        with st.expander("Ver fuentes consultadas"):
            for i, frag in enumerate(fuentes_data, 1):
                st.markdown(f"**Fragmento {i}** — `{frag['fuente']}` · pág. {frag['pagina']}")
                st.text(frag["texto"])

    st.session_state.historial.append(("assistant", respuesta, fuentes_data))
