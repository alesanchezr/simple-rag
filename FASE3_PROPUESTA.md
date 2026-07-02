# Fase 3 — Propuesta: workflow con agente

> Documento de propuesta (no implementación). Registra la idea, las tools,
> la consigna para los alumnos y el enfoque técnico sugerido.

## Contexto: dónde estamos

- **Fase 1** — RAG puro (`rag.py`): embed → retrieve → generate.
- **Fase 2** — LangGraph como orquestador (`langgraph_layer.py`): grafo fijo
  `START → retrieve → generate → END`.
- **Fase 3** — Convertir el workflow fijo en uno con un **agente** en un nodo,
  que maneja al menos una **tool**. Modela un asistente de *customer support*
  (mínima expresión, fines educativos).

## La idea central

Pasar de un **workflow** (nosotros decidimos el camino, siempre igual) a un
**agente** (el LLM decide el camino: qué tool usar, cuántas veces y cuándo
terminar).

El "clic" pedagógico: **nadie programa el if/else**. El agente elige en tiempo
de ejecución si busca en la base de conocimiento, consulta un dato o escala el
caso. Eso añade potencia (autonomía) a cambio de menos predecibilidad.

## Tools propuestas (todas mockeadas)

| Tool | Qué hace (mock) | Aporte didáctico |
|---|---|---|
| `search_knowledge_base(query)` | Envuelve el `retrieve` actual del RAG | El RAG pasa a ser **una tool más**, no el flujo entero. Reutiliza la fase 2. |
| `check_course_schedule(course)` | Devuelve un horario de un dict fijo | **Dato estructurado** que el RAG no sabe responder → obliga a discriminar. |
| `create_support_ticket(email, issue)` | Devuelve un ID falso (`TICKET-1234`) | Tool con **efecto secundario** (una acción, no solo lectura) → escalar. |

Basta con **una** tool para cumplir; se recomiendan las tres porque la tensión
entre ellas (buscar / consultar / escalar) es donde el agente tiene que *elegir*.

## Consigna para los alumnos

Convertir/añadir un nodo **agente** con acceso a herramientas. El agente debe:

1. Recibir la consulta y **decidir por sí mismo** qué tool(s) usar (o ninguna).
2. Manejar **al menos una tool** (se recomiendan las tres).
3. Poder **encadenar** llamadas (p. ej. buscar en la KB y, si no hay respuesta,
   abrir un ticket).
4. Terminar devolviendo la respuesta final y, como extra, **qué tools usó**
   (para que el flujo sea observable).

**Reto de reflexión:** comparar con el workflow de la fase 2. ¿En qué preguntas
acierta más el agente? ¿En cuáles se "va por las ramas"? ¿Qué cambia si se le
quita la tool de tickets?

## Cómo queda el grafo

Dos elementos nuevos frente a la fase 2: una **arista condicional** (el LLM
decide el siguiente paso) y un **ciclo** (vuelve a pensar tras usar una tool).

```
                    ┌──────────────────────────────────────┐
                    │                                       │
                    ▼                                       │
   START ──────► ┌───────┐   ¿el agente pidió una tool?     │
                 │ agent │───────────────┐                  │
                 └───────┘               │ sí               │
                    │                    ▼                  │
                    │ no            ┌─────────┐             │
                    │ (ya responde) │  tools  │             │
                    │               │ ┌─────┐ │             │
                    ▼               │ │ KB  │ │─────────────┘
                  END               │ │sched│ │  (devuelve el
                                    │ │ticket│ │   resultado y
                                    │ └─────┘ │   el agente
                                    └─────────┘   vuelve a pensar)
```

Frente a la fase 2, que era una recta sin decisiones ni vueltas:

```
   START ──► retrieve ──► generate ──► END      (fase 2: camino fijo)
```

## Enfoque técnico sugerido (crudo, con `StateGraph` + `ollama`)

Mantener la misma filosofía de las fases 1 y 2: cliente **`ollama` crudo** y
`langgraph` para el grafo, **sin sumar LangChain**. Ollama ya soporta tool
calling de forma nativa, así que todo el agente se arma a mano. Esto es
justo lo pedagógico: los alumnos **ven** cómo el agente encadena tool → volver
a pensar, sin que ningún helper lo oculte.

Piezas a usar:

- **Estado**: un `TypedDict` con `messages: list` (la conversación, que ahora
  acumula también las llamadas a tools y sus resultados; reemplaza al
  `context`/`answer` planos de la fase 2).
- **Tools**: funciones Python normales + sus **esquemas JSON** (formato de
  función que espera Ollama) y un dict `nombre → función` para despacharlas.
  `search_knowledge_base` reutiliza el `retrieve` de `rag.py`.
- **Nodo `agent`**: llama a `ollama.chat(model=..., messages=..., tools=schemas)`.
  La respuesta (`message`) puede o no traer `tool_calls`. Se añade al estado.
- **Nodo `tools`**: lee `tool_calls` del último mensaje, ejecuta cada
  función vía el dict de despacho y añade cada resultado como un mensaje de
  rol `tool`.
- **Arista condicional**: una función que mira el último mensaje →
  si trae `tool_calls` va a `"tools"`, si no va a `END`.
- **Arista de vuelta** `tools → agent` (esto crea el ciclo).

Wiring del grafo (esquemático, no código final):

```
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)   # llama a ollama.chat(tools=schemas)
builder.add_node("tools", tools_node)   # dispatcher de tools
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route)  # route: ¿hay tool_calls? tools : END
builder.add_edge("tools", "agent")      # ciclo
graph = builder.compile()
```

Donde `route(state)` es simplemente:

```
last = state["messages"][-1]
return "tools" if last.get("tool_calls") else END
```

Nota: llama3 vía Ollama soporta tool calling. Conviene probar el modelo local
temprano; si diera problemas, se puede cambiar a otro modelo de Ollama con buen
soporte de tools sin tocar la estructura del grafo.

## Alcance (mantenerlo mínimo)

- Todo mockeado; sin integraciones reales (ni email, ni tickets de verdad).
- 2–3 tools máximo.
- Reutilizar `retrieve` de `rag.py` dentro de `search_knowledge_base`.