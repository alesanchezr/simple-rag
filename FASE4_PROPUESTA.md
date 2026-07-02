# Fase 4 — Propuesta: harness alrededor del agente

> Documento de propuesta (no implementación). Registra la idea, las piezas del
> harness, la consigna para los alumnos y el enfoque técnico sugerido.
>
> Inspirado en:
>
> - Anthropic — [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
> - LangChain — [The anatomy of an agent harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness)

## Contexto: dónde estamos

- **Fase 1** — RAG puro (`rag.py`): embed → retrieve → generate.
- **Fase 2** — LangGraph como orquestador: grafo fijo `retrieve → generate`.
- **Fase 3** — Un **agente** en un nodo, con tools, que decide el camino
(loop `agent ⇄ tools`). Modela un asistente de *customer support*.
- **Fase 4** — Construir un **harness** alrededor de ese agente: la capa que le
da **durabilidad y control** a lo largo del tiempo.



## La idea central

Un **harness** es *todo el código que rodea al modelo* (el que **no** es el
modelo). El agente de la fase 3 sabe razonar y usar tools, pero es *stateless*:
cada `POST /ask` empieza de cero, sin memoria, y el ciclo `agent ⇄ tools` podría
girar sin fin.

Un asistente de support real atiende una **conversación** de varios turnos, y
debe mantenerse bajo control. El harness es lo que hace posible eso.

El "clic" pedagógico: **el agente por sí solo no basta**. La inteligencia vive
en el modelo; la *supervivencia en el tiempo* vive en el código de alrededor.

## Qué añade el harness (piezas mínimas)

Con el vocabulario de los artículos:


| Capa del harness             | Qué hacemos (mínimo)                                                                                   | Concepto del artículo                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| **Persistencia de sesión**   | `session_id` por conversación; el historial se guarda en un JSON en disco y se resume al volver.       | *Durable state / memory across sessions* |
| **Compactación de contexto** | Cuando el historial crece, se resume lo viejo en un "summary" y se descarta el detalle.                | *Compaction* (pieza central del harness) |
| **Guardarraíl del loop**     | Tope de iteraciones `agent ⇄ tools` por turno; si se excede → escala (crea ticket) en vez de colgarse. | *Safety rails / control loop*            |
| **Observabilidad**           | Trazas con **LangSmith**; logs a archivo como opción local adicional.                                  | *Observation / self-verification*        |




## Cómo se ve (el harness envuelve al agente)

Lo clave: **casi nada de esto vive dentro del grafo de la fase 3**. El grafo
(el agente) es el núcleo; el harness es el sobre que lo rodea. Eso *es* la
lección.

```
  request (session_id, mensaje)
        │
        ▼
┌────────────────── HARNESS (el código que no es el modelo) ──────────────────┐
│                                                                              │
│  1. cargar sesión  ──────────────►  (JSON en disco)                          │
│        │                                                                     │
│        ▼                                                                     │
│  2. ¿historial largo? ──sí──►  compactar (resumir lo viejo)                  │
│        │                                                                     │
│        ▼                                                                     │
│  3. ejecutar el AGENTE (grafo fase 3) con tope de iteraciones                │
│        ┌──────────────────────────────────────────┐                         │
│        │      agent ⇄ tools   (loop, máx. N)       │──► si excede N: escala  │
│        └──────────────────────────────────────────┘                         │
│        │                                                                     │
│        ▼                                                                     │
│  4. guardar sesión (JSON)  +  traza (LangSmith / log)                        │
│        │                                                                     │
└────────┼─────────────────────────────────────────────────────────────────────┘
         ▼
     respuesta
```



## Consigna para los alumnos

Envolver el agente de la fase 3 con un harness que:

1. **Persista la conversación**: aceptar un `session_id`, cargar el historial
  previo, ejecutar el turno y volver a guardarlo. Dos requests con el mismo
   `session_id` continúan la misma conversación.
2. **Compacte el contexto**: al superar un umbral de mensajes, resumir lo viejo
  en un único mensaje de resumen y seguir desde ahí.
3. **Controle el loop**: imponer un tope de iteraciones `agent ⇄ tools`; al
  alcanzarlo, escalar el caso (ticket) en vez de seguir girando.
4. **Sea observable**: trazar la ejecución con LangSmith (y/o un log local).

**Reto de reflexión:** ¿qué pasa si se quita cada pieza? Sin persistencia el
asistente olvida todo entre mensajes; sin compactación el contexto crece hasta
degradarse; sin guardarraíl un caso ambiguo puede hacer girar el loop; sin
observabilidad no hay forma de saber *por qué* respondió lo que respondió.

## Enfoque técnico (crudo, coherente con las fases anteriores)

Sin abstracciones ocultas; el harness se ve y se toca:

- **Persistencia** = archivos JSON por sesión (`sessions/<id>.json`). Los alumnos
ven la memoria como un archivo, no como magia. (LangGraph trae un *checkpointer*
que hace esto; se puede mencionar como alternativa opcional, pero el JSON a mano
enseña más.)
- **Compactación** = una llamada extra a `ollama.chat` que resume los mensajes
viejos y los reemplaza por un único mensaje de rol `system` con el resumen.
- **Guardarraíl** = un contador en el loop (o el `recursion_limit` que ya expone
LangGraph); al toparlo, el harness fuerza `create_support_ticket` y responde
"te escalé el caso".



### Observabilidad con LangSmith

- El grafo ya es un `StateGraph` de LangGraph, así que LangSmith traza su
**estructura** (nodos, aristas, estado) con solo activar variables de entorno:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...        # requiere cuenta en LangSmith
LANGSMITH_PROJECT=simple-rag-support
```

  Es **opt-in**: si no se activan, el grafo corre igual.

- Las llamadas **crudas a** `ollama.chat` no aparecen como spans de LLM por sí
solas: se decoran con `@traceable` (SDK `langsmith`) para que se vean con sus
inputs/outputs. Es una línea por función y es didáctico.
- **Log local (opción adicional)** = *append* a un archivo por sesión con cada
turno y tool usada. Sirve de fallback offline / sin cuenta, y contrasta
"plataforma gestionada vs. DIY".
- Nueva dependencia: `langsmith` en `requirements.txt`.
