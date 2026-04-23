import os
import re
import json
import requests
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LEGISBEV — Copiloto Legal", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# DB Config
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://legisbev:LegisB3v2026!@shuttle.proxy.rlwy.net:45064/legisbev"
)

# OpenRouter Config
OPENROUTER_KEY = os.getenv(
    "OPENROUTER_KEY",
    "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"
)
OPENROUTER_MODEL = "qwen/qwen-2.5-72b-instruct"

SYSTEM_PROMPT = """Eres LEGISBEV, copiloto legal especializado en legislación colombiana de bebidas alcohólicas.
Responde SOLO con la información del contexto. 
Por cada afirmación, cita la norma exacta: tipo, número, año y el fragmento del texto.
Si la información no está en el contexto, di: "No encontrado en el corpus actual."
Formato: párrafo de respuesta + lista de citas al final como [Decreto 1686/2012, Art. X: "fragmento"]."""

# Color mappings for graph nodes
TIPO_COLORS = {
    "Ley": "#2563eb",         # blue
    "Decreto": "#16a34a",     # green
    "Resolución": "#ea580c",  # orange
    "Sentencia": "#6b7280",   # gray
    "Ordenanza": "#7c3aed",   # purple
}

RELACION_COLORS = {
    "modifica": "#f97316",    # orange
    "deroga": "#ef4444",      # red
    "reglamenta": "#22c55e",  # green
    "inexequible": "#991b1b", # dark red
}


def get_db():
    conn = psycopg2.connect(DB_URL)
    return conn


def extract_norm_refs(text: str) -> List[Dict]:
    """Extract norm references from a question using regex."""
    refs = []
    
    # Pattern: Decreto/Ley/Resolución NÚMERO de AÑO or NÚMERO/AÑO
    patterns = [
        r'(Decreto|Ley|Resolución|Resolución|Sentencia|Ordenanza)\s+(\d+)\s+de\s+(\d{4})',
        r'(Decreto|Ley|Resolución|Sentencia|Ordenanza)\s+(\d+)/(\d{4})',
        r'(D\.|L\.)\s*(\d+)/(\d{4})',
    ]
    
    tipo_map = {
        'D.': 'Decreto', 'L.': 'Ley'
    }
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            tipo = match[0].strip().capitalize()
            tipo = tipo_map.get(tipo, tipo)
            # Normalize tipo
            for t in ["Decreto", "Ley", "Resolución", "Sentencia", "Ordenanza"]:
                if tipo.lower() == t.lower():
                    tipo = t
                    break
            refs.append({
                "tipo": tipo,
                "numero": match[1],
                "año": int(match[2])
            })
    
    # Also extract just numbers for keyword search
    numbers = re.findall(r'\b(\d{3,4})\b', text)
    for num in numbers:
        if not any(r["numero"] == num for r in refs):
            refs.append({"tipo": None, "numero": num, "año": None})
    
    return refs


def search_norms_by_refs(conn, refs: List[Dict]) -> List[Dict]:
    """Search norms in DB by extracted references."""
    results = []
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    for ref in refs:
        if ref.get("tipo") and ref.get("numero") and ref.get("año"):
            cur.execute(
                """SELECT id, tipo, numero, año, titulo, texto_limpio as texto, estado_vigencia as estado 
                   FROM normas 
                   WHERE LOWER(tipo) = LOWER(%s) AND numero = %s AND año = %s
                   LIMIT 1""",
                (ref["tipo"], ref["numero"], ref["año"])
            )
        elif ref.get("numero") and ref.get("año"):
            cur.execute(
                """SELECT id, tipo, numero, año, titulo, texto_limpio as texto, estado_vigencia as estado 
                   FROM normas 
                   WHERE numero = %s AND año = %s
                   LIMIT 3""",
                (ref["numero"], ref["año"])
            )
        elif ref.get("numero"):
            cur.execute(
                """SELECT id, tipo, numero, año, titulo, texto_limpio as texto, estado_vigencia as estado 
                   FROM normas 
                   WHERE numero = %s
                   LIMIT 3""",
                (ref["numero"],)
            )
        else:
            continue
        
        rows = cur.fetchall()
        for row in rows:
            if not any(r["id"] == row["id"] for r in results):
                results.append(dict(row))
    
    cur.close()
    return results


def search_norms_by_keywords(conn, question: str) -> List[Dict]:
    """Search norms by keywords from the question."""
    # Extract meaningful keywords
    stopwords = {"que", "el", "la", "los", "las", "de", "del", "en", "un", "una", "es", "son",
                 "qué", "cuál", "cuáles", "puede", "pueden", "está", "están", "con", "por",
                 "para", "si", "no", "y", "o", "a", "al"}
    
    words = re.findall(r'\b[a-záéíóúñü]{4,}\b', question.lower())
    keywords = [w for w in words if w not in stopwords][:5]
    
    if not keywords:
        return []
    
    results = []
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    for keyword in keywords:
        cur.execute(
            """SELECT id, tipo, numero, año, titulo, texto_limpio as texto, estado_vigencia as estado 
               FROM normas 
               WHERE LOWER(titulo) LIKE %s OR LOWER(texto_limpio) LIKE %s
               LIMIT 3""",
            (f"%{keyword}%", f"%{keyword}%")
        )
        rows = cur.fetchall()
        for row in rows:
            if not any(r["id"] == row["id"] for r in results):
                results.append(dict(row))
        
        if len(results) >= 5:
            break
    
    cur.close()
    return results


def get_relations_for_norms(conn, norm_ids: List[int]) -> List[Dict]:
    """Get relations involving the found norms."""
    if not norm_ids:
        return []
    
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    placeholders = ','.join(['%s'] * len(norm_ids))
    cur.execute(
        f"""SELECT r.tipo_relacion, 
               n1.tipo as origen_tipo, n1.numero as origen_numero, n1.año as origen_año,
               n2.tipo as destino_tipo, n2.numero as destino_numero, n2.año as destino_año
           FROM relaciones_normativas r
           JOIN normas n1 ON r.norma_origen_id = n1.id
           JOIN normas n2 ON r.norma_destino_id = n2.id
           WHERE r.norma_origen_id IN ({placeholders}) OR r.norma_destino_id IN ({placeholders})
           LIMIT 20""",
        norm_ids + norm_ids
    )
    
    rows = cur.fetchall()
    cur.close()
    
    relations = []
    for row in rows:
        relations.append({
            "origen": f"{row['origen_tipo']} {row['origen_numero']}/{row['origen_año']}",
            "tipo": row['tipo_relacion'],
            "destino": f"{row['destino_tipo']} {row['destino_numero']}/{row['destino_año']}"
        })
    
    return relations


def call_llm(question: str, context_norms: List[Dict], relations: List[Dict]) -> str:
    """Call OpenRouter LLM with context."""
    
    context_parts = []
    
    if context_norms:
        context_parts.append("=== NORMAS RELEVANTES ===")
        for norm in context_norms[:5]:  # Limit to 5 norms to avoid token overflow
            text_preview = (norm.get("texto") or "")[:1500]
            context_parts.append(
                f"\n--- {norm['tipo']} {norm['numero']} de {norm['año']} ---"
                f"\nTítulo: {norm.get('titulo', 'Sin título')}"
                f"\nEstado: {norm.get('estado', 'desconocido')}"
                f"\nTexto: {text_preview}"
            )
    
    if relations:
        context_parts.append("\n=== RELACIONES ENTRE NORMAS ===")
        for rel in relations[:10]:
            context_parts.append(f"- {rel['origen']} [{rel['tipo']}] → {rel['destino']}")
    
    context = "\n".join(context_parts) if context_parts else "No se encontraron normas relevantes en el corpus."
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}"}
        ],
        "max_tokens": 1000,
        "temperature": 0.2
    }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://legisbev.railway.app",
        "X-Title": "LEGISBEV Copiloto Legal"
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"LLM error: {response.text}")
    
    data = response.json()
    return data["choices"][0]["message"]["content"]


def extract_cited_norms_from_response(response_text: str, db_norms: List[Dict]) -> List[Dict]:
    """Extract cited norms from the LLM response."""
    cited = []
    
    # Look for patterns like [Decreto 1686/2012, ...] or Decreto 1686/2012
    patterns = [
        r'(Decreto|Ley|Resolución|Sentencia|Ordenanza)\s+(\d+)/(\d{4})',
        r'(Decreto|Ley|Resolución|Sentencia|Ordenanza)\s+(\d+)\s+de\s+(\d{4})',
    ]
    
    found_refs = set()
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        for match in matches:
            key = f"{match[0].capitalize()}_{match[1]}_{match[2]}"
            if key not in found_refs:
                found_refs.add(key)
                # Find estado from DB norms
                estado = "desconocido"
                for norm in db_norms:
                    if (norm["numero"] == match[1] and 
                        str(norm["año"]) == match[2] and
                        norm["tipo"].lower() == match[0].lower()):
                        estado = norm.get("estado", "desconocido")
                        break
                
                cited.append({
                    "tipo": match[0].capitalize(),
                    "numero": match[1],
                    "año": int(match[2]),
                    "estado": estado
                })
    
    return cited


def extract_fragments(response_text: str) -> List[str]:
    """Extract quoted fragments from LLM response."""
    fragments = re.findall(r'"([^"]{20,300})"', response_text)
    return fragments[:5]


# ─── Models ───────────────────────────────────────────────────────────────────

class ConsultaRequest(BaseModel):
    pregunta: str


class NormaCitada(BaseModel):
    tipo: str
    numero: str
    año: int
    estado: str


class Relacion(BaseModel):
    origen: str
    tipo: str
    destino: str


class ConsultaResponse(BaseModel):
    respuesta: str
    normas_citadas: List[NormaCitada]
    relaciones: List[Relacion]
    alerta: Optional[str]
    fragmentos: List[str]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM normas")
        normas_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM relaciones_normativas")
        relaciones_count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"status": "ok", "normas": normas_count, "relaciones": relaciones_count}
    except Exception as e:
        return {"status": "degraded", "error": str(e), "normas": 0, "relaciones": 0}


@app.post("/api/consulta", response_model=ConsultaResponse)
def consulta(req: ConsultaRequest):
    pregunta = req.pregunta.strip()
    if not pregunta:
        raise HTTPException(status_code=400, detail="Pregunta vacía")
    
    try:
        conn = get_db()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB connection failed: {str(e)}")
    
    try:
        # 1. Extract norm references from question
        refs = extract_norm_refs(pregunta)
        
        # 2. Search norms by extracted references
        norms = search_norms_by_refs(conn, refs)
        
        # 3. If not enough results, search by keywords
        if len(norms) < 2:
            keyword_norms = search_norms_by_keywords(conn, pregunta)
            for n in keyword_norms:
                if not any(x["id"] == n["id"] for x in norms):
                    norms.append(n)
        
        # 4. Get relations for found norms
        norm_ids = [n["id"] for n in norms]
        relations = get_relations_for_norms(conn, norm_ids)
        
        # 5. Call LLM with context
        llm_response = call_llm(pregunta, norms, relations)
        
        # 6. Extract cited norms from LLM response
        cited_norms = extract_cited_norms_from_response(llm_response, norms)
        
        # 7. Check for alerts
        alert = None
        for norm in cited_norms:
            if norm["estado"] in ["derogada", "inexequible", "suspendida"]:
                alert = f"⚠️ Atención: {norm['tipo']} {norm['numero']}/{norm['año']} está {norm['estado']}. Verifique la norma vigente aplicable."
                break
        
        # 8. Extract fragments
        fragments = extract_fragments(llm_response)
        
        conn.close()
        
        return ConsultaResponse(
            respuesta=llm_response,
            normas_citadas=[NormaCitada(**n) for n in cited_norms],
            relaciones=[Relacion(**r) for r in relations[:10]],
            alerta=alert,
            fragmentos=fragments
        )
    
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/grafo")
def get_grafo():
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Use grafo_relaciones view which has pre-joined data
        cur.execute(
            """SELECT DISTINCT origen_tipo as tipo, origen_numero as numero, origen_año as año,
                  origen_estado as estado_vigencia
               FROM grafo_relaciones
               UNION
               SELECT DISTINCT destino_tipo, destino_numero, destino_año, destino_estado
               FROM grafo_relaciones
               LIMIT 100"""
        )
        raw_nodes = cur.fetchall()
        # Also get any normas not in graph
        cur.execute(
            """SELECT id, tipo, numero, año, estado_vigencia, titulo, resumen FROM normas 
               WHERE numero IS NOT NULL AND año IS NOT NULL LIMIT 100"""
        )
        normas = cur.fetchall()
        
        # Get all relations
        cur.execute(
            """SELECT tipo_relacion, origen_numero, origen_año, origen_tipo,
                      destino_numero, destino_año, destino_tipo
               FROM grafo_relaciones LIMIT 200"""
        )
        relaciones = cur.fetchall()
        
        cur.close()
        conn.close()
        
        # Build nodes from normas table
        nodes = []
        seen_node_ids = set()
        for norm in normas:
            if not norm['numero'] or not norm['año']:
                continue
            node_id = f"{norm['tipo'][0].lower()}{norm['numero']}"
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            color = TIPO_COLORS.get(norm['tipo'], "#6b7280")
            # Preferir resumen (más corto y específico) sobre titulo
            resumen_txt = (norm.get('resumen') or norm.get('titulo') or '').strip()[:120]
            nodes.append({
                "id": node_id,
                "label": f"{norm['tipo'][0]}.{norm['numero']}/{norm['año']}",
                "tipo": norm['tipo'],
                "estado": norm.get('estado_vigencia', 'vigente'),
                "color": color,
                "titulo": resumen_txt
            })
        
        # Build edges — only include edges where BOTH nodes exist
        edges = []
        seen_edges = set()
        for rel in relaciones:
            src_id = f"{rel['origen_tipo'][0].lower()}{rel['origen_numero']}"
            tgt_id = f"{rel['destino_tipo'][0].lower()}{rel['destino_numero']}"
            # Skip if either node is missing
            if src_id not in seen_node_ids or tgt_id not in seen_node_ids:
                continue
            edge_key = f"{src_id}-{rel['tipo_relacion']}-{tgt_id}"
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({
                    "source": src_id,
                    "target": tgt_id,
                    "label": rel['tipo_relacion'],
                    "color": RELACION_COLORS.get(rel['tipo_relacion'], "#9ca3af")
                })
        
        return {"nodes": nodes, "edges": edges}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/norma/{tipo}/{numero}/{año}")
def get_norma(tipo: str, numero: str, año: int):
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute(
            """SELECT id, tipo, numero, año, titulo, texto_limpio as texto, estado_vigencia as estado
               FROM normas
               WHERE LOWER(tipo) = LOWER(%s) AND numero = %s AND año = %s
               LIMIT 1""",
            (tipo, numero, año)
        )
        norm = cur.fetchone()
        
        if not norm:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Norma no encontrada")
        
        norm = dict(norm)
        
        # Get relations
        cur.execute(
            """SELECT r.tipo_relacion,
               n1.tipo as origen_tipo, n1.numero as origen_numero, n1.año as origen_año,
               n2.tipo as destino_tipo, n2.numero as destino_numero, n2.año as destino_año
           FROM relaciones_normativas r
           JOIN normas n1 ON r.norma_origen_id = n1.id
           JOIN normas n2 ON r.norma_destino_id = n2.id
           WHERE r.norma_origen_id = %s OR r.norma_destino_id = %s""",
            (norm["id"], norm["id"])
        )
        
        relations = []
        for row in cur.fetchall():
            relations.append({
                "origen": f"{row['origen_tipo']} {row['origen_numero']}/{row['origen_año']}",
                "tipo": row['tipo_relacion'],
                "destino": f"{row['destino_tipo']} {row['destino_numero']}/{row['destino_año']}"
            })
        
        cur.close()
        conn.close()
        
        norm["relaciones"] = relations
        return norm
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
