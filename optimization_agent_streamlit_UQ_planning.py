# simple_opt_multiagent.py
# One-click Streamlit app for optimization via GPT with LLM-level UQ (epistemic)
# Pipeline: CodeWriter (N samples + UQ) → Execute (sandbox) → Auto-fix loop → Interpreter
# RESULT contract: {objective_value: float, solution: list, feasible: bool, details: dict}
# Requires: pip install streamlit numpy scipy langchain-openai python-dotenv openai

import os, io, sys, json, traceback, itertools
from typing import Any, Dict, List, Tuple
import difflib
import numpy as np
import streamlit as st
from scipy.optimize import minimize, differential_evolution
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Optional semantic embeddings
try:
    from openai import OpenAI  # openai>=1.x
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# ---------------- Boot ----------------
load_dotenv()
st.set_page_config(page_title="One-Click Optimization (GPT + UQ)", layout="wide")

# ---------------- Sidebar (perf & quality controls) ----------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model_name = st.text_input(
        "Chat Model (e.g., gpt-5 or gpt-4.1-mini)",
        value=os.environ.get("OPENAI_MODEL_NAME", "gpt-5"),
        help="Your OpenAI chat model / deployment alias."
    )

    temperature = st.slider("LLM temperature (generation)", 0.0, 1.0, 0.2, 0.1,
                            help="Lower = more deterministic code.")
    repair_attempts = st.slider("Auto-repair attempts", 0, 5, 2)
    seed = st.number_input("Random seed for solvers", value=1, step=1)

    prefer_de = st.checkbox("Prefer Differential Evolution when unclear", value=False)
    prefer_trust_constr = st.checkbox("Prefer trust-constr for smooth constraints", value=True)

    st.divider()
    st.subheader("Uncertainty (Epistemic) Controls")
    n_writer_samples = st.slider("Writer samples (for UQ)", 1, 5, 3,
                                 help="Generate multiple candidate codes to estimate epistemic uncertainty.")
    sample_temperature = st.slider("Writer sampling temperature", 0.1, 1.0, max(0.3, float(temperature)), 0.1,
                                   help="Used only for multi-sample UQ runs (CodeWriter).")
    use_semantic = st.checkbox("Use semantic embeddings for similarity (recommended)", value=True)
    embed_model = st.text_input("Embedding model", value="text-embedding-3-small",
                                help="Used when semantic similarity is enabled.")
    st.caption("Epistemic UQ = 1 - mean pairwise similarity among code samples.")

    st.caption("Quality defaults tuned for robust generation & convergence.")

# ---------------- LLM / Embedding clients (cached) ----------------
@st.cache_resource(show_spinner=False)
def _cached_llm(model: str, temp: float) -> ChatOpenAI:
    # Higher timeout for code-gen; built-in retries
    return ChatOpenAI(model=model, temperature=temp, max_retries=3, timeout=300)

def llm_client(temp_override: float = None) -> ChatOpenAI:
    t = temperature if temp_override is None else temp_override
    return _cached_llm(model_name, t)

@st.cache_resource(show_spinner=False)
def _cached_openai_client() -> Any:
    if not _HAS_OPENAI:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------- Helpers ----------------
def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text.strip()
    try:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    except Exception:
        try:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        except Exception:
            return text.strip()

def _static_sanity_patch(py: str) -> str:
    """Ensure ASSUMPTIONS/RESULT/solve() exist and solve() is called."""
    if "ASSUMPTIONS" not in py:
        py = 'ASSUMPTIONS = ["Default assumptions applied"]\n' + py
    if "def solve(" not in py:
        py += (
            "\n\ndef solve():\n"
            "    global RESULT\n"
            "    RESULT = {\n"
            "        'objective_value': float('nan'),\n"
            "        'solution': [],\n"
            "        'feasible': False,\n"
            "        'details': {'message': 'solve() placeholder'}\n"
            "    }\n"
            "    return RESULT\n"
        )
    if "RESULT" not in py:
        py += "\nRESULT = None\n"
    if "solve()" not in py or "global RESULT" not in py:
        py += "\n_ = solve()\n"
    return py

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a); bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))

def _lexical_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def _semantic_similarity_matrix(texts: List[str], model: str) -> np.ndarray:
    """Return NxN cosine similarity matrix using OpenAI embeddings; falls back to lexical if unavailable."""
    N = len(texts)
    M = np.eye(N, dtype=float)
    client = _cached_openai_client()
    if use_semantic and _HAS_OPENAI and client is not None:
        try:
            # Batched embedding call for efficiency
            embs = client.embeddings.create(input=texts, model=model).data
            vecs = [np.array(row.embedding, dtype=float) for row in embs]
            for i, j in itertools.combinations(range(N), 2):
                s = _cosine(vecs[i], vecs[j])
                M[i, j] = M[j, i] = s
            return M
        except Exception:
            pass  # fall through to lexical
    # Fallback lexical
    for i, j in itertools.combinations(range(N), 2):
        s = _lexical_similarity(texts[i], texts[j])
        M[i, j] = M[j, i] = s
    return M

def _uq_from_matrix(S: np.ndarray) -> Tuple[float, float]:
    """Return (mean_similarity, epistemic_uncertainty) from NxN similarity matrix."""
    N = S.shape[0]
    if N <= 1:
        return 1.0, 0.0
    # Off-diagonal mean
    tri = S[np.triu_indices(N, k=1)]
    mean_sim = float(np.mean(tri)) if tri.size else 1.0
    return mean_sim, 1.0 - mean_sim

def _medoid_index(S: np.ndarray) -> int:
    """Return index of medoid (row with max average similarity to others)."""
    N = S.shape[0]
    if N == 1:
        return 0
    scores = S.sum(axis=1) - np.diag(S)  # sum of similarities to others
    return int(np.argmax(scores))

# ---------------- Prompts ----------------
WRITER_SYS = """You are a senior optimization engineer and Python expert.
Write ONE self-contained Python program (no Markdown, no backticks) to SOLVE the user's optimization problem.

MANDATORY rules:
- Only numpy as np and scipy.optimize (minimize, differential_evolution). No other libs.
- At top: define ASSUMPTIONS: list[str] describing numeric bounds/constraints you inferred.
- Define objective and constraints; add explicit bounds; keep variables x0, x1, ...
- Implement solve() that returns dict RESULT with EXACT keys:
  RESULT = {
    "objective_value": float,
    "solution": list,
    "feasible": bool,
    "details": dict  # include method, success, status, message, nit/nfev when available
  }
- ALWAYS set global RESULT by calling solve() at the end; on failure set feasible=False, objective_value=nan.
- Print a compact 10-line summary (no plots, no files, no network).

Robustness:
- Use getattr(res, "field", res.__dict__.get("field", default)) when reading OptimizeResult.
- Do not shadow builtins (sum, list, getattr, etc.). If needed define safe_getattr().
- Use np.random.default_rng(SEED) if randomness; SEED is provided as a module variable.
"""

WRITER_USER_TMPL = """User problem (free text):
{problem}

Guidance:
- If numbers missing, pick sane bounds; prefer convex or gently non-convex forms.
- If constraints unclear: add mild penalties or simple inequality constraints; justify in ASSUMPTIONS.
- Prefer {preferred_solver} based on problem type.
- Use np.random.default_rng({seed}) for reproducibility if randomness is used.
"""

FIXER_SYS = """You are repairing a Python script that failed at runtime. Return ONLY corrected Python code (no Markdown).
Rules:
- Only numpy/scipy/stdlib.
- Keep overall structure/variables.
- ASSUMPTIONS list must exist.
- solve() must set and return RESULT with keys: objective_value (float), solution (list), feasible (bool), details (dict).
- ALWAYS call solve() at end to set global RESULT, even on failure (feasible=False, objective_value=nan).
- Compact prints (≤10 lines). Use robust getattr/.__dict__.get for OptimizeResult fields.
- Do not shadow Python builtins.
"""

INTERPRET_SYS = """You are a concise technical analyst. Summarize the optimization outcome for a non-expert engineer:
goal, key assumptions, solver used, solution & objective, feasibility, constraint activity, caveats, and next tweaks."""

INTERPRET_USER_TMPL = """Problem:
{problem}

Assumptions (from code): {assumptions}
RESULT (from execution): {result}

Program stdout:
---
{stdout}
---

Write 8–12 succinct bullet points."""

# ---------------- Agents ----------------
def code_writer_once(problem_text: str, preferred_solver: str, seed_val: int, temp: float) -> str:
    msgs = [
        SystemMessage(content=WRITER_SYS),
        HumanMessage(content=WRITER_USER_TMPL.format(
            problem=problem_text.strip(),
            preferred_solver=preferred_solver,
            seed=seed_val
        ))
    ]
    out = llm_client(temp_override=temp).invoke(msgs).content
    return _strip_code_fences(out)

def code_writer_with_uq(problem_text: str, preferred_solver: str, seed_val: int,
                        n_samples: int, samp_temp: float) -> Tuple[str, Dict[str, Any]]:
    """
    Generate N candidate codes, compute similarity-based epistemic UQ,
    pick medoid code as representative.
    Returns (selected_code, uq_info)
    """
    if n_samples <= 1:
        code = code_writer_once(problem_text, preferred_solver, seed_val, temp=temperature)
        return code, {"mean_similarity": 1.0, "epistemic_uq": 0.0,
                      "pairwise": [[1.0]], "selected_index": 0, "samples": [code]}

    samples: List[str] = []
    for _ in range(n_samples):
        samples.append(code_writer_once(problem_text, preferred_solver, seed_val, temp=samp_temp))

    # Similarity matrix (semantic preferred; lexical fallback)
    S = _semantic_similarity_matrix(samples, embed_model)
    mean_sim, ep_uq = _uq_from_matrix(S)
    med_idx = _medoid_index(S)

    return samples[med_idx], {
        "mean_similarity": mean_sim,
        "epistemic_uq": ep_uq,
        "pairwise": S.tolist(),
        "selected_index": med_idx,
        "samples": samples,
    }

def fixer_agent(bad_code: str, error_text: str) -> str:
    msgs = [
        SystemMessage(content=FIXER_SYS),
        HumanMessage(content=f"Runtime error:\n{error_text}\n\nOriginal code:\n{bad_code}")
    ]
    out = llm_client().invoke(msgs).content
    return _strip_code_fences(out)

def interpreter_agent(problem_text: str, assumptions, result_obj, stdout_text: str, uq_info: Dict[str, Any]) -> str:
    # Attach UQ to result (non-breaking)
    result_aug = dict(result_obj or {})
    result_aug["uncertainty"] = {
        "writer_mean_similarity": float(uq_info.get("mean_similarity", 1.0)),
        "writer_epistemic_uq": float(uq_info.get("epistemic_uq", 0.0))
    }
    msgs = [
        SystemMessage(content=INTERPRET_SYS),
        HumanMessage(content=INTERPRET_USER_TMPL.format(
            problem=problem_text.strip(),
            assumptions=json.dumps(assumptions, ensure_ascii=False),
            result=json.dumps(result_aug, ensure_ascii=False),
            stdout=stdout_text[:8000]
        ))
    ]
    return llm_client().invoke(msgs).content.strip()

# ---------------- Sandbox executor ----------------
def safe_exec_user_code(code: str) -> Dict[str, Any]:
    """Execute model code in a restricted environment; capture stdout; return RESULT + ASSUMPTIONS."""
    safe_builtins = {
        "__import__": __import__,
        "bool": bool, "int": int, "float": float, "complex": complex,
        "str": str, "bytes": bytes, "dict": dict, "list": list, "tuple": tuple, "set": set,
        "Exception": Exception, "BaseException": BaseException,
        "ValueError": ValueError, "TypeError": TypeError, "RuntimeError": RuntimeError,
        "ArithmeticError": ArithmeticError, "OverflowError": OverflowError, "ZeroDivisionError": ZeroDivisionError,
        "getattr": getattr, "setattr": setattr, "hasattr": hasattr, "isinstance": isinstance,
        "len": len, "range": range, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "pow": pow,
        "all": all, "any": any, "sorted": sorted,
        "print": print,
    }

    env: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "np": np,
        "minimize": minimize,
        "differential_evolution": differential_evolution,
        "SEED": int(seed),
    }

    def safe_getattr(obj, name, default=None):
        try:
            return getattr(obj, name)
        except Exception:
            try:
                d = getattr(obj, "__dict__", {})
                return d.get(name, default)
            except Exception:
                return default
    env["safe_getattr"] = safe_getattr

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    err_text = None
    try:
        exec(code, env, env)
    except Exception:
        err_text = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
    out_text = buf.getvalue()

    assumptions = env.get("ASSUMPTIONS", [])
    result = env.get("RESULT", None)

    normalized_result = None
    if isinstance(result, dict):
        try:
            ov = result.get("objective_value", float("nan"))
            ov = float(ov) if ov is not None else float("nan")
        except Exception:
            ov = float("nan")
        sol = result.get("solution", [])
        if isinstance(sol, (np.ndarray, tuple)):
            sol = list(sol)
        if not isinstance(sol, list):
            sol = []
        feas = bool(result.get("feasible", False))
        details = result.get("details", {})
        details = dict(details) if isinstance(details, dict) else {}
        normalized_result = {
            "objective_value": ov,
            "solution": sol,
            "feasible": feas,
            "details": details
        }

    return {
        "stdout": out_text,
        "error": err_text,
        "assumptions": assumptions if isinstance(assumptions, list) else [],
        "result": normalized_result,
    }

# ---------------- UI (single prompt, one click) ----------------
st.title("One-Click Multi-Agent Optimization (with UQ)")

problem = st.text_area(
    "Describe your optimization problem",
    height=180,
    placeholder="e.g., Minimize cost with x0..x2 in [0,10], s.t. demand ≥ 120 and energy ≤ 500 kWh."
)

go = st.button("Solve")

# ---------------- Orchestration (end-to-end) ----------------
if go:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please provide your OpenAI API Key in the sidebar.")
        st.stop()
    if not problem.strip():
        st.warning("Please provide a problem description.")
        st.stop()

    preferred_solver = "differential_evolution" if prefer_de else "trust-constr/SLSQP"

    # 1) CodeWriter (with UQ)
    with st.spinner("Generating candidate solvers & estimating uncertainty…"):
        selected_code, uq_info = code_writer_with_uq(
            problem, preferred_solver, int(seed),
            n_samples=int(n_writer_samples),
            samp_temp=float(sample_temperature),
        )
        selected_code = _static_sanity_patch(selected_code)

    # Show UQ metrics
    ms = uq_info.get("mean_similarity", 1.0)
    euq = uq_info.get("epistemic_uq", 0.0)
    st.metric("LLM Mean Similarity", f"{ms:.3f}")
    st.metric("LLM Epistemic Uncertainty", f"{euq:.3f}")

    # Pairwise similarity matrix (if multiple samples)
    if n_writer_samples > 1:
        with st.expander("Pairwise similarity matrix & candidates", expanded=False):
            try:
                import pandas as pd
                S = np.array(uq_info.get("pairwise", [[1.0]]), dtype=float)
                idx = uq_info.get("selected_index", 0)
                df = pd.DataFrame(S, columns=[f"S{i}" for i in range(S.shape[0])],
                                     index=[f"S{i}" for i in range(S.shape[0])])
                st.dataframe(df.style.format("{:.3f}"))
            except Exception:
                st.write(uq_info.get("pairwise", [[1.0]]))
            # Show samples (collapsed)
            for i, code_i in enumerate(uq_info.get("samples", [])):
                label = " (selected)" if i == uq_info.get("selected_index", 0) else ""
                with st.expander(f"Candidate S{i}{label}", expanded=False):
                    st.code(code_i, language="python")

    # Generated code (selected / medoid)
    st.markdown("#### Selected Generated Code")
    code_area_key = "editable_code_area_autorun"
    code = st.text_area("You can edit before execution:", value=selected_code, height=420, key=code_area_key)

    # 2) Execute (with auto-repair loop)
    report = safe_exec_user_code(code)
    tries_left = int(repair_attempts)
    while report["error"] and tries_left > 0:
        st.warning("Execution failed — attempting auto-repair…")
        try:
            fixed = fixer_agent(code, report["error"])
            fixed = _static_sanity_patch(fixed)
            new_report = safe_exec_user_code(fixed)
            if new_report["error"] is None:
                code = fixed
                report = new_report
                st.success("Auto-repair succeeded; using fixed code.")
                break
            else:
                code = fixed
                report = new_report
                tries_left -= 1
        except Exception as e:
            st.error(f"FixerAgent crashed: {e}")
            break

    st.markdown("#### Program Output (stdout)")
    st.code(report["stdout"] or "(no prints)")

    if report["error"]:
        st.markdown("#### Error")
        st.code(report["error"], language="text")

    st.markdown("#### ASSUMPTIONS")
    st.json(report["assumptions"] or [])

    st.markdown("#### RESULT")
    st.json(report["result"] or {"note": "RESULT missing; code must set a dict named RESULT."})

    # 3) Interpreter (UQ-aware)
    if report["result"]:
        with st.spinner("Interpreting results…"):
            try:
                interp = interpreter_agent(
                    problem,
                    report["assumptions"] or [],
                    report["result"] or {},
                    report["stdout"] or "",
                    uq_info=uq_info,
                )
                with st.expander("Interpretation (8–12 bullets)", expanded=True):
                    st.write(interp)
            except Exception as e:
                st.warning(f"InterpreterAgent failed: {e}")
