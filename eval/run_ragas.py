"""
RAGAS 평가 스크립트 — ChaTech 8b / 70b / proto2 비교
Judge LLM: GPT-4o-mini

실행 방법:
    python run_ragas.py --model 8b           # ChaTech 8b (parent_context, k=4)
    python run_ragas.py --model 70b          # ChaTech 70b (parent_context, k=4)
    python run_ragas.py --model proto2       # Proto2 (child_summary, k=3)
    python run_ragas.py --model 8b --fresh   # 체크포인트 무시하고 처음부터
    python run_ragas.py --model 8b --eval-only  # 기존 체크포인트로 RAGAS만 재실행

API 키 전략:
    - GROQ_API_KEY3 → KEY1 → KEY4 → KEY2 순 자동 전환
"""
import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── 모델 선택 (CLI) ──────────────────────────────────────────────────────────
_MODEL_MAP = {
    "8b":     "llama-3.1-8b-instant",
    "70b":    "llama-3.3-70b-versatile",
    "proto2": "llama-3.1-8b-instant",
    "chatech": "llama-3.1-8b-instant",  # 실제 retrieve_with_parent 파이프라인
}
_model_arg = "8b"
for arg in sys.argv[1:]:
    if arg in _MODEL_MAP:
        _model_arg = arg
_MODEL = _MODEL_MAP[_model_arg]
os.environ["MODEL_NAME"] = _MODEL

# ── API 키 설정 (GROQ_API_KEY2 우선) ─────────────────────────────────────────
_KEY1     = os.getenv("GROQ_API_KEY")        # 기존 키
_KEY2     = os.getenv("GROQ_API_KEY2")       # 부계정1
_KEY3     = os.getenv("GROQ_API_KEY3")       # 부계정2 (신규)
_KEY4     = os.getenv("GROQ_API_KEY4")       # 부계정3 (신규)
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# KEY3 → KEY1 → KEY4 순으로 우선 사용 (KEY2 70b 소진됨)
_KEY_PRIORITY = [(k, n) for k, n in [(_KEY3,"KEY3"),(_KEY1,"KEY1"),(_KEY4,"KEY4"),(_KEY2,"KEY2")] if k]
_active_key_name = _KEY_PRIORITY[0][1] if _KEY_PRIORITY else "KEY1"

os.environ["GROQ_API_KEY"] = _KEY_PRIORITY[0][0] if _KEY_PRIORITY else _KEY1

# ── 설정 ────────────────────────────────────────────────────────────────────
_CHROMA_DIR      = str(Path(__file__).resolve().parent / "chroma_db")
_GOLDEN_FILE     = "golden_set.json"
_RESULT_FILE     = f"ragas_results_{_model_arg}.json"
_CHECKPOINT_FILE = f"ragas_checkpoint_{_model_arg}.json"
_N_RESULTS       = 4
_DELAY           = 15   # 질문 간 대기 (Groq TPM 보호)

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datasets import Dataset

judge_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-4o-mini", api_key=_OPENAI_KEY, temperature=0,
))
judge_emb = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small", api_key=_OPENAI_KEY)
)
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

_MODE_DESC = {
    "8b":     f"ChaTech 8b  (similarity_search+parent_context, k={_N_RESULTS})",
    "70b":    f"ChaTech 70b (similarity_search+parent_context, k={_N_RESULTS})",
    "proto2": "Proto2      (similarity_search+child_summary, k=4)",
    "chatech": "ChaTech     (retrieve_with_parent 실제 파이프라인)",
}
print(f"=== RAGAS 평가 시작 ===")
print(f"  모드: {_MODE_DESC.get(_model_arg, _model_arg)}")
print(f"  API 키: {_active_key_name} 우선 → 한도 소진 시 자동 전환")
print(f"  질문 간 delay={_DELAY}s\n")


def load_db() -> Chroma:
    return Chroma(
        collection_name="seoultech_v1",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=_CHROMA_DIR,
    )


def _switch_to_fallback_key() -> None:
    global _active_key_name, _KEY_PRIORITY
    current_names = [n for _, n in _KEY_PRIORITY]
    if _active_key_name in current_names:
        idx = current_names.index(_active_key_name)
        if idx + 1 < len(_KEY_PRIORITY):
            next_key, next_name = _KEY_PRIORITY[idx + 1]
            os.environ["GROQ_API_KEY"] = next_key
            _active_key_name = next_name
            print(f"  ⚠️  {current_names[idx]} 한도 소진 → {next_name} 로 전환", flush=True)
            return
    raise RuntimeError("모든 API 키 한도 소진. 내일 재시도하세요.")


def answer_current(query: str, db: Chroma) -> tuple[str, list[str]]:
    """ChaTech: child로 검색 → parent_context로 답변 (k=4)."""
    from app.web_ui import build_answer

    docs = db.similarity_search(query, k=_N_RESULTS)
    parent_contexts = [doc.metadata.get("parent_context", doc.page_content)[:1500] for doc in docs]
    ctx_str = "\n\n".join(f"[문서 {i+1}]\n{c}" for i, c in enumerate(parent_contexts))

    _ERROR_SIGNAL = "죄송합니다. 현재 답변 생성 중 오류가 발생했습니다."

    for attempt in range(2):
        try:
            answer = build_answer(query, ctx_str, [])
            if _ERROR_SIGNAL in answer:
                raise RuntimeError("build_answer internal error (rate limit or API failure)")
            return answer, parent_contexts
        except Exception as e:
            if attempt == 0:
                _switch_to_fallback_key()
                time.sleep(5)
                continue
            raise
    raise RuntimeError("답변 생성 실패")


def answer_chatech(query: str, db: Chroma) -> tuple[str, list[str]]:
    """ChaTech 실제 파이프라인: retrieve_with_parent + build_answer."""
    import unittest.mock as mock
    with mock.patch("streamlit.cache_resource", lambda **kw: lambda f: f), \
         mock.patch("streamlit.set_page_config", lambda **kw: None), \
         mock.patch("streamlit.secrets", {}, create=True):
        from app.web_ui import retrieve_with_parent, build_answer

    ctx_str, sources = retrieve_with_parent(query, db)
    # RAGAS용 contexts: "---" 구분자로 분리
    contexts = [c.strip() for c in ctx_str.split("---") if c.strip()]

    _ERROR_SIGNAL = "죄송합니다. 현재 답변 생성 중 오류가 발생했습니다."

    for attempt in range(2):
        try:
            answer = build_answer(query, ctx_str, [])
            if _ERROR_SIGNAL in answer:
                raise RuntimeError("build_answer internal error")
            return answer, contexts
        except Exception as e:
            if attempt == 0:
                _switch_to_fallback_key()
                time.sleep(5)
                continue
            raise
    raise RuntimeError("답변 생성 실패")


def answer_proto2(query: str, db: Chroma) -> tuple[str, list[str]]:
    """Proto2: child_summary를 그대로 컨텍스트로 사용 (k=4, parent_context 없음)."""
    from app.web_ui import build_answer

    docs = db.similarity_search(query, k=4)
    child_summaries = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(f"[문서 {i+1}]\n{c}" for i, c in enumerate(child_summaries))

    _ERROR_SIGNAL = "죄송합니다. 현재 답변 생성 중 오류가 발생했습니다."

    for attempt in range(2):
        try:
            answer = build_answer(query, ctx_str, [])
            if _ERROR_SIGNAL in answer:
                raise RuntimeError("build_answer internal error (rate limit or API failure)")
            return answer, child_summaries
        except Exception as e:
            if attempt == 0:
                _switch_to_fallback_key()
                time.sleep(5)
                continue
            raise
    raise RuntimeError("답변 생성 실패")


def _save_checkpoint(idx: int, answers: list, contexts: list) -> None:
    with open(_CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "model": _model_arg,
            "last_completed_idx": idx,
            "answers": answers,
            "contexts": contexts,
        }, f, ensure_ascii=False)
    print(f"    → 체크포인트 저장 [{idx+1}번 완료]", flush=True)


def _load_checkpoint() -> dict | None:
    p = Path(_CHECKPOINT_FILE)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("model") != _model_arg:
        print(f"  ⚠️  체크포인트 모델({data.get('model')}) ≠ 현재 모델({_model_arg}) → 무시")
        return None
    return data


def build_dataset(questions: list[dict], answers: list[str],
                  contexts_list: list[list[str]]) -> Dataset:
    return Dataset.from_dict({
        "question":     [q["question"]     for q in questions],
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": [q["ground_truth"] for q in questions],
    })


def _to_score(val) -> float:
    if isinstance(val, (list, tuple)):
        valid = [v for v in val if v is not None]
        return sum(valid) / len(valid) if valid else 0.0
    return float(val)


def run_ragas(dataset: Dataset) -> dict:
    print(f"\n  RAGAS 평가 중 ({_model_arg})...", flush=True)
    run_config = RunConfig(timeout=120, max_retries=5, max_wait=60)
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
        raise_exceptions=False,
        batch_size=3,
    )
    scores = {
        "faithfulness":      round(_to_score(result["faithfulness"]), 4),
        "answer_relevancy":  round(_to_score(result["answer_relevancy"]), 4),
        "context_precision": round(_to_score(result["context_precision"]), 4),
        "context_recall":    round(_to_score(result["context_recall"]), 4),
    }
    scores["average"] = round(sum(scores.values()) / len(scores), 4)
    return scores


def _print_and_save(golden: list, scores: dict, answers: list) -> None:
    print("\n" + "=" * 50)
    print(f"=== RAGAS 결과 ({_MODEL}) ===\n")
    for metric, val in scores.items():
        label = "★ 평균" if metric == "average" else metric
        print(f"  {label:<22} {val:.4f}")
    print("=" * 50)

    results = {"model": _MODEL, "scores": scores, "answers": answers}
    with open(_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {_RESULT_FILE}")
    print(f"체크포인트 유지: {_CHECKPOINT_FILE}")


def main(fresh: bool = False, eval_only: bool = False) -> None:
    with open(_GOLDEN_FILE, encoding="utf-8") as f:
        golden = json.load(f)
    print(f"골든셋 {len(golden)}개 로드 완료\n", flush=True)

    db = load_db()

    # ── eval-only: 체크포인트 → RAGAS만 ────────────────────────────────────
    if eval_only:
        ckpt = _load_checkpoint()
        if not ckpt:
            print("체크포인트 없음 — 먼저 답변 수집을 완료해주세요.")
            return
        answers  = ckpt["answers"]
        contexts = ckpt["contexts"]
        print(f"체크포인트 로드: {len(answers)}개 답변 → RAGAS 평가 시작\n")
        scores = run_ragas(build_dataset(golden[:len(answers)], answers, contexts))
        _print_and_save(golden[:len(answers)], scores, answers)
        return

    # ── 체크포인트 복원 ──────────────────────────────────────────────────────
    start_idx = 0
    answers, contexts = [], []

    if not fresh:
        ckpt = _load_checkpoint()
        if ckpt:
            start_idx = ckpt["last_completed_idx"] + 1
            answers   = ckpt["answers"]
            contexts  = ckpt["contexts"]
            print(f"체크포인트 복원 → {start_idx}번부터 재시작\n", flush=True)

    # ── 단계 1: 답변 수집 ────────────────────────────────────────────────────
    print("1단계: 답변 수집 중...", flush=True)

    for i, item in enumerate(golden):
        if i < start_idx:
            continue

        q = item["question"]
        print(f"  [{i+1}/{len(golden)}] ({_active_key_name}) {q[:50]}", flush=True)

        if _model_arg == "proto2":
            ans, ctx = answer_proto2(q, db)
        elif _model_arg == "chatech":
            ans, ctx = answer_chatech(q, db)
        else:
            ans, ctx = answer_current(q, db)
        answers.append(ans)
        contexts.append(ctx)
        _save_checkpoint(i, answers, contexts)
        time.sleep(_DELAY)

    # ── 단계 2: RAGAS 평가 ───────────────────────────────────────────────────
    print("\n2단계: RAGAS 평가 중...", flush=True)
    scores = run_ragas(build_dataset(golden, answers, contexts))
    _print_and_save(golden, scores, answers)


if __name__ == "__main__":
    fresh     = "--fresh" in sys.argv
    eval_only = "--eval-only" in sys.argv
    main(fresh=fresh, eval_only=eval_only)
