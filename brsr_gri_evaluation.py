import os
import re
import json
import time
import uuid
import logging
import threading
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, RateLimitError


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter(
    "%(asctime)s [%(threadName)s] %(message)s", datefmt="%H:%M:%S"
))
logger.addHandler(_console)


# =============================================================================
# CONFIGURATION
# =============================================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEYS = [
    os.environ.get("NVAPI_KEY_1", ""),
    os.environ.get("NVAPI_KEY_2", ""),
    os.environ.get("NVAPI_KEY_3", ""),
]
KEY_NAMES = ["KEY_1", "KEY_2", "KEY_3"]

LINKAGE_EXCEL_PATH = "BRSR_final_verified_sorted.xlsx"
FIRMS_FOLDER       = "FY_2023_2024"
OUTPUT_FOLDER      = "output_files"
STATUS_LOG_PATH    = "logs/status_log.xlsx"

RLM_MODEL         = "meta/llama-3.3-70b-instruct"
BATCH_WRITE_EVERY = 20
MAX_WORKERS       = 5
CONTEXT_CHAR_BUDGET = 12000

os.makedirs(OUTPUT_FOLDER,                    exist_ok=True)
os.makedirs(os.path.dirname(STATUS_LOG_PATH), exist_ok=True)

RUN_ID = str(uuid.uuid4())[:8]

OUTPUT_COLUMNS = [
    "file_name", "firm_name", "financial_year", "brsr_code",
    "gri_no", "gri_sub_no", "section_no", "sub_section_no",
    "gri_standard", "gri_substandard", "gri_header", "gri_sub_header",
    "section_header", "section", "sub_section_header", "sub_section",
    "linkage", "actual_text",
    "evaluation", "present_elements", "missing_elements", "evidence_quotes",
    "present_elements_actual", "missing_elements_actual",
    "confidence", "total_num", "present_num", "missing_num", "score",
    "length_evidence_quotes"
]

SUMMARY_COLUMNS = [
    "run_id", "firm_id", "firm_file", "start_time", "end_time",
    "duration_sec", "total_rows_expected", "processed_rows", "status",
    "output_path", "last_active_key", "retries_total", "rate_limit_events",
    "input_tokens_total", "output_tokens_total", "total_tokens",
    "error_message_last"
]

EVENTS_COLUMNS = [
    "run_id", "timestamp", "firm_id", "row_index", "action",
    "key_from", "key_to", "wait_seconds", "input_tokens", "output_tokens",
    "exception_type", "exception_message", "notes"
]


# =============================================================================
# API KEY ROTATION
# =============================================================================

_key_index      = 0
_key_index_lock = threading.Lock()


def get_client(key_index):
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=API_KEYS[key_index]
    )


def next_key_index(current):
    return (current + 1) % len(API_KEYS)


# =============================================================================
# STATUS LOGGING
# =============================================================================

_log_lock      = threading.Lock()
_event_buffer  = []
_summary_cache = None
_events_cache  = None


def _ensure_log_loaded():
    global _summary_cache, _events_cache
    if _summary_cache is not None:
        return
    if os.path.exists(STATUS_LOG_PATH):
        try:
            _summary_cache = pd.read_excel(STATUS_LOG_PATH, sheet_name="summary")
            _events_cache  = pd.read_excel(STATUS_LOG_PATH, sheet_name="events")
            for col in SUMMARY_COLUMNS:
                if col not in _summary_cache.columns:
                    _summary_cache[col] = None
            for col in EVENTS_COLUMNS:
                if col not in _events_cache.columns:
                    _events_cache[col] = None
            _summary_cache = _summary_cache[SUMMARY_COLUMNS]
            _events_cache  = _events_cache[EVENTS_COLUMNS]
            return
        except Exception:
            pass
    _summary_cache = pd.DataFrame(columns=SUMMARY_COLUMNS)
    _events_cache  = pd.DataFrame(columns=EVENTS_COLUMNS)


def _flush_log():
    global _event_buffer, _events_cache
    _ensure_log_loaded()
    if _event_buffer:
        _events_cache = pd.concat(
            [_events_cache, pd.DataFrame(_event_buffer)], ignore_index=True
        )
        _event_buffer = []
    tmp = STATUS_LOG_PATH + ".tmp"
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        _summary_cache.to_excel(writer, sheet_name="summary", index=False)
        _events_cache.to_excel(writer,  sheet_name="events",  index=False)
    os.replace(tmp, STATUS_LOG_PATH)


def _empty_event_row():
    return {col: None for col in EVENTS_COLUMNS}


def log_event(firm_id, action, row_index=None, key_from=None, key_to=None,
              wait_seconds=None, input_tokens=None, output_tokens=None,
              exception_type=None, exception_message=None, notes=None):
    row = {
        **_empty_event_row(),
        "run_id":            RUN_ID,
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "firm_id":           firm_id,
        "row_index":         row_index,
        "action":            action,
        "key_from":          key_from,
        "key_to":            key_to,
        "wait_seconds":      wait_seconds,
        "input_tokens":      input_tokens,
        "output_tokens":     output_tokens,
        "exception_type":    exception_type,
        "exception_message": exception_message,
        "notes":             notes,
    }
    with _log_lock:
        _event_buffer.append(row)


def flush_log():
    with _log_lock:
        _flush_log()


def log_summary_start(firm_id, firm_file, total_rows_expected, output_path):
    with _log_lock:
        _ensure_log_loaded()
        global _summary_cache
        mask = ((_summary_cache["run_id"] == RUN_ID) &
                (_summary_cache["firm_id"] == firm_id))
        row = {
            "run_id":              RUN_ID,
            "firm_id":             firm_id,
            "firm_file":           firm_file,
            "start_time":          datetime.now().isoformat(timespec="seconds"),
            "end_time":            None,
            "duration_sec":        None,
            "total_rows_expected": total_rows_expected,
            "processed_rows":      0,
            "status":              "in_progress",
            "output_path":         output_path,
            "last_active_key":     KEY_NAMES[_key_index],
            "retries_total":       0,
            "rate_limit_events":   0,
            "input_tokens_total":  0,
            "output_tokens_total": 0,
            "total_tokens":        0,
            "error_message_last":  None,
        }
        if mask.any():
            for k, v in row.items():
                _summary_cache.loc[mask, k] = v
        else:
            _summary_cache = pd.concat(
                [_summary_cache, pd.DataFrame([row])], ignore_index=True
            )


def log_summary_update(firm_id, **kwargs):
    with _log_lock:
        _ensure_log_loaded()
        global _summary_cache
        mask = ((_summary_cache["run_id"] == RUN_ID) &
                (_summary_cache["firm_id"] == firm_id))
        if not mask.any():
            return
        for k, v in kwargs.items():
            if k in SUMMARY_COLUMNS:
                _summary_cache.loc[mask, k] = v


def log_summary_complete(firm_id, start_time_str, processed_rows,
                          last_active_key, retries_total, rate_limit_events,
                          input_tokens_total, output_tokens_total,
                          error_message=None):
    end_time = datetime.now()
    try:
        duration = round(
            (end_time - datetime.fromisoformat(start_time_str)).total_seconds(), 1
        )
    except Exception:
        duration = None

    status = "error" if error_message else "completed"
    with _log_lock:
        _ensure_log_loaded()
        global _summary_cache
        mask = ((_summary_cache["run_id"] == RUN_ID) &
                (_summary_cache["firm_id"] == firm_id))
        if not mask.any():
            return
        _summary_cache.loc[mask, "end_time"]            = end_time.isoformat(timespec="seconds")
        _summary_cache.loc[mask, "duration_sec"]        = duration
        _summary_cache.loc[mask, "processed_rows"]      = processed_rows
        _summary_cache.loc[mask, "status"]              = status
        _summary_cache.loc[mask, "last_active_key"]     = last_active_key
        _summary_cache.loc[mask, "retries_total"]       = retries_total
        _summary_cache.loc[mask, "rate_limit_events"]   = rate_limit_events
        _summary_cache.loc[mask, "input_tokens_total"]  = input_tokens_total
        _summary_cache.loc[mask, "output_tokens_total"] = output_tokens_total
        _summary_cache.loc[mask, "total_tokens"]        = input_tokens_total + output_tokens_total
        _summary_cache.loc[mask, "error_message_last"]  = error_message
        _flush_log()


def log_skip(firm_id, firm_file, output_path, total_rows_expected):
    now = datetime.now().isoformat(timespec="seconds")
    with _log_lock:
        _ensure_log_loaded()
        global _summary_cache
        mask = ((_summary_cache["run_id"] == RUN_ID) &
                (_summary_cache["firm_id"] == firm_id))
        if not mask.any():
            row = {
                "run_id":              RUN_ID,
                "firm_id":             firm_id,
                "firm_file":           firm_file,
                "start_time":          now,
                "end_time":            now,
                "duration_sec":        0,
                "total_rows_expected": total_rows_expected,
                "processed_rows":      total_rows_expected,
                "status":              "skipped_already_evaluated",
                "output_path":         output_path,
                "last_active_key":     None,
                "retries_total":       0,
                "rate_limit_events":   0,
                "input_tokens_total":  0,
                "output_tokens_total": 0,
                "total_tokens":        0,
                "error_message_last":  None,
            }
            _summary_cache = pd.concat(
                [_summary_cache, pd.DataFrame([row])], ignore_index=True
            )
        _event_buffer.append({
            **_empty_event_row(),
            "run_id":    RUN_ID,
            "timestamp": now,
            "firm_id":   firm_id,
            "action":    "skip_already_evaluated",
            "notes":     f"output already complete at {output_path}",
        })
        _flush_log()


def is_firm_already_completed(firm_id, expected_row_count, output_path):
    with _log_lock:
        _ensure_log_loaded()
        try:
            firm_rows = _summary_cache[_summary_cache["firm_id"] == firm_id]
            if not firm_rows.empty:
                latest = firm_rows.iloc[-1]
                if (str(latest.get("status", "")) == "completed" and
                        int(latest.get("processed_rows", 0)) >= expected_row_count):
                    return True
        except Exception:
            pass
    if os.path.exists(output_path):
        try:
            if len(pd.read_excel(output_path)) >= expected_row_count:
                return True
        except Exception:
            pass
    return False


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """
You are an ESG reporting evaluation assistant.

STRICT RULES:

1. Use ONLY the Firm Disclosure Context and the GRI actual_text requirement provided.

2. The actual_text contains multiple consolidated GRI sub-requirements.
   You MUST split actual_text into individual sub-requirements and evaluate EACH one separately.

3. present_elements: List ONLY sub-requirement texts from actual_text that the firm HAS disclosed.

4. missing_elements: List ONLY sub-requirement texts from actual_text that the firm has NOT disclosed.

5. evidence_quotes: Exact verbatim quotes copied from the Firm Disclosure Data. No paraphrase. No fabrication.

6. present_elements_actual: Exact excerpts from Firm Disclosure Data confirming each present element.

7. missing_elements_actual: Describe what required disclosures are absent from the Firm Disclosure Data.

8. DO NOT put firm disclosure text in present_elements or missing_elements.

9. Interpretation rules:
   meets           = all sub-requirements clearly present
   partial         = some present, some missing
   does_not_meet   = all or most core sub-requirements missing
   not_applicable  = requirement explicitly not relevant to this firm

10. confidence = certainty in your verdict (0.0 to 1.0)

Return ONLY valid JSON with keys:
evaluation, present_elements, missing_elements, evidence_quotes,
present_elements_actual, missing_elements_actual, confidence
"""


# =============================================================================
# LLM EVALUATION
# =============================================================================

def count_sub_requirements(actual_text):
    parts = re.split(r"(?<=[.;])\s+(?=[a-z]\.\s)", str(actual_text))
    return max(len([p for p in parts if p.strip()]), 1)


def evaluate_llm(user_prompt, firm_id, row_index):
    global _key_index
    retries    = 0
    rl_count   = 0
    in_tokens  = 0
    out_tokens = 0

    while True:
        with _key_index_lock:
            ki = _key_index

        try:
            client   = get_client(ki)
            response = client.chat.completions.create(
                model=RLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            in_tokens  = response.usage.prompt_tokens
            out_tokens = response.usage.completion_tokens
            raw        = response.choices[0].message.content
            result     = json.loads(raw.replace("```json", "").replace("```", "").strip())
            return result, in_tokens, out_tokens, retries, rl_count

        except RateLimitError:
            rl_count += 1
            with _key_index_lock:
                new_ki      = next_key_index(ki)
                _key_index  = new_ki
            log_event(firm_id=firm_id, action="rate_limit_key_switch",
                      row_index=row_index, key_from=KEY_NAMES[ki],
                      key_to=KEY_NAMES[new_ki], wait_seconds=65)
            time.sleep(65)
            retries += 1

        except Exception as exc:
            retries += 1
            if retries > 5:
                raise RuntimeError(f"Max retries exceeded: {exc}") from exc
            time.sleep(5 * retries)


def build_user_prompt(lr, combined_col):
    actual_text  = str(lr.get("actual_text", ""))
    gri_standard = str(lr.get("gri_standard", ""))
    gri_header   = str(lr.get("gri_header",   ""))

    rows_text = combined_col.str[:CONTEXT_CHAR_BUDGET // max(len(combined_col), 1)]
    context   = "
".join(rows_text.tolist())[:CONTEXT_CHAR_BUDGET]

    return (
        f"GRI Standard    : {gri_standard}
"
        f"GRI Header      : {gri_header}
"
        f"Requirement     :
{actual_text}

"
        f"Firm Disclosure Context:
{context}"
    )


def process_row(lr, firm_df, firm_name, year, brsr_code,
                file_name, combined_col, firm_id, row_index):
    gri_sub_no    = str(lr.get("gri_sub_no",    ""))
    sub_section_no = str(lr.get("sub_section_no", ""))
    gri_standard   = str(lr.get("gri_standard",  ""))
    actual_text    = str(lr.get("actual_text",   ""))

    user_prompt = build_user_prompt(lr, combined_col)
    key         = (firm_name, year, gri_sub_no.strip(), sub_section_no.strip())

    try:
        res, in_tok, out_tok, retries, rl_count = evaluate_llm(
            user_prompt, firm_id=firm_id, row_index=row_index
        )
    except RuntimeError as exc:
        logger.error("FAILED: %s", exc)
        res      = {
            "evaluation":              "error",
            "present_elements":        [],
            "missing_elements":        [],
            "evidence_quotes":         [],
            "present_elements_actual": [],
            "missing_elements_actual": [],
            "confidence":              0.0,
        }
        in_tok = out_tok = retries = rl_count = 0

    present_num = len(res.get("present_elements", []))
    missing_num = len(res.get("missing_elements", []))
    total_num   = count_sub_requirements(actual_text)
    score       = min(round(present_num / total_num, 4), 1.0) if total_num > 0 else None

    eq_json    = json.dumps(res.get("evidence_quotes", []), ensure_ascii=False)
    output_row = {col: lr.get(col, "") for col in lr.index}
    output_row.update({
        "file_name":               file_name,
        "firm_name":               firm_name,
        "financial_year":          year,
        "brsr_code":               brsr_code,
        "evaluation":              res.get("evaluation"),
        "present_elements":        json.dumps(res.get("present_elements",        []), ensure_ascii=False),
        "missing_elements":        json.dumps(res.get("missing_elements",        []), ensure_ascii=False),
        "evidence_quotes":         eq_json,
        "present_elements_actual": json.dumps(res.get("present_elements_actual", []), ensure_ascii=False),
        "missing_elements_actual": json.dumps(res.get("missing_elements_actual", []), ensure_ascii=False),
        "confidence":              res.get("confidence", 0.0),
        "total_num":               total_num,
        "present_num":             present_num,
        "missing_num":             missing_num,
        "score":                   score,
        "length_evidence_quotes":  len(eq_json),
    })
    return (key, {col: output_row.get(col, "") for col in OUTPUT_COLUMNS},
            in_tok, out_tok, retries, rl_count)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

logger.info("Run ID  : %s", RUN_ID)
logger.info("Starting evaluation pipeline...")

linkage_df         = pd.read_excel(LINKAGE_EXCEL_PATH).fillna("")
expected_row_count = len(linkage_df)
logger.info("Linkage rows loaded : %d", expected_row_count)

all_csv_files = sorted([f for f in os.listdir(FIRMS_FOLDER) if f.endswith(".csv")])
logger.info("Firm CSV files found : %d", len(all_csv_files))

for file in all_csv_files:
    path      = os.path.join(FIRMS_FOLDER, file)
    base      = file.replace(".csv", "")
    firm_id   = base

    output_filename = base.replace("XBRL", "GRI").replace("_xbrl", "_GRI") + ".xlsx"
    output_path     = os.path.join(OUTPUT_FOLDER, output_filename)

    year_match = re.search(r"(20\d{2}).*(20\d{2})", base)
    year       = (f"FY {year_match.group(1)}-{year_match.group(2)}"
                  if year_match else "UNKNOWN_YEAR")
    firm_name  = re.sub(r"(20\d{2}).*(20\d{2})", "", base).replace("_", " ").strip()
    firm_name  = re.sub(r"xbrl.*", "", firm_name, flags=re.I).strip()
    brsr_code  = (f"{firm_name}_{year.replace('FY ', '')}"
                  if year != "UNKNOWN_YEAR" else f"{firm_name}_UNKNOWN")

    logger.info("=" * 60)
    logger.info("Firm : %s | %s", firm_name, year)

    if is_firm_already_completed(firm_id, expected_row_count, output_path):
        logger.info("[SKIP] Already complete.")
        log_skip(firm_id, file, output_path, expected_row_count)
        continue

    out_df_existing = pd.read_excel(output_path).fillna("") if os.path.exists(output_path) else pd.DataFrame()

    completed = set()
    if not out_df_existing.empty:
        required_cols = {"firm_name", "financial_year", "gri_sub_no", "sub_section_no"}
        if required_cols.issubset(set(out_df_existing.columns)):
            completed = set(zip(
                out_df_existing["firm_name"],
                out_df_existing["financial_year"],
                out_df_existing["gri_sub_no"].astype(str),
                out_df_existing["sub_section_no"].astype(str)
            ))

    pending = [
        (i, lr)
        for i, (_, lr) in enumerate(linkage_df.iterrows())
        if (firm_name, year,
            str(lr.get("gri_sub_no",     "")).strip(),
            str(lr.get("sub_section_no", "")).strip()) not in completed
    ]

    if not pending:
        continue

    firm_df      = pd.read_csv(path).fillna("")
    combined_col = firm_df.apply(
        lambda r: " | ".join(f"{c}: {r[c]}" for c in firm_df.columns), axis=1
    )

    start_time_str = datetime.now().isoformat(timespec="seconds")
    log_summary_start(firm_id, file, expected_row_count, output_path)

    results         = []
    done_count      = 0
    firm_in_tokens  = 0
    firm_out_tokens = 0
    firm_retries    = 0
    firm_rl_events  = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_row,
                lr, firm_df, firm_name, year, brsr_code,
                file, combined_col, firm_id, row_index
            ): (row_index, lr)
            for row_index, lr in pending
        }

        for future in as_completed(futures):
            try:
                key, output_row, in_tok, out_tok, retries, rl_count = future.result()
            except Exception as exc:
                logger.error("Worker error: %s", exc)
                continue

            results.append(output_row)
            completed.add(key)
            done_count      += 1
            firm_in_tokens  += in_tok
            firm_out_tokens += out_tok
            firm_retries    += retries
            firm_rl_events  += rl_count

            if done_count % BATCH_WRITE_EVERY == 0:
                pd.DataFrame(results).to_excel(output_path + ".partial", index=False)
                logger.info("  Checkpoint : %d rows done", done_count)

    if results:
        final_df = pd.concat([out_df_existing, pd.DataFrame(results)], ignore_index=True)
        final_df.to_excel(output_path, index=False)
    else:
        final_df = out_df_existing

    partial_path = output_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    with _key_index_lock:
        active_key_name = KEY_NAMES[_key_index]

    log_summary_complete(
        firm_id=firm_id,
        start_time_str=start_time_str,
        processed_rows=len(final_df),
        last_active_key=active_key_name,
        retries_total=firm_retries,
        rate_limit_events=firm_rl_events,
        input_tokens_total=firm_in_tokens,
        output_tokens_total=firm_out_tokens,
    )
    flush_log()

    logger.info("DONE — %s | Rows: %d | Tokens: %d in / %d out",
                output_path, len(final_df), firm_in_tokens, firm_out_tokens)

flush_log()
logger.info("ALL FIRMS EVALUATED. Outputs: %s", OUTPUT_FOLDER)
