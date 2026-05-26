# BRSR-GRI Disclosure Evaluation Pipeline

## Overview

I built this pipeline to evaluate the quality of corporate ESG disclosures against
GRI (Global Reporting Initiative) standards at scale. It processes Indian firm-level
BRSR (Business Responsibility and Sustainability Reporting) filings and assesses,
at the sub-requirement level, whether each firm's reported disclosures meet, partially
meet, or fail to meet the corresponding GRI criteria.

The evaluation engine is powered by LLaMA-3.3-70B accessed via the NVIDIA NIM API,
with structured prompting that enforces strict grounding in the provided disclosure
text and prohibits fabrication across all output fields.

## Key Features

- Sub-requirement decomposition instructs the model to split multi-part GRI
  requirements and evaluate each sub-clause independently
- Strict grounding rules prohibit paraphrase, fabrication, and mixing of disclosure
  text with requirement text across output fields
- API key rotation with rate-limit handling cycles across multiple keys, detects
  429 responses, applies backoff, and logs all key-switching events
- Parallel processing handles up to five firms concurrently using a thread pool
  with thread-safe logging and in-memory event buffering
- Context budget management uses a character budget mapped to token consumption
  rather than a fixed row cutoff
- Full resume support via a persistent status log — firms already evaluated in a
  prior run are automatically skipped

## Technologies

| Category | Tools |
|---|---|
| LLM | LLaMA-3.3-70B-Instruct (NVIDIA NIM) |
| API Client | OpenAI SDK (custom base URL) |
| Parallelism | ThreadPoolExecutor |
| Data I/O | pandas, openpyxl |
| Logging | Excel-based status log (summary + events sheets) |

## Requirements

```bash
pip install openai pandas openpyxl python-dotenv tqdm
```

## Configuration

Create a `.env` file in the project root before running:
