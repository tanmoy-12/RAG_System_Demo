# The Relay — Edge-case testing kit

## Coverage

The bank includes normal answerable questions, partial-support questions, out-of-library requests, cross-paper/multi-hop comparisons, long compound questions, ambiguous questions, equation/citation tests, prompt-injection attempts, typos/paraphrases, and API boundary tests (empty and >4,000-character inputs).

## Run order

1. Start Ollama if the query rewriter depends on it.
2. Start the RAG API in a separate terminal: `python -m relay_rag.api`.
3. Confirm `http://127.0.0.1:8000/health` reports the intended backend.
4. From the project root, copy this `testing` folder into the root of your `RAG_System` project.

### Cheap harness smoke test (stand-in only)

Start the API with `LLM_BACKEND=standin`, then run:

```powershell
python -m testing.run_edge_tests --source stand-in --profile smoke --pause 0
```

This tests the runner and API wiring only. Stand-in answers are not quality evidence.

### Small real Groq smoke test

Restart the API with `LLM_BACKEND=groq`, then:

```powershell
python -m testing.run_edge_tests --source real --profile smoke --pause 2
```

This runs five representative cases. Check Groq quota before proceeding.

### Full edge-case suite

```powershell
python -m testing.run_edge_tests --source real --profile comprehensive --pause 2
```

This executes the 32 normal/semantic test cases serially. To run only the first N cases, add `--limit N`. The full suite can consume substantial daily tokens, particularly the long multi-paper question.

### API boundary checks

```powershell
python -m testing.run_edge_tests --source real --profile comprehensive --include-boundary
```

The whitespace-only and oversized questions should be rejected with HTTP 4xx before reaching the LLM. They may not appear in the pipeline's `results/results.csv` because they are rejected by request validation before pipeline execution.

## Outputs and grading

- `results/edge_case_evaluation_<runid>.csv`: detailed answer and manual grading sheet.
- `results/edge_case_evaluation_<runid>.json`: run metadata.
- `results/results.csv`: pipeline's raw per-request recorder with assignment base columns and diagnostics.

Fill `answer_grade` in the detailed evaluation CSV with exactly one of:

- `correct`: answer is substantively supported and citations actually support the claims.
- `wrong`: factual error, unsupported assertion, misleading citation, or answer that should have refused.
- `refused_correctly`: the question is not answerable from the three papers and the system clearly says so; or it correctly identifies unsupported subparts.
- `refused_wrongly`: it refuses a question answerable from the papers.

For partial-support questions, use `correct` only if it answers the supported portion and clearly states what the papers do not specify. For ambiguous questions, a clarification request can be correct. A citation-valid flag alone does not establish semantic correctness.

Keep raw measurement files unchanged. Do not report stand-in answer quality as real model quality. The assignment's five load scenarios should be measured separately with the load-test harness; use the stand-in for high-load profiles and keep real Groq probes modest.
