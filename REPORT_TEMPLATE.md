# The Relay — results write-up (fill only from your measured runs)

Keep the body to the assignment's two-page limit. Replace every bracketed field. Do not invent measurements; use `not measured` where appropriate.

## 1. What I built and why
[Describe the pipeline and the principal design trade-off. State the retrieval threshold and why you chose it.]

## 2. The real service and my stand-in
[Report Groq model ID, account/model quota evidence, response headers and observed 429 behavior. Describe stand-in configuration and calibration from the actual real-service sample. Distinguish documented, observed and inferred limits; mark confidence.]

## 3. Results
[Insert completed Table 1, Table 2a and Table 2b. Report p50/p95/p99 and sample counts. Keep `results.csv` unchanged as the source of truth.]

## 4. Where the time goes
[Choose one slow request_id. Split total latency into queue wait, work and retry wait using the diagnostic columns in results.csv. Say whether the slow part occurred before or after draft first token.]

## 5. When a limit is hit
[Explain RPM/TPM handling separately from RPD/TPD handling. Identify who waits, who is dropped and why.]

## Retrieval: quality versus cost
[Insert Table 3 for small/balanced/large settings. Explain the selected setting and how retrieval is shortened or altered when the remaining quota is low.]

## How this library will mislead a naive retriever
1. [Failure mode + concrete example question + why naive retrieval fails.]
2. [Failure mode + concrete example question + why naive retrieval fails.]
3. [Failure mode + concrete example question + why naive retrieval fails.]

## 6. Retries
[State retry conditions, maximum attempts, deadline behavior and one measured case where retrying worsened latency or throughput.]

## 7. What the person sees
[Describe the exact response when the daily budget is exhausted at 3pm. Base it on your configured/observed behavior.]

## 8. What I'm unsure about, what I left out, what I learned
[Least certain assumption and consequence if wrong. Two deliberately omitted features. One new learning explained in your own words.]

## Your feedback on Vaani
[Your own visit to https://www.vaani.dev/: what stood out, what confused you, and one change you would make.]

## How I worked with the assistant
[Attach the real conversation history or prompts. In about half a page: three most important prompts and why; one assistant error you caught and how; one decision you overruled; what is distinctly yours.]
