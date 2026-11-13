# Citation validation fix

This patch updates the citation validator and its diagnostics.

## What changed

- Wrapped lines in one prose paragraph are joined before citation validation; they are no longer incorrectly treated as separate factual paragraphs.
- Markdown bullet/numbered-list items are still checked individually.
- Citations accept harmless spacing/label variants such as `[Moshi, p.6]` and `[Moshi, page 6]`, but still require the paper name and page to match a retrieved chunk.
- Citations to pages or papers not present in retrieved chunks remain rejected.
- Malformed page citations remain rejected.
- Refusal detection is narrower, so a refusal sentence cannot accidentally make unrelated uncited factual paragraphs pass.
- A single repair prompt now includes the exact allowed citation list and the validator's failure reason.
- The CLI reports `citation_validation_reason`, `citation_found`, `citation_invalid`, and `citation_uncited_blocks`.
- Citation-validation failures are recorded as `status=error` with `error_kind=citation_validation_failed`, rather than appearing as successful requests.
- The raw results CSV includes diagnostic columns for the validation reason and invalid references.

## Apply to an existing workspace

Extract the `relay_rag` and `tests` directories from this patch into the existing project root, allowing files to be overwritten. The existing PDFs and indexes are not included or deleted.

Then run:

```powershell
python -m compileall -q relay_rag tests
python -m unittest discover -s tests -v
```

Expected result: 23 tests pass.

Run the same test query again:

```powershell
python -m relay_rag.cli "How does scaled dot-product attention work?"
```

If validation still fails, the JSON diagnostic now identifies whether the issue is an invalid paper/page pair, malformed citation syntax, or an uncited factual block. Share that diagnostic JSON (never your `.env` or API key) for the next investigation.

## Display-math citation handling

The validator now treats standalone display equations as factual blocks. It supports
`\\[ ... \\]`, `$$ ... $$`, and common LaTeX environments (`equation`, `align`,
`alignat`, `flalign`, `gather`, `multline`, `eqnarray`, and `displaymath`).

A display equation is accepted only when a valid retrieved paper/page citation is:

- in the same paragraph/block as the equation; or
- in a citation-only paragraph immediately after the equation.

An uncited equation remains a validation failure. A citation to a page that was
not among the retrieved chunks remains invalid. The system prompt and one-time
repair prompt instruct the model to place the citation immediately after the
closing math delimiter.

The expanded local test suite currently contains 23 tests, including acceptance
and rejection cases for display equations.
