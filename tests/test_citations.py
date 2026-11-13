import unittest

from relay_rag.citations import validate_citations
from relay_rag.retriever import RetrievedChunk


class CitationValidationTests(unittest.TestCase):
    def setUp(self):
        self.chunk = RetrievedChunk(
            chunk_id="1706.03762v7_p0003_c001",
            paper="Attention Is All You Need",
            citation_name="Attention Is All You Need",
            arxiv_id="1706.03762v7",
            page=3,
            text="Scaled dot-product attention.",
            semantic_score=0.7,
            final_score=0.7,
        )
        self.moshi = RetrievedChunk(
            chunk_id="2410.00037v1_p0006_c001",
            paper="Moshi: a speech-text foundation model for real-time dialogue",
            citation_name="Moshi",
            arxiv_id="2410.00037v1",
            page=6,
            text="Moshi architecture excerpt.",
            semantic_score=0.7,
            final_score=0.7,
        )

    def test_valid_citation(self):
        result = validate_citations(
            "The paper describes attention [Attention Is All You Need, p. 3].", [self.chunk]
        )
        self.assertTrue(result.valid)

    def test_citation_without_space_after_period_is_valid(self):
        result = validate_citations("Scaled dot-product attention uses Q and K [Attention Is All You Need, p.3].", [self.chunk])
        self.assertTrue(result.valid)

    def test_page_word_variant_is_valid(self):
        result = validate_citations("Moshi uses these components [Moshi, page 6].", [self.moshi])
        self.assertTrue(result.valid)

    def test_hard_wrapped_paragraph_is_one_validation_block(self):
        answer = (
            "Scaled dot-product attention computes compatibility scores between queries and keys,\n"
            "then uses those scores to weight the values [Attention Is All You Need, p. 3]."
        )
        result = validate_citations(answer, [self.chunk])
        self.assertTrue(result.valid, result.reason)

    def test_bullets_are_validated_individually(self):
        answer = (
            "The model has two components:\n"
            "- Query-key compatibility is computed first [Attention Is All You Need, p. 3].\n"
            "- Values are combined using the attention weights [Attention Is All You Need, p. 3]."
        )
        result = validate_citations(answer, [self.chunk])
        self.assertTrue(result.valid, result.reason)

    def test_bullet_without_own_citation_is_covered_by_section_citation(self):
        answer = (
            "**Components**\n\n"
            "- Query-key compatibility is computed first.\n"
            "- Values are combined using attention weights.\n\n"
            "[Attention Is All You Need, p. 3]"
        )
        result = validate_citations(answer, [self.chunk])
        self.assertTrue(result.valid, result.reason)

    def test_non_retrieved_page_rejected(self):
        result = validate_citations("The paper says so [Attention Is All You Need, p. 4].", [self.chunk])
        self.assertFalse(result.valid)
        self.assertEqual(result.invalid, ["[Attention Is All You Need, p. 4]"])

    def test_non_retrieved_paper_rejected(self):
        result = validate_citations("This is stated [Moshi, p. 3].", [self.chunk])
        self.assertFalse(result.valid)

    def test_substantive_uncited_answer_rejected(self):
        result = validate_citations("The Transformer uses an encoder and decoder.", [self.chunk])
        self.assertFalse(result.valid)

    def test_clear_refusal_without_citation_allowed(self):
        result = validate_citations("The retrieved passages do not state this.", [self.chunk])
        self.assertTrue(result.valid)

    def test_refusal_sentence_does_not_hide_uncited_factual_block(self):
        answer = (
            "The Transformer uses an encoder and decoder.\n\n"
            "The provided papers do not specify the exact deployment hardware."
        )
        result = validate_citations(answer, [self.chunk])
        self.assertFalse(result.valid)

    def test_malformed_citation_detected(self):
        result = validate_citations("The paper says so [Attention Is All You Need, pp. 3-4].", [self.chunk])
        self.assertFalse(result.valid)
        self.assertTrue(result.malformed)

    def test_paragraphs_in_one_section_can_share_a_citation(self):
        answer = (
            "The Transformer uses an encoder and decoder.\n\n"
            "The encoder and decoder contain attention sublayers.\n\n"
            "[Attention Is All You Need, p. 3]"
        )
        result = validate_citations(answer, [self.chunk])
        self.assertTrue(result.valid, result.reason)

    def test_uncited_separate_section_is_rejected(self):
        answer = (
            "## Encoder\n\n"
            "The encoder uses self-attention.\n\n"
            "[Attention Is All You Need, p. 3]\n\n"
            "## Decoder\n\n"
            "The decoder uses masked self-attention."
        )
        result = validate_citations(answer, [self.chunk])
        self.assertFalse(result.valid)
        self.assertIn("section", result.reason.lower())

    def test_display_equation_with_same_block_citation_is_valid(self):
        answer = (
            r"\[ \operatorname{Attention}(Q,K,V)="
            r"\operatorname{softmax}(QK^\top/\sqrt{d_k})V \] "
            "[Attention Is All You Need, p. 3]"
        )
        result = validate_citations(answer, [self.chunk])
        self.assertTrue(result.valid, result.reason)

    def test_display_equation_with_immediately_following_citation_paragraph_is_valid(self):
        answer = (
            r"\[ \operatorname{Attention}(Q,K,V)="
            r"\operatorname{softmax}(QK^\top/\sqrt{d_k})V \]"
            "\n\n[Attention Is All You Need, p. 3]"
        )
        result = validate_citations(answer, [self.chunk])
        self.assertTrue(result.valid, result.reason)

    def test_display_equation_with_citation_on_next_line_is_valid(self):
        answer = (
            r"\[ \operatorname{Attention}(Q,K,V)="
            r"\operatorname{softmax}(QK^\top/\sqrt{d_k})V \]"
            "\n[Attention Is All You Need, p. 3]"
        )
        result = validate_citations(answer, [self.chunk])
        self.assertTrue(result.valid, result.reason)

    def test_uncited_display_equation_is_rejected(self):
        answer = (
            r"\[ \operatorname{Attention}(Q,K,V)="
            r"\operatorname{softmax}(QK^\top/\sqrt{d_k})V \]"
        )
        result = validate_citations(answer, [self.chunk])
        self.assertFalse(result.valid)
        self.assertTrue(result.uncited_blocks)

    def test_display_equation_with_non_retrieved_citation_is_rejected(self):
        answer = (
            r"\[ \operatorname{Attention}(Q,K,V)="
            r"\operatorname{softmax}(QK^\top/\sqrt{d_k})V \]"
            "\n\n[Attention Is All You Need, p. 4]"
        )
        result = validate_citations(answer, [self.chunk])
        self.assertFalse(result.valid)
        self.assertEqual(result.invalid, ["[Attention Is All You Need, p. 4]"])

    def test_display_equation_and_prose_can_share_section_citation(self):
        answer = (
            r"\[ QK^\top \]"
            "\n\nThe equation represents query-key compatibility. "
            "[Attention Is All You Need, p. 3]"
        )
        result = validate_citations(answer, [self.chunk])
        self.assertTrue(result.valid, result.reason)

    def test_display_math_environment_with_citation_is_valid(self):
        answer = (
            r"\begin{equation} QK^\top \end{equation}"
            "\n\n[Attention Is All You Need, p. 3]"
        )
        result = validate_citations(answer, [self.chunk])
        self.assertTrue(result.valid, result.reason)

    def test_dollar_display_equation_with_following_citation_is_valid(self):
        answer = (
            "$$ QK^\\top $$"
            "\n\n[Attention Is All You Need, p. 3]"
        )
        result = validate_citations(answer, [self.chunk])
        self.assertTrue(result.valid, result.reason)

    def test_align_environment_with_following_citation_is_valid(self):
        answer = (
            r"\begin{align} QK^\top &= S \end{align}"
            "\n\n[Attention Is All You Need, p. 3]"
        )
        result = validate_citations(answer, [self.chunk])
        self.assertTrue(result.valid, result.reason)


if __name__ == "__main__":
    unittest.main()

    def test_one_citation_covers_multiple_paragraphs_in_same_section(self):
        answer = (
            "The encoder contains self-attention and feed-forward sublayers.\n\n"
            "The decoder also uses masked self-attention and encoder-decoder attention.\n\n"
            "[Attention Is All You Need, p. 3]"
        )
        result = validate_citations(answer, self.retrieved)
        self.assertTrue(result.valid, result.reason)

    def test_bold_section_heading_and_equation_with_one_section_citation(self):
        answer = (
            "**Scaled dot-product attention**\n\n"
            "The operation computes compatibility scores between queries and keys.\n\n"
            r"\[ \operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^T/\sqrt{d_k})V \]"
            "\n\n"
            "The scaling term controls the magnitude of the dot products.\n\n"
            "[Attention Is All You Need, p. 4]"
        )
        result = validate_citations(answer, self.retrieved)
        self.assertTrue(result.valid, result.reason)

    def test_bullets_can_share_section_citation(self):
        answer = (
            "**Components**\n\n"
            "- The encoder uses self-attention.\n"
            "- The decoder uses masked self-attention.\n"
            "- The decoder attends to encoder outputs.\n\n"
            "[Attention Is All You Need, p. 3]"
        )
        result = validate_citations(answer, self.retrieved)
        self.assertTrue(result.valid, result.reason)

    def test_each_heading_section_needs_a_citation(self):
        answer = (
            "## Encoder\n\n"
            "The encoder uses self-attention.\n\n"
            "[Attention Is All You Need, p. 3]\n\n"
            "## Decoder\n\n"
            "The decoder uses masked self-attention."
        )
        result = validate_citations(answer, self.retrieved)
        self.assertFalse(result.valid)
        self.assertIn("section", result.reason.lower())

    def test_wrong_page_still_rejected_under_relaxed_policy(self):
        answer = (
            "The encoder uses self-attention.\n\n"
            "The decoder uses masked self-attention.\n\n"
            "[Attention Is All You Need, p. 999]"
        )
        result = validate_citations(answer, self.retrieved)
        self.assertFalse(result.valid)
        self.assertTrue(result.invalid)

    def test_answer_with_no_citations_still_rejected(self):
        answer = "The Transformer uses attention instead of recurrence."
        result = validate_citations(answer, self.retrieved)
        self.assertFalse(result.valid)

