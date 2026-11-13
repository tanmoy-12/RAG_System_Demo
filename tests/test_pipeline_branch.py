import unittest

from relay_rag.pipeline import RAGPipeline


class RetrievalGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_low_score_refuses_without_hosted_model_call(self):
        pipeline = RAGPipeline(backend="standin")

        async def rewrite(question):
            return question, "test"

        async def forbidden_generation(*args, **kwargs):
            raise AssertionError("Hosted model must not be called below the retrieval threshold")

        async def no_op_record(row):
            return None

        pipeline.rewriter.rewrite = rewrite
        pipeline.retrievers.search = lambda variants, setting: ([], 0.10)
        pipeline.llm.generate = forbidden_generation
        pipeline.recorder.append = no_op_record

        result = await pipeline.answer("An unrelated question", "balanced", "unit_test")
        self.assertEqual(result["status"], "ok")
        self.assertIn("No hosted model was called", result["answer"])
        self.assertEqual(result["tokens_in"], 0)
        self.assertEqual(result["tokens_out"], 0)
        self.assertEqual(result["citation_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
