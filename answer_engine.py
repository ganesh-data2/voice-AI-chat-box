from faq_processor import FAQProcessor

class AnswerEngine:
    def __init__(self, faq_pdf="ilovepdf_merged.pdf"):
        self.processor = FAQProcessor(faq_pdf)

    def get_answer(self, question: str) -> str:
        if not question.strip():
            return "❗ Please ask a valid question."

        try:
            answer, score = self.processor.get_best_answer(question)
            if answer:
                return answer
            else:
                return (
                    "🤖 I do not have an answer to that question right now. "
                    "Please contact a live agent."
                )
        except Exception as e:
            return f"⚠️ Error processing your question: {e}"

