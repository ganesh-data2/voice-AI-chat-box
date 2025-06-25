import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FAQProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.qa_pairs = self.extract_qa_pairs()

        if not self.qa_pairs:
            raise ValueError("❌ No valid Q&A pairs found. Please check your PDF formatting.")

        self.questions = [q for q, _ in self.qa_pairs]
        self.answers = [a for _, a in self.qa_pairs]
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def extract_qa_pairs(self):
        doc = fitz.open(self.pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        lines = text.splitlines()

        qa_pairs = []
        current_q = None
        current_a = []

        question_pattern = re.compile(
            r"^(How|What|Why|When|Who|Can|Is|Do|Does|Are|Could|Would|Will|Shall)\b", re.IGNORECASE
        )

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.endswith("?") or question_pattern.match(line):
                if current_q and current_a:
                    qa_pairs.append((current_q.strip(), " ".join(current_a).strip()))
                current_q = line
                current_a = []
            elif current_q:
                current_a.append(line)

        if current_q and current_a:
            qa_pairs.append((current_q.strip(), " ".join(current_a).strip()))

        return qa_pairs

    def get_best_answer(self, user_question, threshold=0.1):
        user_vec = self.vectorizer.transform([user_question])
        similarities = cosine_similarity(user_vec, self.question_vectors)
        best_idx = similarities.argmax()
        best_score = similarities[0][best_idx]

        print(f"[DEBUG] Question: {user_question} | Score: {best_score:.2f}")
        if best_score >= threshold:
            return self.answers[best_idx], best_score
        else:
            return None, best_score


# ✅ Simple wrapper for app.py
faq_model = FAQProcessor("ilovepdf_merged.pdf")

def get_faq_answer(user_question: str) -> str:
    answer, score = faq_model.get_best_answer(user_question)
    if answer:
        return answer
    return "No relevant answer found in the FAQs."


