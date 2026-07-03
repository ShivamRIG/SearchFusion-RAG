# app/evaluation.py

from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import settings


class Evaluator:

    def __init__(self):

        self.llm = ChatOpenAI(
            model=settings.CHAT_MODEL,
            temperature=0
        )

        self.faithfulness_prompt = ChatPromptTemplate.from_template(
            """
You are evaluating an AI assistant.

Context:
{context}

Answer:
{answer}

Question:
{question}

Does the answer stay faithful to the context?

Return ONLY a score between 0 and 1.
"""
        )

        self.relevance_prompt = ChatPromptTemplate.from_template(
            """
Question:
{question}

Answer:
{answer}

How relevant is the answer to the user's question?

Return ONLY a score between 0 and 1.
"""
        )



    def _score(self, prompt):

        response = self.llm.invoke(prompt)

        try:
            return float(response.content.strip())
        except Exception:
            return 0.0


    def faithfulness(
        self,
        question: str,
        context: List[str],
        answer: str,
    ):

        prompt = self.faithfulness_prompt.invoke(
            {
                "question": question,
                "context": "\n\n".join(context),
                "answer": answer,
            }
        )

        return self._score(prompt)



    def relevance(
        self,
        question: str,
        answer: str,
    ):

        prompt = self.relevance_prompt.invoke(
            {
                "question": question,
                "answer": answer,
            }
        )

        return self._score(prompt)


    def evaluate(
        self,
        question: str,
        context: List[str],
        answer: str,
    ) -> Dict:

        faithfulness = self.faithfulness(
            question,
            context,
            answer,
        )

        relevance = self.relevance(
            question,
            answer,
        )

        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
        }