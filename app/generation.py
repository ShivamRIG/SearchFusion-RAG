from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import settings


class Generator:

    def __init__(self):

        # LLM
        self.llm = ChatOpenAI(
            model=settings.CHAT_MODEL,
            temperature=0
        )

        # Prompt template (professional RAG style)
        self.prompt = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant.

You must answer ONLY using the provided context.

If the answer is not in the context, say: "I don't know based on the provided data."

---

Context:
{context}

---

Question:
{question}

---

Answer:
"""
        )

    def build_context(self, docs):

        """
        Convert retrieved docs into clean context string
        """

        return "\n\n".join(docs)


    def create_prompt(self, question: str, docs):

        context = self.build_context(docs)

        return self.prompt.invoke(
            {
                "context": context,
                "question": question,
            }
        )


    def generate(self, question: str, docs):

        """
        Full generation pipeline
        """

        prompt = self.create_prompt(question, docs)

        response = self.llm.invoke(prompt)

        return response.content