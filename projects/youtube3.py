import dspy 
from typing import Any
from pydantic import BaseModel
from pydantic import Field
from constants.env import OPENROUTER_API_KEY
from constants.env import OPENROUTER_BASE_URL
from constants.model import OpenRouterModel

openai_model = OpenRouterModel.OPENROUTER_OPENAI_MODEL
deepseek_model = OpenRouterModel.OPENROUTER_DEEPSEEK_MODEL

llm = dspy.LM(
    model=openai_model,
    api_key=OPENROUTER_API_KEY,
    api_base=OPENROUTER_BASE_URL
)

dspy.configure(lm=llm)

class AnswerConfidence(BaseModel):
    answer: str = Field(description="Answer, 1-5 words")
    confidence: float = Field(description="Your confidence between 0-1", ge=0, le=1)

class QAWithConfidence(dspy.Signature):
    question: str = dspy.InputField()
    answer: AnswerConfidence = dspy.OutputField()

predict = dspy.ChainOfThought(QAWithConfidence)
question = "who scored maximum runs in the world cup finals when india won the cricket world cup"
output = predict(question=question)
print(output)

