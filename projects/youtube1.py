import dspy 

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

class QuestionAnswer(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

question_answer = dspy.Predict(QuestionAnswer)

question = "What is the capital of France?"
# answer = question_answer(question=question)

# print(answer.answer)

generate_answer = dspy.ChainOfThought(QuestionAnswer)
answer = generate_answer(question=question)

print(answer.answer)
print("\nReasoning:\n")
print(answer.reasoning)