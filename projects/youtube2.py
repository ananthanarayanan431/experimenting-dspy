import dspy 
from typing import Any
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
    "Generate a question and answer"
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class ChainOfThoughtQuestionAnswer(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought(QuestionAnswer)

    def forward(self, question: str):
        return self.cot(question=question)

# question_answer = ChainOfThoughtQuestionAnswer()
# question = "What is the capital of France?"
# answer = question_answer(question=question)

# print(answer.answer)
# print("\nReasoning:\n")
# print(answer.reasoning)

class QuestionToThought(dspy.Signature):
    "Generate a step by step thoughts for the given question"
    question: str = dspy.InputField()
    step_by_step_thoughts: Any = dspy.OutputField()

class OneWordAnswer(dspy.Signature):
    "Generate a one word answer with the given question and the thoughts"
    question: str = dspy.InputField()
    thoughts: Any = dspy.InputField()
    one_word_answer: str = dspy.OutputField()

class OutputPrediction(dspy.Signature):
    thoughts: Any = dspy.InputField()
    answer: Any = dspy.InputField()
    response: str = dspy.OutputField()


class DoubleChainOfThoughtQuestionAnswer(dspy.Module):
    def __init__(self):
        self.cot1 = dspy.ChainOfThought(QuestionToThought)
        self.cot2 = dspy.ChainOfThought(OneWordAnswer)
    
    def forward(self, question:str):
        thoughts = self.cot1(question=question).step_by_step_thoughts
        answer = self.cot2(question=question, thoughts=thoughts).one_word_answer
        value = dspy.Predict(OutputPrediction)
        return value(thoughts=thoughts, answer=answer)

question_answer = DoubleChainOfThoughtQuestionAnswer()
question = "What is the capital of India and who is the captain of India when india won the cricket world cup and tell me about him"
answer = question_answer(question=question)

print("\nAnswer\n")
print(answer.response)