import dspy 

from constants.env import OPENROUTER_API_KEY
from constants.env import OPENROUTER_BASE_URL
from constants.model import OpenRouterModel

openai_model = OpenRouterModel.OPENROUTER_OPENAI_MODEL
deepseek_model = OpenRouterModel.OPENROUTER_DEEPSEEK_MODEL

llm = dspy.LM(
    model=deepseek_model,
    api_key=OPENROUTER_API_KEY,
    api_base=OPENROUTER_BASE_URL
)

dspy.configure(lm=llm)

# FIXED tools
def evaluate_math(expression: str) -> float:
    return dspy.PythonInterpreter({}).execute(expression)

def search_wikipedia(query: str) -> str:
    colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    results = colbert(query=query, k=3)  
    return [x['text'] for x in results]

# ReAct agent
react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])
pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
print(pred.answer)  
print("\nReasoning\n")
print(pred.reasoning)