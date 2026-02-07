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


signature = dspy.Signature("question -> answer")
system_message = dspy.ChatAdapter().format_system_message(signature)
print(system_message)
