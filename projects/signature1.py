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

toxicity_signature = dspy.Signature(
    "comment -> toxic: bool",
    instructions="Mark as 'toxic' if the comment includes insults, harassment, or sarcastic derogatory remarks.",
)

toxicity = dspy.Predict(toxicity_signature)

comment = "you're beautiful"
val = toxicity(comment=comment).toxic 

print(val)
dspy.inspect_history()