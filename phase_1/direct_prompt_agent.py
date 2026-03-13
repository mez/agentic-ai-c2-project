# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# TODO: 2 - Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(openai_api_key)

direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)

print(f"The response was generated using the DirectPromptAgent with no additional knowledge source, relying solely on the prompt provided by the user.")
