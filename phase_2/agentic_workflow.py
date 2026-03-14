# agentic_workflow.py

from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
with open(os.path.join(os.path.dirname(__file__), "Product-Spec-Email-Router.txt"), "r") as file:
    product_spec = file.read().strip()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"Product Spec: {product_spec}"
)

product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager,
    description="A Product Manager who defines and writes user stories from a product specification."
)

# Product Manager - Evaluation Agent
# The evaluation_criteria should specify the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona="You are an evaluation agent that checks the answers of other worker agents.",
    evaluation_criteria="The answer should be user stories that follow this exact structure: " \
                        "As a [type of user], I want [an action or feature] so that [benefit/value].",
    worker_agent=product_manager_knowledge_agent,
    description="An evaluation agent that checks the quality of user stories defined by the Product Manager, ensuring they are well-structured and comprehensive.",
    max_interactions=3
)


# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager,
    description="You are a Program Manager, you are responsible for defining the features for a product."
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."


# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
evaluation_criteria = (
    "The answer should be product features that follow the following structure: " \
    "Feature Name: A clear, concise title that identifies the capability\n" \
    "Description: A brief explanation of what the feature does and its purpose\n" \
    "Key Functionality: The specific capabilities or actions the feature provides\n" \
    "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria,
    worker_agent=program_manager_knowledge_agent,
    description="An evaluation agent that checks the quality of product features defined by the Program Manager, ensuring they are well-structured and comprehensive.",
    max_interactions=10
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer,
    description="A Development Engineer who defines technical tasks required to implement user stories."
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
development_engineer_evaluation_criteria = (
    "The answer should be tasks following this exact structure: " \
    "Task ID: A unique identifier for tracking purposes\n" \
    "Task Title: Brief description of the specific development work\n" \
    "Related User Story: Reference to the parent user story\n" \
    "Description: Detailed explanation of the technical work required\n" \
    "Acceptance Criteria: Specific requirements that must be met for completion\n" \
    "Estimated Effort: Time or complexity estimation\n" \
    "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=development_engineer_evaluation_criteria,
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=10,
    description="An evaluation agent that checks the quality of development tasks defined by the Development Engineer, ensuring they are well-structured and comprehensive."
)


# Routing Agent
routing_agent = RoutingAgent(openai_api_key, [
    product_manager_knowledge_agent,
    program_manager_knowledge_agent,
    development_engineer_knowledge_agent
])

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
steps = action_planning_agent.respond(workflow_prompt)
print(f"Workflow steps: {steps}\n")

completed_steps = []
for i, step in enumerate(steps):
    print(f"\n--- Step {i + 1}: {step} ---")
    result = routing_agent.respond(step)
    completed_steps.append(result)
    print(f"Result:\n{result}")

print("\n*** Workflow execution completed ***\n")
print("Final output:\n")
print(completed_steps[-1])
