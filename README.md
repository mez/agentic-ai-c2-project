## AI-Powered Agentic Workflow for Project Management

### Introduction
Welcome to the AI-Powered Agentic Workflow for Project Management! This project positions you as an AI Workflow Architect, specializing in intelligent agentic systems that dynamically manage and automate project tasks. Your client, InnovateNext Solutions—a fast-growing startup—faces challenges with inconsistent project execution and seeks a revolutionary approach to manage their product development lifecycle.

### The Challenge: Building a Scalable Engine for Innovation
Technical Project Managers (TPMs) at InnovateNext Solutions are overwhelmed, leading to miscommunications, inconsistent output, and project delays. The company needs a foundational, AI-driven project management framework that can be applied across all teams and products.

### Your Role
As the AI Workflow Architect, your responsibilities are:
1. **Construct a robust library of reusable AI agents**—the core toolkit for advanced agentic systems.
2. **Deploy these agents to build a general-purpose agentic workflow for technical project management**, piloted on the "Email Router" product.

### Audience
This solution is designed for technical project managers and the leadership team at InnovateNext Solutions, including the Head of Product and Lead Technical Program Manager. The goal is to provide a scalable system that supports both the current pilot and future product development efforts.

---

## Deliverables

### Phase 1: The Agentic Toolkit
- A Python package (`workflow_agents`) containing seven individually tested agent classes:
  - `DirectPromptAgent`
  - `AugmentedPromptAgent`
  - `KnowledgeAugmentedPromptAgent`
  - `RAGKnowledgePromptAgent` (provided)
  - `EvaluationAgent`
  - `RoutingAgent`
  - `ActionPlanningAgent`
- Standalone test scripts for each agent, with screenshots of successful test runs.

### Phase 2: Project Management Workflow Implementation
- A primary Python script (`agentic_workflow.py`) orchestrating selected agents to perform multi-step technical project management tasks.
- For the pilot ("Email Router"), the workflow will:
  1. Accept a high-level prompt and the product specification as input.
  2. Use an Action Planning Agent to break down the goal into sub-tasks.
  3. Use a Routing Agent to assign sub-tasks to specialized agent teams.
  4. Simulate three teams:
     - **Product Manager team:** Generates user stories (KnowledgeAugmentedPromptAgent + EvaluationAgent)
     - **Program Manager team:** Defines product features (KnowledgeAugmentedPromptAgent + EvaluationAgent)
     - **Development Engineer team:** Creates engineering tasks (KnowledgeAugmentedPromptAgent + EvaluationAgent)
  5. Produce a final, structured output representing the planned project for the Email Router.

---

This project demonstrates a scalable, AI-driven approach to project management, ensuring consistent, high-quality output for InnovateNext Solutions and beyond.
