from typing import List

import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
from openai import OpenAI

# quick helper function to create an OpenAI client vocareum instance with the provided API key and base URL
def get_openai_client(api_key):
    """Helper function to create an OpenAI client instance."""
    return OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")

class BaseAgent:
    def __init__(self, openai_api_key: str, description: str = None):
        self.openai_api_key = openai_api_key
        # Use provided description or fallback to class docstring
        self.description = description if description is not None else self.__class__.__doc__ or ""

    def respond(self, prompt) -> str | List[str]:
        raise NotImplementedError("Subclasses must implement this method.")
    
class DirectPromptAgent(BaseAgent):
    """An agent that generates responses based on a provided prompt, without relying on its own pre-existing knowledge."""
    def __init__(self, openai_api_key: str, description: str = None):
        """Initialize the agent with the OpenAI API key.
            Parameters:
            openai_api_key (str): API key for accessing OpenAI.
            description (str): Optional description override.
        """
        super().__init__(openai_api_key, description)

    def respond(self, prompt):
        """Generate a response using the OpenAI API."""
        client = getOpenAIClient(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
        

class AugementedPromptAgent(BaseAgent):
    """An agent that generates responses based on a provided persona, without relying on its own pre-existing knowledge.

        parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
    """
    def __init__(self, openai_api_key: str, persona: str, description: str = None):
        """Initialize the agent with given attributes."""
        super().__init__(openai_api_key, description)
        self.persona = persona

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = getOpenAIClient(self.openai_api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}. Forget all previous context. Be sure to use your persona to respond accordingly"},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent(BaseAgent):
    """
    An agent that generates responses based on a provided persona and specific knowledge, 
    without relying on its own pre-existing knowledge.
    """
    def __init__(self, openai_api_key: str, persona: str, knowledge: str, description: str = None):
        super().__init__(openai_api_key, description)
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, input_text):
        """Generate a response using the OpenAI API.
        """
        client = getOpenAIClient(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona} knowledge-based assistant. Forget all previous context."},
                {"role": "system", "content": f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}"},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent(BaseAgent):
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key: str, persona: str, chunk_size: int = 2000, chunk_overlap: int = 100, description: str = None):
        super().__init__(openai_api_key, description)
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = getOpenAIClient(self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        text_length = len(text)
        print(f"Total text length: {text_length} characters. Chunk size: {self.chunk_size} characters. There will be a total of {max(1, (text_length - self.chunk_overlap) // (self.chunk_size - self.chunk_overlap))} chunks.")

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            if end >= len(text):
                break
            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = getOpenAIClient(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content


class EvaluationAgent(BaseAgent):
    """An agent that evaluates the response of a worker agent based on specific criteria, and provides feedback for improvement until a satisfactory solution is achieved or a maximum number of interactions is reached."""
    def __init__(self, openai_api_key: str, persona: str, evaluation_criteria: str, worker_agent: BaseAgent, max_interactions: int, description: str = None):
        super().__init__(openai_api_key, description)
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def respond(self, prompt):
        return self.evaluate(prompt)
    
    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        client = getOpenAIClient(self.openai_api_key)
        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are {self.persona}, an evaluation agent."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You are {self.persona}, an evaluation agent."},
                        {"role": "user", "content": instruction_prompt}
                    ],
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": i + 1
        }   


class RoutingAgent(BaseAgent):
    """
    An agent that routes user input to the most appropriate agent based on the similarity of the input to the descriptions of available agents, using embeddings for comparison.
    """
    def __init__(self, openai_api_key: str, agents: list[BaseAgent], description: str = None):
        super().__init__(openai_api_key, description)
        self.agents = agents
       

    def get_embedding(self, text):
        client = getOpenAIClient(self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
      
        # Extract and return the embedding vector from the response
        embedding = response.data[0].embedding
        return embedding 

    def respond(self, prompt):
        return self.route(prompt) 
    
    def route(self, user_input):
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent.description)
            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            # print(f"Similarity with {agent.description}: {similarity}")
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent.description} (score={best_score:.3f})")
        return best_agent.respond(user_input)



class ActionPlanningAgent(BaseAgent):
    """An agent that extracts actionable steps from a user prompt based on its knowledge, without relying on its own pre-existing knowledge."""
    def __init__(self, openai_api_key: str, knowledge: str, description: str = None):
        super().__init__(openai_api_key, description)
        self.knowledge = knowledge
       
    def respond(self, prompt):
        steps = self.extract_steps_from_prompt(prompt)
        return steps
    
    def extract_steps_from_prompt(self, prompt):
        client = getOpenAIClient(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {self.knowledge}"},
                {"role": "user", "content": prompt}
            ],  
            temperature=0
        )
       
        response_text = response.choices[0].message.content.strip()  
        
        steps = response_text.split("\n")
        steps = [step.strip() for step in steps if step.strip()]

        return steps
