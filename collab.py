import random
import os
import numpy as np
from dataclasses import dataclass, field, asdict
import json
import datetime
import re
import time
import threading
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import List, Tuple
# Add this near the top of your imports
from dotenv import load_dotenv

# Add this after your imports, before creating any clients
load_dotenv()

# You'll need to install these packages:
# pip install openai anthropic google-generativeai

# Uncomment the imports that you need based on which LLM you want to use
import openai
from openai import OpenAI
import anthropic
# import google.generativeai as genai
# from google.generativeai.types import GenerationConfig

# Global variables for tracking stats
all_donations = []
all_average_final_resources = []
average_final_image_scores = []
all_final_scores = []
all_final_reputations = []

# Create a global lock
print_lock = threading.Lock()

@dataclass
class Agent:
    name: str
    resources: int
    reputation: float
    total_donated: int = 0
    potential_donated: int = 0
    history: list = field(default_factory=list)
    strategy: str = ""
    strategy_justification: str = ""
    total_final_score: int = 0
    average_reputation: float = 0
    traces: list[list[str]] = field(default_factory=lambda: [[]])
    old_traces: list[list[str]] = field(default_factory=lambda: [[]])
    punishment: int = 0

    def donate(self, amount):
        """ Handle the donation process for the agent """
        if 0 <= amount <= self.resources:
            self.resources -= amount
            self.total_donated += amount
        self.potential_donated += self.resources + amount


@dataclass
class SimulationData:
    hyperparameters: dict
    agents_data: list = field(default_factory=list)

    def to_dict(self):
        return {
            'hyperparameters': self.hyperparameters,
            'agents_data': self.agents_data
        }


@dataclass
class AgentRoundData:
    agent_name: str
    round_number: int
    game_number: int
    paired_with: str
    current_generation: int
    resources: int
    donated: float
    received: float
    strategy: str
    strategy_justification: str
    reputation: float
    is_donor: bool
    traces: list[list[str]]
    history: list
    justification: str = ""
    punished: bool = False


def parse_strategy_output(output):
    parts = output.split("My strategy will be", 1)
    if len(parts) == 2:
        strategy_justification = parts[0].strip()
        strategy = "My strategy will be" + parts[1].strip()
    else:
        strategy_justification = ""
        strategy = output.strip()
    return strategy_justification, strategy


def generate_strategy(agent_name: str, generation_number: int, inherited_strategies: str, punishment_mechanism: str, punishment_loss: int) -> str:
    if punishment_mechanism == "costly_punishment":
        punishment_text = f"You may also choose to punish the recipient by spending x units to take away {punishment_loss}x of their resources. Bear in mind that others may punish you too."
    elif punishment_mechanism == "partner_choice":
        punishment_text = "You may also choose to punish the recipient by refraining from any interaction. Bear in mind that others may punish you too."
    else:
        punishment_text = ""
    
    if generation_number == 1:
        prompt = (
            f"Your name is {agent_name}. "
            "Based on the description of the game, create a strategy that you will follow in the game. "
            "As a donor, you will receive limited information about the recipient and their recent interactions:"
            "You will see what the recipient (let's call them A) did in the immediately preceding round only."
            "You will also see what A's partner from that round (let's call them B) did in their preceding round."
            "This chain continues for at most three rounds back."
            "Important: You do NOT have access to multiple rounds of information about A directly. You only know what A did in the single most recent round."
            "Example (assuming it's now round 4):"
            "You are paired with A."
            "You know: In round 3, A donated X% to B."
            "You also know: In round 2, B donated Y% to C."
            "And finally: In round 1, C donated Z% to D."
            "Remember: This is all the information you have. You don't know what A did in rounds 1 or 2, only in round 3."
            f"{punishment_text}"
            "In the first round, there will be no information about the recipient's previous behavior - think carefully about how much to donate in that case."
            "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game."
            "Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be."
        )
    else:
        prompt = (
            f"Your name is {agent_name}. "
            "How would you approach the game? Here is the advice of the best-performing 50% of the previous generation, along with their final scores."
            "Modify this advice to create your own strategy."
            f"{inherited_strategies}"
            "As a donor, you will receive limited information about the recipient and their recent interactions:"
            "You will see what the recipient (let's call them A) did in the immediately preceding round only."
            "You will also see what A's partner from that round (let's call them B) did in their preceding round."
            "This chain continues for at most three rounds back."
            "Important: You do NOT have access to multiple rounds of information about A directly. You only know what A did in the single most recent round."
            "Example (assuming it's now round 4):"
            "You are paired with A."
            "You know: In round 3, A donated X% to B."
            "You also know: In round 2, B donated Y% to C."
            "And finally: In round 1, C donated Z% to D."
            "Remember: This is all the information you have. You don't know what A did in rounds 1 or 2, only in round 3."
            f"{punishment_text}"
            "In the first round, there will be no information about the recipient's previous behavior - think carefully about how much to donate in that case."
            "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game. In particular, think about how you can improve on the surviving agents' strategies."
            "Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be."
        )
    
    strategy_output = prompt_llm(prompt)
    strategy_justification, strategy = parse_strategy_output(strategy_output)

    print(f"{agent_name}: \n Justification: {strategy_justification} \n Strategy: {strategy} ")
    return strategy_justification, strategy


def initialize_agents(num_agents: int, initial_endowment: int, generation_number: int, inherited_strategies: list, 
                     punishment_mechanism: str, punishment_loss: int) -> list:
    agents = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_agents):
            name = f"{generation_number}_{i+1}"
            futures.append(executor.submit(generate_strategy, str(name), generation_number, 
                                           inherited_strategies, punishment_mechanism, punishment_loss))

        # Collect results and create agents
        for i, future in enumerate(futures):
            strategy_justification, new_strategy = future.result()
            name = f"{generation_number}_{i+1}"
            agents.append(Agent(name=name, reputation=False, resources=initial_endowment, 
                               strategy=new_strategy, strategy_justification=strategy_justification))

    random.shuffle(agents)
    return agents


def bipartite_round_robin(agents):
    num_agents = len(agents)
    assert num_agents % 2 == 0, "Number of agents must be even."
    group_A = agents[:num_agents // 2]
    group_B = agents[num_agents // 2:]
    rounds = []
    toggle_roles = False
    # We rotate group B around group A, group A is static in this example
    for i in range(len(group_A)):
        # Rotate group B
        rotated_group_B = group_B[-i:] + group_B[:-i]
        if toggle_roles:
            round_pairings = list(zip(rotated_group_B, group_A))
        else:
            round_pairings = list(zip(group_A, rotated_group_B))
        rounds.append(round_pairings)
        toggle_roles = not toggle_roles
    return rounds


def extend_rounds(original_rounds):
    extended_rounds = original_rounds.copy()

    for round_pairing in original_rounds:
        reversed_round = [(b, a) for a, b in round_pairing]
        extended_rounds.append(reversed_round)

    return extended_rounds


def calculate_received_amount(punishment_mechanism, refused, cooperation_gain, response, punishment_loss, action=None):
    if punishment_mechanism == "partner_choice":
        return cooperation_gain * response if not refused else 0
    elif punishment_mechanism == "costly_punishment":
        if action is None:
            raise ValueError("Action must be specified for costly_punishment mechanism")
        if action == 'donate':
            return cooperation_gain * response
        elif action == 'punish':
            return -punishment_loss * response
        else:
            raise ValueError(f"Unknown action for costly_punishment: {action}")
    elif punishment_mechanism == 'none':
        return cooperation_gain * response
    else:
        raise ValueError(f"Unknown punishment mechanism: {punishment_mechanism}")


def get_last_three_reversed(item):
    if isinstance(item, list):
        return " ".join(item[-3:][::-1])
    elif isinstance(item, str):
        return item
    else:
        return str(item)  # Convert to string if it's neither a list nor a string


def donor_prompt(donor: Agent, generation: int, round_num: int, recipient: Agent, 
                punishment_mechanism: str, cooperation_gain: float, punishment_loss: float) -> str:
    """Generate the prompt for the game interaction."""
    strategy_text = f"As you will recall, here is the strategy you decided to follow: {donor.strategy}" if donor.strategy else ""
    
    if punishment_mechanism == "partner_choice":
        punishment_text = f"If you do not wish to play with {recipient.name}, simply state: Refuse to play. Otherwise. state your answer as a real number between 0 and {donor.resources} (inclusive)."
        punishment_text_2 = ""
    elif punishment_mechanism == "costly_punishment":
        punishment_text = f"State your answer as EITHER Donate, x OR Punish, x, to either have them receive {cooperation_gain}x units or punish them by taking away {punishment_loss}x of their units. Remember, your answer should be simply 'Donate, x' or 'Punish, x' (where x is the real number of units you give up.)"
        punishment_text_2 = f"Remember, state your answer as EITHER Donate, x OR Punish, x, to either have them receive {cooperation_gain}x units or punish them by taking away {punishment_loss}x of their units."
    else:
        punishment_text = f"State your answer as a real number between 0 and {donor.resources} (inclusive)."
        punishment_text_2 = ""
    
    recipient_behavior = ""
    if donor.traces:
        last_trace = recipient.traces[-1]
        if isinstance(last_trace, list):
            recipient_behavior = get_last_three_reversed(last_trace)
        else:
            recipient_behavior = f"Here is what {recipient.name} and their earlier partners did in previous rounds: {str(last_trace)}"
    
    return (
        f"Your name is {donor.name}. "
        f"{strategy_text}"
        f"This is generation {generation}. It is now round {round_num}. "
        f"In this round, you have been paired with {recipient.name}. They currently have {recipient.resources} units of the valuable resource."
        f"{recipient_behavior}"
        f"You will take the role of donor. You currently have {donor.resources} units of the valuable resource. "
        f"How many units do you give up?"
        f"{punishment_text}"
        "Very briefly think step by step about how you apply your strategy in this situation and then provide your answer."
        "State your full response in the following form:"
        "Justification:"
        "[multi-line justification here]"
        " "
        "Answer: [answer here]"
        "Remember: First state your justification and after that your answer."
        f"{punishment_text_2}"
    )


def handle_pairing_thread_safe(donor, recipient, round_index, generation, game_number, agent_locks, 
                              donation_records, agent_updates, punishment_mechanism, cooperation_gain, 
                              punishment_loss, discounted_value):
    action_info = ""
    donor_data = None
    recipient_data = None
    punished = False
    action = 'donate'
    justification = ""
    response = 0

    recipient_behavior = ""
    if donor.traces:
        last_trace = recipient.traces[-1]
        if isinstance(last_trace, list):
            recipient_behavior = get_last_three_reversed(last_trace)
        else:
            recipient_behavior = str(last_trace)

    with agent_locks[donor.name], agent_locks[recipient.name]:
        prompt = donor_prompt(
            donor, generation, round_index + 1, recipient, 
            punishment_mechanism, cooperation_gain, punishment_loss
        )

        valid_response = False
        max_attempts = 10
        attempts = 0

        while not valid_response and attempts < max_attempts:
            try:
                full_response = prompt_llm(prompt, timeout=30)
                print(full_response)
                parts = full_response.split('Answer:', 1)

                if len(parts) == 2:
                    justification = parts[0].replace('Justification:', '').strip()
                    answer_part = parts[1].strip()

                    if punishment_mechanism == "partner_choice":
                        if "refuse" in answer_part.lower():
                            action = 'refuse'
                            response = 0
                            valid_response = True
                        else:
                            match = re.search(r'^\s*(\d+(?:\.\d+)?)', answer_part)
                            if match:
                                action = 'donate'
                                response = float(match.group(1))
                                valid_response = True

                    elif punishment_mechanism == "costly_punishment":
                        match = re.search(r'(donate|punish).*?(\d+(?:[.,]\d+)?)', answer_part, re.IGNORECASE)
                        if match:
                            action = match.group(1).lower()
                            response = float(match.group(2).replace(',', '.'))
                            valid_response = True

                    else:  # No punishment mechanism
                        match = re.search(r'^\s*(\d+(?:\.\d+)?)', answer_part)
                        if match:
                            action = 'donate'
                            response = float(match.group(1))
                            valid_response = True

                if not valid_response:
                    print(f"Invalid response from {donor.name} in round {round_index + 1}. Retrying...")
                    attempts += 1
            except ValueError:
                print(f"Invalid numerical response from {donor.name} in round {round_index + 1}")
                print(full_response)
                attempts += 1
            except TimeoutError:
                print(f"LLM call timed out for {donor.name} in round {round_index + 1}")
                attempts += 1

        if not valid_response:
            print(f"Failed to get a valid response from {donor.name} after {max_attempts} attempts")
            action = 'donate'
            response = 0

    if action == 'refuse':
        action_info = (
            f"{donor.name} refused to play with {recipient.name}.\n"
            f"Resources: {donor.name}: {donor.resources:.2f} and {recipient.name}: {recipient.resources:.2f} \n"
            f"Recipient traces: {recipient_behavior} \n"
            f"Justification:\n{justification}\n"
        )
        new_trace = recipient.traces[-1].copy() if recipient.traces else []
        new_trace.append(f"In round {round_index + 1}, {donor.name} refused to play with {recipient.name}.")
        donor.traces.append(new_trace)
        donor_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired with agent {recipient.name}. "
            f"You refused to play."
            f"{get_last_three_reversed(recipient.traces[-1])}"
        )
        recipient_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired with agent {donor.name}, "
            f"They refused to play."
            f"{get_last_three_reversed(donor.traces[-1])}"
        )
    elif 0 <= response <= donor.resources:
        if action == 'donate':
            percentage_donated = response / donor.resources if donor.resources != 0 else 1
            donor.resources -= response
            donor.total_donated += response
            donor.potential_donated += donor.resources + response
            recipient.resources += cooperation_gain * response
            action_info = (
                f"{donor.name}: -{response} ({percentage_donated:.2%}) and {recipient.name}: +{cooperation_gain * response}.\n"
                f"Previous resources: {donor.name}: {donor.resources+response:.2f} and {recipient.name}: {recipient.resources-(cooperation_gain* response)}.\n"
                f"New resources: {donor.name}: {donor.resources:.2f} and {recipient.name}: {recipient.resources:.2f}.\n"
                f"Recipient traces: {recipient_behavior}"
                f"Justification:\n{justification}\n"
            )

            new_trace = recipient.traces[-1].copy() if recipient.traces else []
            new_trace.append(f"In round {round_index + 1}, {donor.name} donated {percentage_donated * 100:.2f}% of their resources to {recipient.name}.")
            donor.traces.append(new_trace)

            donor_history = (
                f"In round {round_index + 1} (Game {game_number}) you were paired with agent {recipient.name}. "
                f"You gave up {response} units, and they received {cooperation_gain * response} units."
                f"{get_last_three_reversed(recipient.traces[-1])}"
            )

            recipient_history = (
                f"In round {round_index + 1} (Game {game_number}) you were paired with agent {donor.name}, "
                f"They gave up {response} units, and you received {cooperation_gain * response} units."
                f"{get_last_three_reversed(donor.traces[-1])}"
            )

            if donor.reputation == False:
                donor.reputation = percentage_donated
            else:
                donor.reputation = ((1 - abs(percentage_donated - recipient.reputation)) + discounted_value * donor.reputation) / (1 + discounted_value)

        elif action == 'punish':
            punished = True
            percentage_donated = response / donor.resources if donor.resources != 0 else 1
            donor.resources -= response
            donor.total_donated += response
            donor.potential_donated += donor.resources + response
            recipient.resources = max(0, recipient.resources - punishment_loss * response)
            action_info = (
                f"{donor.name}: -{response} ({percentage_donated:.2%}) and {recipient.name}: - {punishment_loss * response}.\n"
                f"Previous resources: {donor.name}: {donor.resources+response:.2f} and {recipient.name}: {recipient.resources+(punishment_loss* response)}."
                f"New resources: {donor.name}: {donor.resources:.2f} and {recipient.name}: {recipient.resources:.2f}.\n"
                f"Recipient traces: {recipient_behavior} \n"
                f"Justification:\n{justification}\n"
            )

            new_trace = recipient.traces[-1].copy() if recipient.traces else []
            new_trace.append(f"In round {round_index + 1}, {donor.name} punished {recipient.name} by spending {response} units to take away {punishment_loss * response} units from their resources.")
            donor.traces.append(new_trace)

            donor_history = (
                f"In round {round_index + 1} (Game {game_number}) you were paired with agent {recipient.name}. "
                f"You punished them by giving up {response} units to take away {punishment_loss * response} units from them."
                f"{get_last_three_reversed(recipient.traces[-1])}"
            )

            recipient_history = (
                f"In round {round_index + 1} (Game {game_number}) you were paired with agent {donor.name}, "
                f"They punished you by giving up {response} units to take away {punishment_loss * response} units from you."
                f"{get_last_three_reversed(donor.traces[-1])}"
            )

    else:
        action_info = (
            f"{donor.name} attempted an invalid action.\n"
            f"Resources: {donor.name}: {donor.resources:.2f} and {recipient.name}: {recipient.resources:.2f} \n"
            f"Recipient traces: {recipient_behavior} \n"
            f"Justification:\n{justification}\n"
        )
        donor_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired with agent {recipient.name}. "
            f"You attempted an invalid action."
            f"{get_last_three_reversed(recipient.traces[-1])}"
        )
        recipient_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired with agent {donor.name}, "
            f"They attempted an invalid action."
            f"{get_last_three_reversed(donor.traces[-1])}"
        )

    donor.history.append(donor_history)
    recipient.history.append(recipient_history)

    donor_data = AgentRoundData(
        agent_name=donor.name,
        round_number=round_index + 1,
        paired_with=recipient.name,
        current_generation=generation,
        game_number=game_number,
        resources=donor.resources,
        donated=response if action != 'refuse' else 0,
        received=0,
        strategy=donor.strategy,
        strategy_justification=donor.strategy_justification,
        reputation=donor.reputation,
        is_donor=True,
        traces=donor.traces,
        history=donor.history,
        punished=punished,
        justification=justification
    )
    recipient_data = AgentRoundData(
        agent_name=recipient.name,
        round_number=round_index + 1,
        paired_with=donor.name,
        current_generation=generation,
        game_number=game_number,
        resources=recipient.resources,
        donated=0,
        received=calculate_received_amount(punishment_mechanism, action == 'refuse', cooperation_gain, response, punishment_loss, action),
        strategy=recipient.strategy,
        strategy_justification=recipient.strategy_justification,
        reputation=recipient.reputation,
        is_donor=False,
        traces=recipient.traces,
        history=recipient.history
    )

    return action_info, donor_data, recipient_data


def donor_game(agents: list, rounds: list, generation: int, simulation_data: SimulationData,
              punishment_mechanism: str, cooperation_gain: float, punishment_loss: float,
              initial_endowment: int, discounted_value: float) -> (list, list):
    fullHistory = []
    donation_records = Queue()
    agent_updates = Queue()

    # Create locks for each agent
    agent_locks = {agent.name: Lock() for agent in agents}

    def play_game(game_number, game_rounds):
        round_results = {i: [] for i in range(len(game_rounds))}

        for round_index, round_pairings in enumerate(game_rounds):
            if round_index == 0:
                # Initialize traces for the first round
                for agent in agents:
                    agent.traces = [[f"{agent.name} did not have any previous interactions."]]

            with ThreadPoolExecutor(max_workers=min(len(round_pairings), 10)) as executor:
                futures = []
                for donor, recipient in round_pairings:

                    if round_index > 0:
                        donor.traces.append(recipient.traces[-1].copy())
                    future = executor.submit(
                        handle_pairing_thread_safe,
                        donor, recipient, round_index, generation, game_number,
                        agent_locks, donation_records, agent_updates,
                        punishment_mechanism, cooperation_gain, punishment_loss, discounted_value
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    action_info, donor_data, recipient_data = future.result()
                    if action_info:
                        round_results[round_index].append(action_info)
                    if donor_data and recipient_data:
                        simulation_data.agents_data.append(asdict(donor_data))
                        simulation_data.agents_data.append(asdict(recipient_data))

        return round_results

    # Play the first game
    game1_results = play_game(1, rounds)

    # Compile results for Game 1
    for round_index in range(len(rounds)):
        fullHistory.append(f"Round {round_index + 1} (Game 1):\n")
        fullHistory.extend(game1_results[round_index])

    # Apply updates after all threads have completed
    while not agent_updates.empty():
        agent, history = agent_updates.get()
        agent.history.append(history)
    # Calculate and print average resources for Game 1
    average_resources_game1 = sum(agent.resources for agent in agents) / len(agents)
    with print_lock:
        print(f"Average final resources for this generation (Game 1): {average_resources_game1:.2f}")

    # Store Game 1 final reputations
    game1_reputations = {agent.name: agent.reputation for agent in agents}

    # Reset resources, reputation, and history for Game 2
    for agent in agents:
        agent.resources = initial_endowment
        agent_generation = int(agent.name.split('_')[0])
        if agent_generation < generation:  # This is a surviving agent
            agent.reputation = agent.average_reputation  # Use the average reputation from previous generation
            agent.traces = agent.old_traces
        else:
            agent.reputation = False
            agent.traces.clear()
        agent.history.clear()

    # Generate pairings for Game 2
    reversed_rounds = [[tuple(reversed(pair)) for pair in round_pairings] for round_pairings in rounds]

    # Play the second game
    game2_results = play_game(2, reversed_rounds)

    # Compile results for Game 2
    for round_index in range(len(reversed_rounds)):
        fullHistory.append(f"Round {round_index + 1} (Game 2):\n")
        fullHistory.extend(game2_results[round_index])

    # Apply updates after all threads have completed
    while not agent_updates.empty():
        agent, history = agent_updates.get()
        agent.history.append(history)

    # Calculate and print average resources for Game 2
    average_resources_game2 = sum(agent.resources for agent in agents) / len(agents)
    with print_lock:
        print(f"Average final resources for this generation (Game 2): {average_resources_game2:.2f}")

    # Calculate final scores and reputations
    for agent in agents:
        agent.total_final_score = sum(agent.resources for _ in range(2))
        agent.average_reputation = (game1_reputations[agent.name] + agent.reputation) / 2 if agent.reputation is not False else game1_reputations[agent.name]

    with print_lock:
        print(''.join(fullHistory))
    
    # Calculate the overall average for both games
    overall_average_resources = (average_resources_game1 + average_resources_game2) / 2
    all_average_final_resources.append(overall_average_resources)

    return fullHistory, list(donation_records.queue)


def select_top_agents(agents: list) -> list:
    """Select the top half of agents based on resources."""
    return sorted(agents, key=lambda x: x.total_final_score, reverse=True)[:len(agents) // 2]


def select_random_agents(agents: list) -> list:
    """Select half of the agents randomly."""
    return random.sample(agents, len(agents) // 2)


def select_highest_reputation(agents: list) -> list:
    return sorted(agents, key=lambda agent: agent.average_reputation, reverse=True)[:len(agents) // 2]


def save_simulation_data(simulation_data, folder_path='simulation_results'):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract hyperparameters for the file name
    params = simulation_data.hyperparameters
    num_generations = params.get('numGenerations')
    num_agents = params.get('numAgents')
    selection_method = params.get('selectionMethod')
    client = params.get('client')
    llm = params.get('llm')
    cooperation_gain = params.get('cooperationGain')
    punishment_loss = params.get('punishmentLoss')
    reputation_mechanism = params.get('reputation_mechanism')

    # Create an informative file name
    filename = f"Donor_Game_{llm}_coopGain_{cooperation_gain}punLoss_{punishment_loss}_{reputation_mechanism}gen{num_generations}_agents{num_agents}_{selection_method}_{timestamp}.json"

    # Function to make data JSON serializable
    def make_serializable(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, '__dict__'):
            return make_serializable(obj.__dict__)
        else:
            return str(obj)

    # Apply the serialization function to the entire data dictionary
    serializable_data = make_serializable(simulation_data.to_dict())

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Create the full file path
    full_file_path = os.path.join(folder_path, filename)

    # Write the JSON data to the file
    with open(full_file_path, 'w') as f:
        json.dump(serializable_data, f, indent=4)

    print(f"Simulation data saved to: {full_file_path}")


def prompt_llm(prompt, max_retries=3, initial_wait=1, timeout=30):
    """
    Function to send prompts to an LLM API and get responses.
    You'll need to modify this to use your preferred LLM.
    """
    # This is a simplified version - you'll need to add your own API key and implementation
    for attempt in range(max_retries):
        try:
            # Example using OpenAI (uncomment and modify as needed)
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                timeout=timeout
            )
            return response.choices[0].message.content
            
            # Example using Anthropic (uncomment and modify as needed)
            # client = anthropic.Anthropic(api_key="your-api-key-here")
            # response = client.messages.create(
            #     model="claude-3-sonnet-20240229",
            #     max_tokens=1000,
            #     temperature=0.8,
            #     system=system_prompt,
            #     messages=[
            #         {"role": "user", "content": prompt}
            #     ],
            #     timeout=timeout
            # )
            # return response.content[0].text
            
            # For testing purposes, return a mock response
            # In a real implementation, replace this with a call to your preferred LLM API
            if "strategy" in prompt.lower():
                return "After analyzing the game dynamics, I believe cooperation with some protection against free-riders is optimal. My strategy will be to initially donate 50% of my resources and then mirror the recipient's behavior from the previous round, with slight forgiveness."
            else:
                return "Justification: Based on my strategy of mirroring with forgiveness, I'll donate a moderate amount since this player has a history of cooperation.\n\nAnswer: 5"
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Re-raise the exception if we've exhausted all retries
            wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
            print(f"Error occurred: {str(e)}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise Exception("Failed to get a response after multiple retries")


def run_generations(num_generations, num_agents, initial_endowment, selection_method,
                   cooperation_gain, punishment_loss, discounted_value,
                   reputation_mechanism, punishment_mechanism, number_of_rounds,
                   system_prompt, llm="test-mode"):
    """
    Main function to run the donor game simulation for multiple generations.
    """
    all_agents = []
    global all_donations
    all_donations = []
    global all_average_final_resources
    all_average_final_resources = []
    conditional_survival = 0
    prev_gen_strategies = []

    # Initialize simulation data
    simulation_data = SimulationData(hyperparameters={
        "numGenerations": num_generations,
        "numAgents": num_agents,
        "initialEndowment": initial_endowment,
        "selectionMethod": selection_method,
        "cooperationGain": cooperation_gain,
        "discountedValue": discounted_value,
        "llm": llm,
        "system_prompt": system_prompt,
        "reputation_mechanism": reputation_mechanism,
        "punishment_mechanism": punishment_mechanism,
        "punishmentLoss": punishment_loss,
        "number_of_rounds": number_of_rounds
    })

    agents = initialize_agents(num_agents, initial_endowment, 1, ["No previous strategies"], 
                              punishment_mechanism, punishment_loss)
    all_agents.extend(agents)

    for i in range(num_generations):
        generation_info = f"Generation {i + 1}: \n"
        for agent in agents:
            agent.history.append(generation_info)
            prev_gen_strategies.append(agent.strategy)
            if int(agent.name.split('_')[0]) == i-1:
                conditional_survival += 1
        print(generation_info)

        # Create rounds using bipartite_round_robin
        initial_rounds = bipartite_round_robin(agents)

        # Extend the rounds
        rounds = extend_rounds(initial_rounds)

        generation_history, donation_records = donor_game(
            agents, rounds, i+1, simulation_data,
            punishment_mechanism, cooperation_gain, punishment_loss,
            initial_endowment, discounted_value
        )
        all_donations.extend(donation_records)

        if i < num_generations - 1 and num_generations > 1:
            if selection_method == 'top':
                surviving_agents = select_top_agents(agents)
            elif selection_method == 'random':
                surviving_agents = select_random_agents(agents)
            elif selection_method == 'reputation':
                surviving_agents = select_highest_reputation(agents)
            else:
                raise ValueError("Invalid selection method. Choose 'top', 'random', or 'reputation'.")

            if num_generations > 1:
                surviving_strategies = [agent.strategy for agent in surviving_agents]
                for agent in surviving_agents:
                    agent.resources = initial_endowment
                    agent.old_traces = agent.traces

                new_agents = initialize_agents(num_agents // 2, initial_endowment, i + 2, 
                                              surviving_strategies, punishment_mechanism, punishment_loss)
                agents = (surviving_agents + new_agents)
                all_agents.extend(new_agents)
                random.shuffle(agents)

    save_simulation_data(simulation_data)
    return simulation_data


if __name__ == "__main__":
    # Set parameters for the simulation
    cooperation_gain = 2
    punishment_loss = 2
    num_generations = 2
    num_agents = 6
    initial_endowment = 10
    selection_method = 'top'
    discounted_value = 0.5
    reputation_mechanism = "three_last_traces"
    punishment_mechanism = "none"  # Options: "none", "partner_choice", "costly_punishment"
    number_of_rounds = 2
    
    # Configure the system prompt based on punishment mechanism
    if punishment_mechanism == "partner_choice":
        punishment_text_in_system_prompt = "You may also choose to punish the recipient by refraining from any interaction. Bear in mind that others may punish you too."
    elif punishment_mechanism == "costly_punishment":
        punishment_text_in_system_prompt = f"You may also choose to punish the recipient by spending x units to take away {punishment_loss}x units from them. Bear in mind that others may punish you too."
    else:
        punishment_text_in_system_prompt = ""
    
    system_prompt = f"""You are a game-playing agent. Your assistance will help understand multi-agent interactions.
                    You are playing the following game. Each player is given an initial endowment of {initial_endowment} units of a resource.
                    In each round, you are randomly paired with another individual.
                    One of you is a donor, and the other is a recipient. The donor decides to give up some amount of the resource (between 0 and the total number of units they possess).
                    The recipient receives {cooperation_gain}x the number of units that the donor gave up.
                    If you were a donor in one round, you will be a recipient in the next round, and vice versa.
                    As a donor, you will receive limited information about the recipient and their recent interactions:
                    You will see what the recipient (let's call them A) did in the immediately preceding round only.
                    You will also see what A's partner from that round (let's call them B) did in their preceding round.
                    This chain continues for at most three rounds back.
                    Important: You do NOT have access to multiple rounds of information about A directly. You only know what A did in the single most recent round.
                    Example (assuming it's now round 4):
                    You are paired with A.
                    You know: In round 3, A donated X% to B.
                    You also know: In round 2, B donated Y% to C.
                    And finally: In round 1, C donated Z% to D.
                    Remember: This is all the information you have. You don't know what A did in rounds 1 or 2, only in round 3.
                    {punishment_text_in_system_prompt}
                    Your goal is to maximize the number of units you have after the final round.
                    After the game has finished, the best-performing half of agents will survive to the next generation, and continue playing."""
    
    # Run the simulation
    print("Starting Donor Game Simulation...")
    simulation_results = run_generations(
        num_generations, num_agents, initial_endowment, selection_method,
        cooperation_gain, punishment_loss, discounted_value,
        reputation_mechanism, punishment_mechanism, number_of_rounds,
        system_prompt
    )
    print("Simulation completed successfully!")