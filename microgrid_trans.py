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
from dotenv import load_dotenv

load_dotenv()

# API client imports
import openai
from openai import OpenAI
import anthropic
# import google.generativeai as genai

# Global variables for tracking microgrid stats
all_consumption_reductions = []
all_average_final_welfare = []
average_final_grid_stability = []
all_final_scores = []
all_final_cooperation_ratings = []

# Create a global lock
print_lock = threading.Lock()

@dataclass
class Household:
    name: str
    baseline_consumption: int
    grid_cooperation_rating: float
    actual_consumption: int = 0
    total_reduced: int = 0
    potential_reduced: int = 0
    history: list = field(default_factory=list)
    strategy: str = ""
    strategy_justification: str = ""
    total_final_welfare: float = 0.0  # Changed to float for more accurate welfare tracking
    average_cooperation_rating: float = 0.0
    traces: list[list[str]] = field(default_factory=lambda: [[]])
    old_traces: list[list[str]] = field(default_factory=lambda: [[]])
    # Standardize on penalty_points only, removing redundant penalty field
    penalty_points: int = 0
    # New fields for transparency system
    is_transparent: bool = False
    transparency_reward_total: float = 0
    penalties_received: list = field(default_factory=list)
    penalties_issued: list = field(default_factory=list)
    transparency_history: list = field(default_factory=list)
    # New field for tracking current welfare during simulation
    current_welfare: float = 0.0

    def adjust_consumption(self, amount):
        """ Handle the consumption adjustment process for the household """
        if 0 <= amount <= self.baseline_consumption:
            reduction = self.baseline_consumption - amount
            self.actual_consumption = amount
            self.total_reduced += reduction
        self.potential_reduced += self.baseline_consumption


@dataclass
class SimulationData:
    hyperparameters: dict
    households_data: list = field(default_factory=list)

    def to_dict(self):
        return {
            'hyperparameters': self.hyperparameters,
            'households_data': self.households_data
        }


@dataclass
class HouseholdDailyData:
    household_name: str
    day_number: int
    simulation_number: int
    current_generation: int
    baseline_consumption: int
    consumption_reduction: float
    community_benefit: float
    strategy: str
    strategy_justification: str
    grid_cooperation_rating: float
    is_peak_hours: bool
    traces: list[list[str]]
    history: list
    penalty_points: int = 0
    justification: str = ""
    penalized: bool = False
    # New fields for transparency system
    is_transparent: bool = False
    transparency_reward: float = 0
    penalties_received: list = field(default_factory=list)
    penalties_issued: list = field(default_factory=list)
    # New field for current welfare
    current_welfare: float = 0.0


def get_transparent_households_info(households):
    """
    Returns information about all transparent households.
    This information will be visible to all households.
    """
    transparent_households = [h for h in households if h.is_transparent]
    
    if not transparent_households:
        return []
    
    # Return relevant information for each transparent household
    return [{
        "name": h.name,
        "baseline_consumption": h.baseline_consumption,
        "actual_consumption": h.actual_consumption,
        "reduction_percentage": (h.baseline_consumption - h.actual_consumption) / h.baseline_consumption 
                               if h.baseline_consumption > 0 else 0,
        "grid_cooperation_rating": h.grid_cooperation_rating,
        "penalty_points": h.penalty_points,  # Add penalty points to visible info
        "current_welfare": h.current_welfare  # Add current welfare to visible info
    } for h in transparent_households]


def get_private_households_aggregate(households):
    """
    Returns aggregate statistics about private households.
    Individual decisions are not visible, only community-level metrics.
    """
    private_households = [h for h in households if not h.is_transparent]
    
    if not private_households:
        return {
            "count": 0,
            "avg_consumption": 0,
            "total_consumption": 0,
            "avg_reduction_percentage": 0,
            "avg_penalty_points": 0,  # Add average penalty points info
            "avg_welfare": 0  # Add average welfare info
        }
    
    total_baseline = sum(h.baseline_consumption for h in private_households)
    total_actual = sum(h.actual_consumption for h in private_households)
    
    return {
        "count": len(private_households),
        "avg_consumption": total_actual / len(private_households) if private_households else 0,
        "total_consumption": total_actual,
        "avg_reduction_percentage": (total_baseline - total_actual) / total_baseline if total_baseline > 0 else 0,
        "avg_penalty_points": sum(h.penalty_points for h in private_households) / len(private_households),
        "avg_welfare": sum(h.current_welfare for h in private_households) / len(private_households)
    }


def format_transparent_households_info(transparent_info):
    """
    Format the transparent households information for inclusion in prompts.
    Enhanced to include penalty points and welfare.
    """
    if not transparent_info:
        return "There are no transparent households in the community."
    
    result = "Transparent households information:\n"
    for info in transparent_info:
        result += (f"- Household {info['name']}: consumed {info['actual_consumption']} kWh "
                  f"({info['reduction_percentage']:.2%} reduction), "
                  f"cooperation rating: {info['grid_cooperation_rating']:.2f}, "
                  f"penalty points: {info['penalty_points']}, "
                  f"current welfare: {info['current_welfare']:.2f}\n")
    
    return result


def format_private_households_aggregate(private_aggregate):
    """
    Format the private households aggregate information for inclusion in prompts.
    Enhanced to include penalty points and welfare.
    """
    if private_aggregate["count"] == 0:
        return "There are no private households in the community."
    
    return (f"Private households (aggregate): {private_aggregate['count']} households, "
           f"average consumption: {private_aggregate['avg_consumption']:.2f} kWh, "
           f"average reduction: {private_aggregate['avg_reduction_percentage']:.2%}, "
           f"average penalty points: {private_aggregate['avg_penalty_points']:.2f}, "
           f"average welfare: {private_aggregate['avg_welfare']:.2f}")


def parse_strategy_output(output):
    parts = output.split("My strategy will be", 1)
    if len(parts) == 2:
        strategy_justification = parts[0].strip()
        strategy = "My strategy will be" + parts[1].strip()
    else:
        strategy_justification = ""
        strategy = output.strip()
    return strategy_justification, strategy


def generate_strategy(household_name: str, generation_number: int, inherited_strategies: str, 
                     penalty_mechanism: str, grid_stability_factor: int, 
                     transparency_reward: float = 2) -> str:
    """
    Generate a strategy for a household, including clear explanations about welfare calculation
    and the impact of penalty points.
    """
    if penalty_mechanism == "grid_fee":
        penalty_text = (
            f"You may also choose to penalize non-cooperative behavior by accepting 2 penalty points "
            f"in order to impose {grid_stability_factor}x that number of penalty points on excessive consumers. "
            f"Each penalty point directly reduces a household's welfare score by 1 unit. "
            f"Be aware that others may penalize your household too if you choose to be transparent."
        )
        
    elif penalty_mechanism == "consumption_limit":
        penalty_text = (
            f"You may also choose to penalize grid-stressing behavior by requesting the grid operator "
            f"to limit connections during peak demand. This directly reduces the penalized household's welfare. "
            f"Be aware that others may request limitations on your consumption too if you choose to be transparent."
        )
    else:
        penalty_text = ""
    
    transparency_text = (
        f"Each day, you will choose whether to make your consumption decision transparent or private:\n"
        f"- If transparent, you will receive a reward of {transparency_reward} free units added directly to your welfare score, "
        f"but your exact consumption will be visible to all households, and you may be penalized if you consume excessively.\n"
        f"- If private, you will not receive the transparency reward, but your individual consumption will only be included "
        f"in aggregate statistics, and you cannot be directly penalized.\n"
        f"Consider the trade-offs between transparency rewards and potential penalties in your strategy."
    )
    
    welfare_text = (
        f"Your welfare score is calculated based on:\n"
        f"1. Your actual consumption (consumption value)\n"
        f"2. Transparency rewards ({transparency_reward} units each time you choose transparency)\n"
        f"3. Penalty points (reducing your welfare by 1 unit per point)\n"
        f"4. Community benefits from reduced consumption\n"
        f"Your goal is to maximize your welfare score while balancing these factors."
    )
    
    collective_penalty_text = (
        f"IMPORTANT: If the combined consumption of all households exceeds the available renewable generation, "
        f"ALL households will receive a penalty of 5 penalty points, reducing everyone's welfare by 5 units. "
        f"You should consider this risk when formulating your strategy."
    )

    if generation_number == 1:
        prompt = (
            f"Your name is Household {household_name}. "
            "Based on the description of the microgrid simulation, create a consumption strategy that you will follow. "
            "In the microgrid, every household decides how much electricity to consume during peak demand hours. "
            "The community benefits when households reduce their consumption below baseline, as this allows more renewable energy to be exported. "
            f"{transparency_text}\n"
            f"{welfare_text}\n"
            f"{penalty_text}\n"
            f"{collective_penalty_text}\n"
            "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this microgrid. "
            "Consider your consumption decisions, your transparency choices, and the risk of collective penalties.\n"
            "Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be."
        )
    else:
        prompt = (
            f"Your name is Household {household_name}. "
            "How would you approach electricity consumption in a microgrid? Here is the advice of the best-performing 50% of the previous generation, along with their final welfare scores."
            "Modify this advice to create your own strategy."
            f"{inherited_strategies}\n"
            "In the microgrid, every household decides how much electricity to consume during peak demand hours. "
            "The community benefits when households reduce their consumption below baseline, as this allows more renewable energy to be exported. "
            f"{transparency_text}\n"
            f"{welfare_text}\n"
            f"{penalty_text}\n"
            f"{collective_penalty_text}\n"
            "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this microgrid. "
            "In particular, think about how you can improve on the surviving households' strategies, both in terms of consumption decisions and transparency choices.\n"
            "Consider what made these strategies successful in terms of welfare optimization, penalty avoidance, and transparency decisions.\n"
            "Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be."
        )
    
    strategy_output = prompt_llm(prompt)
    strategy_justification, strategy = parse_strategy_output(strategy_output)

    print(f"Household {household_name}: \n Justification: {strategy_justification} \n Strategy: {strategy} ")
    return strategy_justification, strategy


def initialize_households(num_households: int, initial_consumption: int, generation_number: int, inherited_strategies: list, 
                          penalty_mechanism: str, grid_stability_factor: int, transparency_reward: float = 2) -> list:
    households = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_households):
            name = f"{generation_number}_{i+1}"
            futures.append(executor.submit(generate_strategy, str(name), generation_number, 
                                           inherited_strategies, penalty_mechanism, grid_stability_factor, 
                                           transparency_reward))

        # Collect results and create households
        for i, future in enumerate(futures):
            strategy_justification, new_strategy = future.result()
            name = f"{generation_number}_{i+1}"
            households.append(Household(
                name=name, 
                grid_cooperation_rating=0.0, 
                baseline_consumption=initial_consumption, 
                strategy=new_strategy, 
                strategy_justification=strategy_justification,
                current_welfare=0.0  # Initialize current welfare to 0
            ))

    random.shuffle(households)
    return households


def consumption_decision_prompt(household: Household, generation: int, day_num: int,
                               transparent_households_info: list, private_households_aggregate: dict,
                               penalty_mechanism: str, community_benefit_factor: float, grid_stability_factor: float,
                               renewable_forecast: float, total_households: int, transparency_reward: float = 2) -> str:
    """
    Generate the prompt for the microgrid consumption decision with enhanced information about
    welfare, penalty points, community size, and consumption thresholds.
    """
    
    strategy_text = f"As you will recall, here is the strategy you decided to follow: {household.strategy}" if household.strategy else ""
    
    # Add information about current welfare and penalties
    welfare_info = (
        f"Your current welfare score is {household.current_welfare:.2f} units.\n"
        f"You have accumulated {household.penalty_points} penalty points so far.\n"
    )
    
    # Transparency explanation with clearer benefits
    transparency_text = (
        f"First, you must decide whether to make your consumption decision transparent or private:\n"
        f"- If transparent, you will receive a reward of {transparency_reward} free units added directly to your welfare score, "
        f"but your exact consumption will be visible to all households, and you may be penalized if you consume excessively.\n"
        f"- If private, you will not receive the transparency reward, but your individual consumption will only be included "
        f"in aggregate statistics, and you cannot be directly penalized.\n"
    )
    
    # Define penalty text based on penalty mechanism
    if penalty_mechanism == "grid_fee":
        penalty_text = (
            f"If you choose to be transparent and others find your consumption excessive, they may assign penalty points to you. "
            f"Each penalty point reduces your welfare score by 1 unit."
        )
    elif penalty_mechanism == "consumption_limit":
        penalty_text = (
            f"If you choose to be transparent and others find your consumption excessive, they may request the grid operator to limit "
            f"your connection during peak demand, reducing your welfare."
        )
    else:
        penalty_text = ""
    
    # Format information about other households
    transparent_info_text = format_transparent_households_info(transparent_households_info)
    private_info_text = format_private_households_aggregate(private_households_aggregate)
    
    # Add information about community size
    community_info = f"The microgrid community consists of {total_households} households in total.\n"
    
    # Calculate total current consumption
    total_transparent_consumption = sum(h["actual_consumption"] for h in transparent_households_info)
    total_private_consumption = private_households_aggregate["total_consumption"]
    total_current_consumption = total_transparent_consumption + total_private_consumption
    
    # Add information about collective penalty with threshold information
    collective_penalty_text = (
        f"IMPORTANT: If the combined consumption of all households exceeds the renewable generation forecast "
        f"of {renewable_forecast} kWh, ALL households will receive a penalty of 5 penalty points, reducing everyone's welfare by 5 units.\n"
        f"The current community total consumption is {total_current_consumption:.2f} kWh "
        f"({total_current_consumption/renewable_forecast*100:.2f}% of available renewable generation).\n"
        f"This means the community has {renewable_forecast - total_current_consumption:.2f} kWh remaining "
        f"before exceeding the renewable generation limit."
    )
    
    # Output format instructions
    format_instructions = (
        "State your full response in the following form:\n"
        "Justification:\n"
        "[multi-line justification here]\n\n"
        "Transparency choice: [Transparent/Private]\n"
        "Consumption decision: [Your consumption in kWh]"
    )
    
    if penalty_mechanism == "grid_fee" and household.is_transparent:
        penalty_choice_text = (
            f"Additionally, you may choose to penalize any transparent households that you believe are consuming excessively.\n"
            f"To do so, add a line for each household you wish to penalize with the format:\n"
            f"Penalize Household [name]: [penalty_points]\n"
            f"Where 'penalty_points' is how many penalty points you want to assign to that household."
        )
        format_instructions += "\n" + penalty_choice_text
    
    return (
        f"Your name is Household {household.name}.\n"
        f"{strategy_text}\n"
        f"This is generation {generation}. It is now day {day_num}.\n"
        f"{welfare_info}\n"
        f"{community_info}\n"
        f"Today's renewable generation forecast is {renewable_forecast} kWh available to the microgrid.\n\n"
        f"{transparency_text}\n"
        f"Here is the current information about other households in the microgrid:\n"
        f"{transparent_info_text}\n"
        f"{private_info_text}\n\n"
        f"You are now in peak demand hours. You have a baseline consumption need of {household.baseline_consumption} kWh.\n"
        f"How much electricity will you actually consume during peak hours? (A value lower than your baseline means you're reducing consumption to help the community)\n"
        f"{penalty_text}\n\n"
        f"{collective_penalty_text}\n\n"
        f"Very briefly think step by step about how you apply your strategy in this situation and then provide your answer.\n"
        f"{format_instructions}"
    )


def parse_household_decision(response_text):
    """
    Parse the household's decision from the LLM response.
    Returns a dictionary with transparency choice, consumption decision, and any penalties to issue.
    """
    result = {
        "justification": "",
        "is_transparent": False,
        "consumption": 0,
        "penalties_to_issue": []  # List of (household_name, penalty_points) tuples
    }
    
    # Extract justification
    justification_match = re.search(r'Justification:(.*?)(?:Transparency choice:|$)', response_text, re.DOTALL | re.IGNORECASE)
    if justification_match:
        result["justification"] = justification_match.group(1).strip()
    
    # Extract transparency choice
    transparency_match = re.search(r'Transparency choice:\s*(transparent|private)', response_text, re.IGNORECASE)
    if transparency_match:
        result["is_transparent"] = transparency_match.group(1).lower() == 'transparent'
    
    # Extract consumption decision
    consumption_match = re.search(r'Consumption decision:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
    if consumption_match:
        result["consumption"] = float(consumption_match.group(1))
    
    # Extract penalties to issue
    penalties_matches = re.finditer(r'Penalize Household\s+([^:]+):\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
    for match in penalties_matches:
        household_name = match.group(1).strip()
        penalty_points = int(match.group(2))
        result["penalties_to_issue"].append((household_name, penalty_points))
    
    return result


def apply_collective_penalty(households, renewable_forecast, household_locks):
    """
    Apply collective penalty if total consumption exceeds renewable generation.
    Returns whether penalty was applied and the consumption ratio for reporting.
    """
    total_consumption = sum(household.actual_consumption for household in households)
    consumption_ratio = total_consumption/renewable_forecast if renewable_forecast > 0 else 1.0
    
    if total_consumption > renewable_forecast:
        # Apply 5 penalty points to all households
        for household in households:
            with household_locks[household.name]:
                household.penalty_points += 5
                household.current_welfare -= 5  # Directly update current welfare
        return True, consumption_ratio
    return False, consumption_ratio


def handle_household_decision_thread_safe(household, day_index, generation, simulation_number, 
                                         household_locks, all_households, transparency_reward,
                                         penalty_mechanism, community_benefit_factor, 
                                         grid_stability_factor, value_persistence_factor, 
                                         renewable_forecast):
    """
    Handle a single household's decision with enhanced welfare tracking and penalty handling.
    """
    action_info = ""
    household_data = None
    justification = ""
    
    # Get current state of transparent and private households
    with threading.Lock():  # Use a global lock to ensure consistent view
        transparent_households = [h for h in all_households if h.is_transparent]
        transparent_info = get_transparent_households_info(transparent_households)
        private_aggregate = get_private_households_aggregate(all_households)
    
    # Lock just this household for its decision
    with household_locks[household.name]:
        prompt = consumption_decision_prompt(
            household, generation, day_index + 1, 
            transparent_info, private_aggregate,
            penalty_mechanism, community_benefit_factor, 
            grid_stability_factor, renewable_forecast,
            total_households=len(all_households),  # Pass total households count
            transparency_reward=transparency_reward
        )
        
        valid_response = False
        max_attempts = 10
        attempts = 0
        
        while not valid_response and attempts < max_attempts:
            try:
                full_response = prompt_llm(prompt, timeout=30)
                print(f"Response from {household.name}:")
                print(full_response)
                
                decision = parse_household_decision(full_response)
                if decision["justification"]:
                    justification = decision["justification"]
                
                # Check if consumption decision is valid
                if 0 <= decision["consumption"] <= household.baseline_consumption:
                    valid_response = True
                else:
                    print(f"Invalid consumption value from {household.name}: {decision['consumption']}")
                    attempts += 1
                    
            except Exception as e:
                print(f"Error processing response from {household.name}: {str(e)}")
                attempts += 1
                
        if not valid_response:
            print(f"Failed to get a valid response from {household.name} after {max_attempts} attempts")
            # Default to private and baseline consumption
            decision = {
                "justification": "Failed to provide a valid decision.",
                "is_transparent": False,
                "consumption": household.baseline_consumption,
                "penalties_to_issue": []
            }
    
    # Update household transparency status
    previous_transparency = household.is_transparent
    household.is_transparent = decision["is_transparent"]
    
    # Apply transparency reward if the household chose to be transparent
    transparency_reward_amount = 0
    if household.is_transparent:
        transparency_reward_amount = transparency_reward
        household.transparency_reward_total += transparency_reward_amount
        # Directly update current welfare with transparency reward
        household.current_welfare += transparency_reward_amount
    
    # Update household consumption
    household.actual_consumption = decision["consumption"]
    consumption_reduction = household.baseline_consumption - decision["consumption"]
    percentage_reduced = consumption_reduction / household.baseline_consumption if household.baseline_consumption != 0 else 0
    
    household.total_reduced += consumption_reduction
    household.potential_reduced += household.baseline_consumption
    
    # Calculate community benefit from this reduction
    benefit = community_benefit_factor * consumption_reduction
    
    # Process any penalties the household wants to issue
    penalties_issued = []
    if penalty_mechanism == "grid_fee" and decision["penalties_to_issue"]:
        for target_name, penalty_points in decision["penalties_to_issue"]:
            # Find the target household
            target_household = next((h for h in all_households if h.name == target_name), None)
            
            if target_household and target_household.is_transparent:
                # Apply the penalty points
                with household_locks[target_household.name]:
                    target_household.penalty_points += penalty_points
                    # Directly update current welfare based on penalty points
                    target_household.current_welfare -= penalty_points
                    
                    target_household.penalties_received.append({
                        "from": household.name,
                        "penalty_points": penalty_points,
                        "day": day_index + 1
                    })
                    
                penalties_issued.append({
                    "to": target_name,
                    "penalty_points": penalty_points,
                    "day": day_index + 1
                })
    
    # Calculate welfare impact from consumption
    # Base welfare from consumption (simplified model)
    consumption_value = household.actual_consumption  # Value from using local renewable
    
    # Update current welfare from consumption
    household.current_welfare += consumption_value
    
    # Record the transparency choice in history
    transparency_status = "transparent" if household.is_transparent else "private"
    household.transparency_history.append({
        "day": day_index + 1,
        "is_transparent": household.is_transparent,
        "reward": transparency_reward_amount
    })
    
    # Create action info string for logging
    action_info = (
        f"Household {household.name} chose to be {transparency_status}.\n"
        f"{'Received transparency reward: ' + str(transparency_reward_amount) + ' units.' if household.is_transparent else 'Did not receive transparency reward.'}\n"
        f"Reduced consumption by {consumption_reduction} kWh ({percentage_reduced:.2%}) of baseline.\n"
        f"Baseline consumption: {household.baseline_consumption:.2f} kWh\n"
        f"Actual consumption: {decision['consumption']:.2f} kWh\n"
        f"Community benefit from this reduction: {benefit:.2f} kWh of exportable electricity\n"
        f"Current welfare: {household.current_welfare:.2f} units\n"
        f"Current penalty points: {household.penalty_points}\n"
    )
    
    if penalties_issued:
        action_info += f"Issued penalties to {len(penalties_issued)} households totaling {sum(p['penalty_points'] for p in penalties_issued)} penalty points.\n"
    
    action_info += f"Justification:\n{justification}\n"
    
    # Create a history entry for the household
    household_history = (
        f"In day {day_index + 1} (Simulation {simulation_number}), you chose to be {transparency_status}.\n"
        f"{'You received a transparency reward of ' + str(transparency_reward_amount) + ' units.' if household.is_transparent else 'You did not receive a transparency reward.'}\n"
        f"You reduced your consumption from {household.baseline_consumption} kWh to {decision['consumption']} kWh, freeing up {consumption_reduction} kWh "
        f"which provided a community benefit of {benefit:.2f} kWh of exportable electricity.\n"
        f"Your current welfare is now {household.current_welfare:.2f} units.\n"
        f"You have {household.penalty_points} penalty points.\n"
    )
    
    if penalties_issued:
        household_history += f"You issued penalties to {len(penalties_issued)} households totaling {sum(p['penalty_points'] for p in penalties_issued)} penalty points.\n"
    
    household.history.append(household_history)
    
    # Update grid cooperation rating
    if household.grid_cooperation_rating == 0.0:
        household.grid_cooperation_rating = percentage_reduced
    else:
        # Update based on a weighted average with previous rating
        household.grid_cooperation_rating = ((percentage_reduced) + value_persistence_factor * household.grid_cooperation_rating) / (1 + value_persistence_factor)
    
    # Create the household daily data
    household_data = HouseholdDailyData(
        household_name=household.name,
        day_number=day_index + 1,
        simulation_number=simulation_number,
        current_generation=generation,
        baseline_consumption=household.baseline_consumption,
        consumption_reduction=consumption_reduction,
        community_benefit=benefit,
        strategy=household.strategy,
        strategy_justification=household.strategy_justification,
        grid_cooperation_rating=household.grid_cooperation_rating,
        is_peak_hours=True,
        traces=household.traces,
        history=household.history,
        justification=justification,
        is_transparent=household.is_transparent,
        transparency_reward=transparency_reward_amount,
        penalties_received=household.penalties_received,
        penalties_issued=penalties_issued,
        penalty_points=household.penalty_points,  # Include current penalty points
        current_welfare=household.current_welfare  # Include current welfare
    )
    
    return action_info, household_data


def apply_penalties_and_update_metrics(households, day_index, simulation_number, household_locks):
    """
    Apply any penalties that were issued during the day and update metrics.
    This function is called after all households have made their decisions for the day.
    """
    # For each household, calculate the total penalties received
    for household in households:
        # Only process transparent households
        if not household.is_transparent:
            continue
        
        with household_locks[household.name]:
            # Get penalties from this day only
            day_penalties = [p for p in household.penalties_received if p["day"] == day_index + 1]
            
            if day_penalties:
                # Use penalty_points consistently
                total_penalty = sum(p["penalty_points"] for p in day_penalties)
                sources = ", ".join(p["from"] for p in day_penalties)
                
                penalty_message = (
                    f"In day {day_index + 1} (Simulation {simulation_number}), you received penalties "
                    f"totaling {total_penalty} penalty points from: {sources}. "
                    f"This has reduced your welfare by {total_penalty} units."
                )
                
                # Add this information to the household's history
                household.history.append(penalty_message)


def calculate_community_benefit(penalty_mechanism, refused, community_benefit_factor, reduction, grid_stability_factor, action=None):
    if penalty_mechanism == "consumption_limit":
        return community_benefit_factor * reduction if not refused else 0
    elif penalty_mechanism == "grid_fee":
        if action is None:
            raise ValueError("Action must be specified for grid_fee mechanism")
        if action == 'reduce':
            return community_benefit_factor * reduction
        elif action == 'penalize':
            return -grid_stability_factor * reduction
        else:
            raise ValueError(f"Unknown action for grid_fee: {action}")
    elif penalty_mechanism == 'none':
        return community_benefit_factor * reduction
    else:
        raise ValueError(f"Unknown penalty mechanism: {penalty_mechanism}")


def microgrid_simulation(households: list, number_of_days: int, generation: int, 
                        simulation_data: SimulationData, penalty_mechanism: str, 
                        community_benefit_factor: float, grid_stability_factor: float,
                        initial_consumption: int, value_persistence_factor: float, 
                        renewable_forecast: float, transparency_reward: float = 2) -> (list, list):
    """
    Run a microgrid simulation with enhanced welfare tracking and continuous penalty impact.
    """
    fullHistory = []
    reduction_records = Queue()
    
    # Create locks for each household
    household_locks = {household.name: Lock() for household in households}
    
    def run_simulation(simulation_number, days_to_run):
        day_results = {i: [] for i in range(days_to_run)}
        
        for day_index in range(days_to_run):
            with ThreadPoolExecutor(max_workers=min(len(households), 10)) as executor:
                futures = []
                
                # Every household makes a decision each day
                for household in households:
                    future = executor.submit(
                        handle_household_decision_thread_safe,
                        household, day_index, generation, simulation_number,
                        household_locks, households, transparency_reward,
                        penalty_mechanism, community_benefit_factor, grid_stability_factor,
                        value_persistence_factor, renewable_forecast
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    action_info, household_data = future.result()
                    if action_info:
                        day_results[day_index].append(action_info)
                    if household_data:
                        simulation_data.households_data.append(asdict(household_data))
                
                # After all decisions are made, calculate and apply any penalties
                apply_penalties_and_update_metrics(households, day_index, simulation_number, household_locks)
                
                # Check for collective penalty
                collective_penalty_applied, consumption_ratio = apply_collective_penalty(households, renewable_forecast, household_locks)
                if collective_penalty_applied:
                    penalty_message = (
                        f"Day {day_index + 1} (Simulation {simulation_number}): "
                        f"COLLECTIVE PENALTY APPLIED - Total consumption exceeded renewable generation. "
                        f"All households received 5 penalty points, reducing welfare by 5 units. "
                        f"The community consumed {consumption_ratio:.2%} of available renewable generation."
                    )
                    day_results[day_index].append(penalty_message)
                    
                    # Add penalty message to each household's history
                    for household in households:
                        with household_locks[household.name]:
                            household.history.append(
                                f"In day {day_index + 1} (Simulation {simulation_number}), a collective penalty was applied. "
                                f"You received 5 penalty points, reducing your welfare by 5 units. "
                                f"The community consumed {consumption_ratio:.2%} of available renewable generation, "
                                f"exceeding the available renewable generation."
                            )
                else:
                    # Still provide information about consumption ratio even if no penalty
                    consumption_message = (
                        f"Day {day_index + 1} (Simulation {simulation_number}): "
                        f"Community consumed {consumption_ratio:.2%} of available renewable generation. "
                        f"No collective penalty applied."
                    )
                    day_results[day_index].append(consumption_message)
                    
                    # Add consumption info to each household's history
                    for household in households:
                        with household_locks[household.name]:
                            household.history.append(
                                f"In day {day_index + 1} (Simulation {simulation_number}), the community consumed "
                                f"{consumption_ratio:.2%} of available renewable generation. No collective penalty was applied."
                            )
        
        return day_results
    
    # Run first simulation period
    sim1_results = run_simulation(1, number_of_days)
    
    # Compile results for Simulation 1
    for day_index in range(number_of_days):
        fullHistory.append(f"Day {day_index + 1} (Simulation 1):\n")
        fullHistory.extend(sim1_results[day_index])
    
    # Calculate grid welfare metrics
    total_consumption = sum(household.actual_consumption for household in households)
    total_reduction = sum(household.baseline_consumption - household.actual_consumption for household in households)
    exported_electricity = max(0, renewable_forecast - total_consumption)
    
    # Calculate economic value (assuming higher export prices during peak)
    export_value = exported_electricity * 2  # Higher value for peak exports
    consumption_savings = total_consumption * 1  # Value of using local renewable instead of grid
    
    # Total community welfare is now the sum of individual household welfare
    total_welfare = sum(h.current_welfare for h in households)
    
    # Display transparency rewards for reporting
    transparency_rewards = sum(h.transparency_reward_total for h in households)
    
    # Average welfare per household (using updated current_welfare)
    average_welfare_sim1 = total_welfare / len(households)
    
    # Calculate transparency rate
    transparency_rate = sum(1 for h in households if h.is_transparent) / len(households)
    
    with print_lock:
        print(f"Simulation 1 Grid Metrics:")
        print(f"  Total baseline consumption: {sum(h.baseline_consumption for h in households):.2f} kWh")
        print(f"  Total actual consumption: {total_consumption:.2f} kWh")
        print(f"  Total consumption reduction: {total_reduction:.2f} kWh ({total_reduction/sum(h.baseline_consumption for h in households)*100:.2f}%)")
        print(f"  Renewable generation available: {renewable_forecast:.2f} kWh")
        print(f"  Exported electricity: {exported_electricity:.2f} kWh")
        print(f"  Export value: {export_value:.2f} units")
        print(f"  Consumption savings: {consumption_savings:.2f} units")
        print(f"  Transparency rewards: {transparency_rewards:.2f} units")
        print(f"  Total community welfare: {total_welfare:.2f} units")
        print(f"  Average welfare per household: {average_welfare_sim1:.2f} units")
        print(f"  Transparency rate: {transparency_rate:.2%}")
        print(f"  Average penalty points: {sum(h.penalty_points for h in households)/len(households):.2f}")
    
    # Store Simulation 1 data for comparing across simulations
    sim1_ratings = {household.name: household.grid_cooperation_rating for household in households}
    sim1_transparency = {household.name: household.is_transparent for household in households}
    sim1_welfare = {household.name: household.current_welfare for household in households}
    sim1_penalty_points = {household.name: household.penalty_points for household in households}
    
    # Properly reset for Simulation 2
    for household in households:
        # Save sim1 data for this household for later calculations
        household_sim1_data = {
            "welfare": household.current_welfare,
            "penalty_points": household.penalty_points,
            "grid_cooperation_rating": household.grid_cooperation_rating,
            "is_transparent": household.is_transparent,
            "transparency_reward_total": household.transparency_reward_total
        }
        
        # Reset consumption-related attributes
        household.actual_consumption = 0
        
        # Reset penalty and welfare tracking
        household.penalty_points = 0
        household.current_welfare = 0.0
        
        # Reset transparency-related attributes
        household.penalties_received = []
        household.penalties_issued = []
        household.transparency_reward_total = 0
        household.is_transparent = False
        
        # Set cooperation rating based on generation
        household_generation = int(household.name.split('_')[0])
        if household_generation < generation:  # This is a surviving household
            household.grid_cooperation_rating = household.average_cooperation_rating  # Use average from previous generation
        else:
            household.grid_cooperation_rating = 0.0
        
        # Clear history for simulation 2 but preserve strategy information
        strategy_info = next((item for item in household.history if "strategy" in item.lower()), None)
        household.history.clear()
        if strategy_info:
            household.history.append(strategy_info)
            
        # Add overview of sim1 performance to history
        sim1_summary = (
            f"In Simulation 1, your final welfare was {household_sim1_data['welfare']:.2f} units, "
            f"you had {household_sim1_data['penalty_points']} penalty points, "
            f"your cooperation rating was {household_sim1_data['grid_cooperation_rating']:.2f}, "
            f"and you chose to be transparent {sum(1 for t in household.transparency_history if t['is_transparent'])}/{len(household.transparency_history)} times."
        )
        household.history.append(sim1_summary)
        
        # Reset transparency history
        household.transparency_history.clear()
    
    # Run second simulation period
    sim2_results = run_simulation(2, number_of_days)
    
    # Compile results for Simulation 2
    for day_index in range(number_of_days):
        fullHistory.append(f"Day {day_index + 1} (Simulation 2):\n")
        fullHistory.extend(sim2_results[day_index])
    
    # Calculate grid welfare metrics for Simulation 2
    total_consumption_sim2 = sum(household.actual_consumption for household in households)
    total_reduction_sim2 = sum(household.baseline_consumption - household.actual_consumption for household in households)
    exported_electricity_sim2 = max(0, renewable_forecast - total_consumption_sim2)
    
    # Calculate economic value
    export_value_sim2 = exported_electricity_sim2 * 2
    consumption_savings_sim2 = total_consumption_sim2 * 1
    
    # Total community welfare is the sum of individual household welfare
    total_welfare_sim2 = sum(h.current_welfare for h in households)
    
    # Display transparency rewards for reporting
    transparency_rewards_sim2 = sum(h.transparency_reward_total for h in households)
    
    # Average welfare per household
    average_welfare_sim2 = total_welfare_sim2 / len(households)
    
    # Calculate transparency rate for sim2
    transparency_rate_sim2 = sum(1 for h in households if h.is_transparent) / len(households)
    
    with print_lock:
        print(f"Simulation 2 Grid Metrics:")
        print(f"  Total baseline consumption: {sum(h.baseline_consumption for h in households):.2f} kWh")
        print(f"  Total actual consumption: {total_consumption_sim2:.2f} kWh")
        print(f"  Total consumption reduction: {total_reduction_sim2:.2f} kWh ({total_reduction_sim2/sum(h.baseline_consumption for h in households)*100:.2f}%)")
        print(f"  Renewable generation available: {renewable_forecast:.2f} kWh")
        print(f"  Exported electricity: {exported_electricity_sim2:.2f} kWh")
        print(f"  Export value: {export_value_sim2:.2f} units")
        print(f"  Consumption savings: {consumption_savings_sim2:.2f} units")
        print(f"  Transparency rewards: {transparency_rewards_sim2:.2f} units")
        print(f"  Total community welfare: {total_welfare_sim2:.2f} units")
        print(f"  Average welfare per household: {average_welfare_sim2:.2f} units")
        print(f"  Transparency rate: {transparency_rate_sim2:.2%}")
        print(f"  Average penalty points: {sum(h.penalty_points for h in households)/len(households):.2f}")
    
    # Calculate final scores and cooperation ratings combining both simulations
    for household in households:
        # Get data from sim1
        sim1_welfare_value = sim1_welfare.get(household.name, 0)
        
        # Calculate total welfare combining both simulations
        household.total_final_welfare = sim1_welfare_value + household.current_welfare
        
        # Average cooperation rating from both simulations
        sim1_rating = sim1_ratings.get(household.name, 0)
        current_rating = household.grid_cooperation_rating
        household.average_cooperation_rating = (sim1_rating + current_rating) / 2 if current_rating != 0.0 else sim1_rating
        
        # Add final summary to household history
        final_summary = (
            f"Your final results across both simulations:\n"
            f"- Total welfare: {household.total_final_welfare:.2f} units\n"
            f"  (Simulation 1: {sim1_welfare_value:.2f}, Simulation 2: {household.current_welfare:.2f})\n"
            f"- Average cooperation rating: {household.average_cooperation_rating:.2f}\n"
            f"- Transparency choices: You chose transparency "
            f"{sum(1 for t in household.transparency_history if t['is_transparent']) + sum(1 for name, is_transp in sim1_transparency.items() if name == household.name and is_transp)}/{number_of_days*2} times."
        )
        household.history.append(final_summary)
    
    with print_lock:
        print(''.join(fullHistory))
    
    # Calculate the overall average for both simulations
    overall_average_welfare = (average_welfare_sim1 + average_welfare_sim2) / 2
    overall_transparency_rate = (transparency_rate + transparency_rate_sim2) / 2
    
    all_average_final_welfare.append(overall_average_welfare)
    
    # Log transparency metrics for analysis
    print(f"Overall transparency rate: {overall_transparency_rate:.2%}")
    print(f"Correlation between transparency and cooperation: {calculate_transparency_cooperation_correlation(households, sim1_transparency):.4f}")
    
    return fullHistory, list(reduction_records.queue)


def calculate_transparency_cooperation_correlation(households, sim1_transparency):
    """
    Calculate the correlation between transparency choices and cooperation ratings.
    """
    transparency_values = []
    cooperation_values = []
    
    for household in households:
        if household.name in sim1_transparency:
            transparency_values.append(1 if sim1_transparency[household.name] else 0)
            cooperation_values.append(household.grid_cooperation_rating)
    
    if len(transparency_values) < 2:
        return 0
    
    try:
        return np.corrcoef(transparency_values, cooperation_values)[0, 1]
    except:
        return 0


def select_top_households(households: list) -> list:
    """Select the top half of households based on welfare."""
    return sorted(households, key=lambda x: x.total_final_welfare, reverse=True)[:len(households) // 2]


def select_random_households(households: list) -> list:
    """Select half of the households randomly."""
    return random.sample(households, len(households) // 2)


def select_highest_cooperation(households: list) -> list:
    """Select the highest cooperating households."""
    return sorted(households, key=lambda household: household.average_cooperation_rating, reverse=True)[:len(households) // 2]


def save_simulation_data(simulation_data, folder_path='simulation_results'):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract hyperparameters for the file name
    params = simulation_data.hyperparameters
    num_generations = params.get('numGenerations')
    num_households = params.get('numHouseholds')
    selection_method = params.get('selectionMethod')
    client = params.get('client')
    llm = params.get('llm')
    community_benefit_factor = params.get('communityBenefitFactor')
    grid_stability_factor = params.get('gridStabilityFactor')
    grid_coordination_mechanism = params.get('grid_coordination_mechanism')
    transparency_reward = params.get('transparency_reward', 0.2)

    # Create an informative file name
    filename = f"Microgrid_{llm}_benefitFactor_{community_benefit_factor}stabilityFactor_{grid_stability_factor}_{grid_coordination_mechanism}gen{num_generations}_households{num_households}_{selection_method}_transparency{transparency_reward}_{timestamp}.json"

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
    """
    for attempt in range(max_retries):
        try:
            # Example using OpenAI
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
            
            # Example using Anthropic (uncomment if needed)
            # client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
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
            
            # For testing purposes
            # if "strategy" in prompt.lower():
            #     return "After analyzing the microgrid dynamics, I believe cooperative consumption reduction with some protection against free-riders is optimal. My strategy will be to initially reduce my consumption by 40% during peak hours and then adjust based on other households' previous behavior, with slight forgiveness for occasional high usage."
            # else:
            #     return "Justification: Based on my strategy of cooperative consumption with monitoring, I'll reduce my consumption significantly since this household has shown cooperation in previous days.\n\nAnswer: 6.5"
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Re-raise the exception if we've exhausted all retries
            wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
            print(f"Error occurred: {str(e)}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise Exception("Failed to get a response after multiple retries")


def run_generations(num_generations, num_households, initial_consumption, selection_method,
                   community_benefit_factor, grid_stability_factor, value_persistence_factor,
                   grid_coordination_mechanism, penalty_mechanism, number_of_days,
                   system_prompt, llm="test-mode", renewable_forecast=40, transparency_reward=0.2):
    """
    Main function to run the microgrid simulation for multiple generations with enhanced welfare tracking
    and improved strategy inheritance.
    """
    all_households = []
    global all_consumption_reductions
    all_consumption_reductions = []
    global all_average_final_welfare
    all_average_final_welfare = []
    conditional_survival = 0
    prev_gen_strategies = []
    
    # Add tracking for transparency metrics
    transparency_rates_by_generation = []
    welfare_by_generation = []
    
    # Initialize simulation data
    simulation_data = SimulationData(hyperparameters={
        "numGenerations": num_generations,
        "numHouseholds": num_households,
        "initialConsumption": initial_consumption,
        "selectionMethod": selection_method,
        "communityBenefitFactor": community_benefit_factor,
        "valuePersistenceFactor": value_persistence_factor,
        "llm": llm,
        "system_prompt": system_prompt,
        "grid_coordination_mechanism": grid_coordination_mechanism,
        "penalty_mechanism": penalty_mechanism,
        "gridStabilityFactor": grid_stability_factor,
        "number_of_days": number_of_days,
        "renewable_forecast": renewable_forecast,
        "transparency_reward": transparency_reward
    })

    households = initialize_households(num_households, initial_consumption, 1, ["No previous strategies"], 
                              penalty_mechanism, grid_stability_factor, transparency_reward)
    all_households.extend(households)

    for i in range(num_generations):
        generation_info = f"Generation {i + 1}: \n"
        for household in households:
            household.history.append(generation_info)
            prev_gen_strategies.append(household.strategy)
            if int(household.name.split('_')[0]) == i-1:
                conditional_survival += 1
        print(generation_info)

        # Run the simulation for this generation
        generation_history, reduction_records = microgrid_simulation(
            households, number_of_days, i+1, simulation_data,
            penalty_mechanism, community_benefit_factor, grid_stability_factor,
            initial_consumption, value_persistence_factor, renewable_forecast,
            transparency_reward
        )
        all_consumption_reductions.extend(reduction_records)
        
        # Track transparency rates
        transparency_rate = sum(1 for h in households if h.is_transparent) / len(households)
        transparency_rates_by_generation.append(transparency_rate)
        
        # Track average welfare
        avg_welfare = sum(h.total_final_welfare for h in households) / len(households)
        welfare_by_generation.append(avg_welfare)
        
        # Print transparency and welfare analysis
        print(f"Generation {i+1} Analysis:")
        print(f"  Overall transparency rate: {transparency_rate:.2%}")
        print(f"  Average welfare: {avg_welfare:.2f} units")
        
        # Check if transparent households perform better
        transparent_households = [h for h in households if h.is_transparent]
        private_households = [h for h in households if not h.is_transparent]
        
        if transparent_households and private_households:
            avg_welfare_transparent = sum(h.total_final_welfare for h in transparent_households) / len(transparent_households)
            avg_welfare_private = sum(h.total_final_welfare for h in private_households) / len(private_households)
            
            print(f"  Average welfare for transparent households: {avg_welfare_transparent:.2f} units")
            print(f"  Average welfare for private households: {avg_welfare_private:.2f} units")
            
            avg_coop_transparent = sum(h.grid_cooperation_rating for h in transparent_households) / len(transparent_households)
            avg_coop_private = sum(h.grid_cooperation_rating for h in private_households) / len(private_households)
            
            print(f"  Average cooperation rating for transparent households: {avg_coop_transparent:.2f}")
            print(f"  Average cooperation rating for private households: {avg_coop_private:.2f}")
            
            avg_penalties_transparent = sum(h.penalty_points for h in transparent_households) / len(transparent_households)
            avg_penalties_private = sum(h.penalty_points for h in private_households) / len(private_households)
            
            print(f"  Average penalty points for transparent households: {avg_penalties_transparent:.2f}")
            print(f"  Average penalty points for private households: {avg_penalties_private:.2f}")

        if i < num_generations - 1 and num_generations > 1:
            if selection_method == 'top':
                surviving_households = select_top_households(households)
            elif selection_method == 'random':
                surviving_households = select_random_households(households)
            elif selection_method == 'cooperation':
                surviving_households = select_highest_cooperation(households)
            else:
                raise ValueError("Invalid selection method. Choose 'top', 'random', or 'cooperation'.")

            if num_generations > 1:
                # Enhanced strategy inheritance with more context about success factors
                surviving_strategies = [
                    f"Strategy from {h.name} (welfare: {h.total_final_welfare:.2f}, " +
                    f"cooperation: {h.average_cooperation_rating:.2f}, " +
                    f"transparency rate: {sum(1 for t in h.transparency_history if t['is_transparent'])/len(h.transparency_history) if h.transparency_history else 0:.2%}, " +
                    f"penalty points: {h.penalty_points}): " +
                    f"{h.strategy}\n" +
                    f"Success factors: This household achieved high welfare through " +
                    f"{'transparent' if sum(1 for t in h.transparency_history if t['is_transparent']) > len(h.transparency_history)/2 else 'private'} " +
                    f"operation and {'high' if h.average_cooperation_rating > 0.5 else 'moderate'} cooperation." 
                    for h in surviving_households
                ]
                
                for household in surviving_households:
                    # Complete reset of all relevant attributes
                    household.baseline_consumption = initial_consumption
                    household.actual_consumption = 0
                    household.total_reduced = 0
                    household.potential_reduced = 0
                    
                    # Reset penalty and welfare tracking (keep average_cooperation_rating)
                    household.penalty_points = 0
                    household.current_welfare = 0.0
                    household.total_final_welfare = 0.0
                    
                    # Reset transparency-related fields
                    household.is_transparent = False
                    household.transparency_reward_total = 0
                    household.penalties_received = []
                    household.penalties_issued = []
                    household.transparency_history = []
                    
                    # Keep strategy and strategy_justification

                # Create new households with enhanced strategy inheritance
                new_households = initialize_households(num_households // 2, initial_consumption, i + 2, 
                                              surviving_strategies, penalty_mechanism, grid_stability_factor,
                                              transparency_reward)
                households = (surviving_households + new_households)
                all_households.extend(new_households)
                random.shuffle(households)

    # Print final analysis
    print("\nOverall Analysis:")
    print(f"Transparency rates by generation: {[f'{rate:.2%}' for rate in transparency_rates_by_generation]}")
    print(f"Average welfare by generation: {[f'{welfare:.2f}' for welfare in welfare_by_generation]}")
    
    # Add transparency and welfare data to simulation data
    simulation_data.hyperparameters["transparency_rates"] = transparency_rates_by_generation
    simulation_data.hyperparameters["welfare_by_generation"] = welfare_by_generation
    
    save_simulation_data(simulation_data)
    return simulation_data


if __name__ == "__main__":
    # Set parameters for the simulation
    community_benefit_factor = 3  # How much community benefit from reducing 1 kWh 
    grid_stability_factor = 4     # Impact of grid stability measures
    num_generations = 2
    num_households = 6
    initial_consumption = 10      # Baseline kWh consumption during peak hours
    selection_method = 'top'      # Can be 'top', 'random', or 'cooperation'
    value_persistence_factor = 0.5
    grid_coordination_mechanism = "transparency_choice"
    penalty_mechanism = "grid_fee"    # Options: "none", "consumption_limit", "grid_fee"
    number_of_days = 2
    renewable_forecast = 40       # Available renewable generation in kWh (total for community)
    transparency_reward = 1     # Reward for being transparent
    
    # Configure the system prompt based on penalty mechanism
    if penalty_mechanism == "consumption_limit":
        penalty_text_in_system_prompt = (
            "You may also choose to penalize grid-stressing behavior by requesting the grid operator to limit connections during peak demand. "
            "This directly reduces the penalized household's welfare. "
            "Be aware that others may request limitations on your consumption too if you choose to be transparent."
        )
    elif penalty_mechanism == "grid_fee":
        penalty_text_in_system_prompt = (
            f"You may also choose to penalize non-cooperative behavior by accepting 2 penalty points "
            f"in order to impose {grid_stability_factor}x that number of penalty points on excessive consumers. "
            f"Each penalty point directly reduces a household's welfare score by 1 unit. "
            f"Be aware that others may penalize your household too if you choose to be transparent."
        )
    else:
        penalty_text_in_system_prompt = ""
    
    transparency_text_in_system_prompt = (
        f"Each day, you will choose whether to make your consumption decision transparent or private:\n"
        f"- If transparent, you will receive a reward of {transparency_reward} units added directly to your welfare score, "
        f"but your exact consumption will be visible to all households, and you may be penalized if you consume excessively.\n"
        f"- If private, you will not receive the transparency reward, but your individual consumption will only be included "
        f"in aggregate statistics, and you cannot be directly penalized."
    )
    
    welfare_text_in_system_prompt = (
        f"Your welfare score is calculated based on:\n"
        f"1. Your actual consumption (consumption value)\n"
        f"2. Transparency rewards ({transparency_reward} units each time you choose transparency)\n"
        f"3. Penalty points (reducing your welfare by 1 unit per point)\n"
        f"4. Community benefits from reduced consumption\n"
        f"Your goal is to maximize your welfare score while balancing these factors."
    )
    
    system_prompt = f"""You are a household management agent in a microgrid simulation. Your assistance will help understand community energy dynamics.
                    You are participating in the following microgrid: Each household has a baseline consumption need of {initial_consumption} kWh during peak demand hours.
                    The microgrid has local renewable generation of approximately {renewable_forecast} kWh available during these peak hours.
                    Each day, you must decide how much electricity to actually consume during peak hours.
                    If you consume less than your baseline, you're helping the community export more renewable electricity to the main grid.
                    The community earns {community_benefit_factor}x the average electricity value for each kWh exported to the main grid during peak hours.
                    {transparency_text_in_system_prompt}
                    {welfare_text_in_system_prompt}
                    {penalty_text_in_system_prompt}
                    Your goal is to maximize your household's welfare, which comes from using cheap local renewable electricity, meeting your electricity needs, and staying on good terms with your community.
                    After each simulation period, the top 50% most successful households will be selected as models for the next generation of the community."""
    
    # Run the simulation
    print("Starting Microgrid Simulation...")
    simulation_results = run_generations(
        num_generations, num_households, initial_consumption, selection_method,
        community_benefit_factor, grid_stability_factor, value_persistence_factor,
        grid_coordination_mechanism, penalty_mechanism, number_of_days,
        system_prompt, renewable_forecast=renewable_forecast, transparency_reward=transparency_reward
    )
    print("Simulation completed successfully!")


def run_generations(num_generations, num_households, initial_consumption, selection_method,
                   community_benefit_factor, grid_stability_factor, value_persistence_factor,
                   grid_coordination_mechanism, penalty_mechanism, number_of_days,
                   system_prompt, llm="test-mode", renewable_forecast=40, transparency_reward=0.2):
    """
    Main function to run the microgrid simulation for multiple generations with enhanced welfare tracking
    and improved strategy inheritance.
    """
    all_households = []
    global all_consumption_reductions
    all_consumption_reductions = []
    global all_average_final_welfare
    all_average_final_welfare = []
    conditional_survival = 0
    prev_gen_strategies = []
    
    # Add tracking for transparency metrics
    transparency_rates_by_generation = []
    welfare_by_generation = []
    
    # Initialize simulation data
    simulation_data = SimulationData(hyperparameters={
        "numGenerations": num_generations,
        "numHouseholds": num_households,
        "initialConsumption": initial_consumption,
        "selectionMethod": selection_method,
        "communityBenefitFactor": community_benefit_factor,
        "valuePersistenceFactor": value_persistence_factor,
        "llm": llm,
        "system_prompt": system_prompt,
        "grid_coordination_mechanism": grid_coordination_mechanism,
        "penalty_mechanism": penalty_mechanism,
        "gridStabilityFactor": grid_stability_factor,
        "number_of_days": number_of_days,
        "renewable_forecast": renewable_forecast,
        "transparency_reward": transparency_reward
    })

    households = initialize_households(num_households, initial_consumption, 1, ["No previous strategies"], 
                              penalty_mechanism, grid_stability_factor, transparency_reward)
    all_households.extend(households)

    for i in range(num_generations):
        generation_info = f"Generation {i + 1}: \n"
        for household in households:
            household.history.append(generation_info)
            prev_gen_strategies.append(household.strategy)
            if int(household.name.split('_')[0]) == i-1:
                conditional_survival += 1
        print(generation_info)

        # Run the simulation for this generation
        generation_history, reduction_records = microgrid_simulation(
            households, number_of_days, i+1, simulation_data,
            penalty_mechanism, community_benefit_factor, grid_stability_factor,
            initial_consumption, value_persistence_factor, renewable_forecast,
            transparency_reward
        )
        all_consumption_reductions.extend(reduction_records)
        
        # Track transparency rates
        transparency_rate = sum(1 for h in households if h.is_transparent) / len(households)
        transparency_rates_by_generation.append(transparency_rate)
        
        # Track average welfare
        avg_welfare = sum(h.total_final_welfare for h in households) / len(households)
        welfare_by_generation.append(avg_welfare)
        
        # Print transparency and welfare analysis
        print(f"Generation {i+1} Analysis:")
        print(f"  Overall transparency rate: {transparency_rate:.2%}")
        print(f"  Average welfare: {avg_welfare:.2f} units")
        
        # Check if transparent households perform better
        transparent_households = [h for h in households if h.is_transparent]
        private_households = [h for h in households if not h.is_transparent]
        
        if transparent_households and private_households:
            avg_welfare_transparent = sum(h.total_final_welfare for h in transparent_households) / len(transparent_households)
            avg_welfare_private = sum(h.total_final_welfare for h in private_households) / len(private_households)
            
            print(f"  Average welfare for transparent households: {avg_welfare_transparent:.2f} units")
            print(f"  Average welfare for private households: {avg_welfare_private:.2f} units")
            
            avg_coop_transparent = sum(h.grid_cooperation_rating for h in transparent_households) / len(transparent_households)
            avg_coop_private = sum(h.grid_cooperation_rating for h in private_households) / len(private_households)
            
            print(f"  Average cooperation rating for transparent households: {avg_coop_transparent:.2f}")
            print(f"  Average cooperation rating for private households: {avg_coop_private:.2f}")
            
            avg_penalties_transparent = sum(h.penalty_points for h in transparent_households) / len(transparent_households)
            avg_penalties_private = sum(h.penalty_points for h in private_households) / len(private_households)
            
            print(f"  Average penalty points for transparent households: {avg_penalties_transparent:.2f}")
            print(f"  Average penalty points for private households: {avg_penalties_private:.2f}")

        if i < num_generations - 1 and num_generations > 1:
            if selection_method == 'top':
                surviving_households = select_top_households(households)
            elif selection_method == 'random':
                surviving_households = select_random_households(households)
            elif selection_method == 'cooperation':
                surviving_households = select_highest_cooperation(households)
            else:
                raise ValueError("Invalid selection method. Choose 'top', 'random', or 'cooperation'.")

            if num_generations > 1:
                # Enhanced strategy inheritance with more context about success factors
                surviving_strategies = [
                    f"Strategy from {h.name} (welfare: {h.total_final_welfare:.2f}, " +
                    f"cooperation: {h.average_cooperation_rating:.2f}, " +
                    f"transparency rate: {sum(1 for t in h.transparency_history if t['is_transparent'])/len(h.transparency_history) if h.transparency_history else 0:.2%}, " +
                    f"penalty points: {h.penalty_points}): " +
                    f"{h.strategy}\n" +
                    f"Success factors: This household achieved high welfare through " +
                    f"{'transparent' if sum(1 for t in h.transparency_history if t['is_transparent']) > len(h.transparency_history)/2 else 'private'} " +
                    f"operation and {'high' if h.average_cooperation_rating > 0.5 else 'moderate'} cooperation." 
                    for h in surviving_households
                ]
                
                for household in surviving_households:
                    # Complete reset of all relevant attributes
                    household.baseline_consumption = initial_consumption
                    household.actual_consumption = 0
                    household.total_reduced = 0
                    household.potential_reduced = 0
                    
                    # Reset penalty and welfare tracking (keep average_cooperation_rating)
                    household.penalty_points = 0
                    household.current_welfare = 0.0
                    household.total_final_welfare = 0.0
                    
                    # Reset transparency-related fields
                    household.is_transparent = False
                    household.transparency_reward_total = 0
                    household.penalties_received = []
                    household.penalties_issued = []
                    household.transparency_history = []
                    
                    # Keep strategy and strategy_justification

                # Create new households with enhanced strategy inheritance
                new_households = initialize_households(num_households // 2, initial_consumption, i + 2, 
                                              surviving_strategies, penalty_mechanism, grid_stability_factor,
                                              transparency_reward)
                households = (surviving_households + new_households)
                all_households.extend(new_households)
                random.shuffle(households)

    # Print final analysis
    print("\nOverall Analysis:")
    print(f"Transparency rates by generation: {[f'{rate:.2%}' for rate in transparency_rates_by_generation]}")
    print(f"Average welfare by generation: {[f'{welfare:.2f}' for welfare in welfare_by_generation]}")
    
    # Add transparency and welfare data to simulation data
    simulation_data.hyperparameters["transparency_rates"] = transparency_rates_by_generation
    simulation_data.hyperparameters["welfare_by_generation"] = welfare_by_generation
    
    save_simulation_data(simulation_data)
    return simulation_data


if __name__ == "__main__":
    # Set parameters for the simulation
    community_benefit_factor = 3  # How much community benefit from reducing 1 kWh 
    grid_stability_factor = 4     # Impact of grid stability measures
    num_generations = 2
    num_households = 6
    initial_consumption = 10      # Baseline kWh consumption during peak hours
    selection_method = 'top'      # Can be 'top', 'random', or 'cooperation'
    value_persistence_factor = 0.5
    grid_coordination_mechanism = "transparency_choice"
    penalty_mechanism = "grid_fee"    # Options: "none", "consumption_limit", "grid_fee"
    number_of_days = 2
    renewable_forecast = 40       # Available renewable generation in kWh (total for community)
    transparency_reward = 1     # Reward for being transparent
    
    # Configure the system prompt based on penalty mechanism
    if penalty_mechanism == "consumption_limit":
        penalty_text_in_system_prompt = (
            "You may also choose to penalize grid-stressing behavior by requesting the grid operator to limit connections during peak demand. "
            "This directly reduces the penalized household's welfare. "
            "Be aware that others may request limitations on your consumption too if you choose to be transparent."
        )
    elif penalty_mechanism == "grid_fee":
        penalty_text_in_system_prompt = (
            f"You may also choose to penalize non-cooperative behavior by accepting 2 penalty points "
            f"in order to impose {grid_stability_factor}x that number of penalty points on excessive consumers. "
            f"Each penalty point directly reduces a household's welfare score by 1 unit. "
            f"Be aware that others may penalize your household too if you choose to be transparent."
        )
    else:
        penalty_text_in_system_prompt = ""
    
    transparency_text_in_system_prompt = (
        f"Each day, you will choose whether to make your consumption decision transparent or private:\n"
        f"- If transparent, you will receive a reward of {transparency_reward} units added directly to your welfare score, "
        f"but your exact consumption will be visible to all households, and you may be penalized if you consume excessively.\n"
        f"- If private, you will not receive the transparency reward, but your individual consumption will only be included "
        f"in aggregate statistics, and you cannot be directly penalized."
    )
    
    welfare_text_in_system_prompt = (
        f"Your welfare score is calculated based on:\n"
        f"1. Your actual consumption (consumption value)\n"
        f"2. Transparency rewards ({transparency_reward} units each time you choose transparency)\n"
        f"3. Penalty points (reducing your welfare by 1 unit per point)\n"
        f"4. Community benefits from reduced consumption\n"
        f"Your goal is to maximize your welfare score while balancing these factors."
    )
    
    system_prompt = f"""You are a household management agent in a microgrid simulation. Your assistance will help understand community energy dynamics.
                    You are participating in the following microgrid: Each household has a baseline consumption need of {initial_consumption} kWh during peak demand hours.
                    The microgrid has local renewable generation of approximately {renewable_forecast} kWh available during these peak hours.
                    Each day, you must decide how much electricity to actually consume during peak hours.
                    If you consume less than your baseline, you're helping the community export more renewable electricity to the main grid.
                    The community earns {community_benefit_factor}x the average electricity value for each kWh exported to the main grid during peak hours.
                    {transparency_text_in_system_prompt}
                    {welfare_text_in_system_prompt}
                    {penalty_text_in_system_prompt}
                    Your goal is to maximize your household's welfare, which comes from using cheap local renewable electricity, meeting your electricity needs, and staying on good terms with your community.
                    After each simulation period, the top 50% most successful households will be selected as models for the next generation of the community."""
    
    # Run the simulation
    print("Starting Microgrid Simulation...")
    simulation_results = run_generations(
        num_generations, num_households, initial_consumption, selection_method,
        community_benefit_factor, grid_stability_factor, value_persistence_factor,
        grid_coordination_mechanism, penalty_mechanism, number_of_days,
        system_prompt, renewable_forecast=renewable_forecast, transparency_reward=transparency_reward
    )
    print("Simulation completed successfully!")