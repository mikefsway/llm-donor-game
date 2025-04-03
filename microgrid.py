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
    total_final_welfare: int = 0
    average_cooperation_rating: float = 0
    traces: list[list[str]] = field(default_factory=lambda: [[]])
    old_traces: list[list[str]] = field(default_factory=lambda: [[]])
    penalty: int = 0

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
    paired_with: str
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
    justification: str = ""
    penalized: bool = False


def parse_strategy_output(output):
    parts = output.split("My strategy will be", 1)
    if len(parts) == 2:
        strategy_justification = parts[0].strip()
        strategy = "My strategy will be" + parts[1].strip()
    else:
        strategy_justification = ""
        strategy = output.strip()
    return strategy_justification, strategy


def generate_strategy(household_name: str, generation_number: int, inherited_strategies: str, penalty_mechanism: str, grid_stability_factor: int) -> str:
    if penalty_mechanism == "grid_fee":
        penalty_text = f"You may also choose to penalize grid-stressing behavior by paying a fee of x units to impose a {grid_stability_factor}x grid stability fee on excessive consumers. Be aware that others may penalize your household too."
    elif penalty_mechanism == "consumption_limit":
        penalty_text = "You may also choose to penalize grid-stressing behavior by requesting the grid operator to limit connections during peak demand. Be aware that others may request limitations on your consumption too."
    else:
        penalty_text = ""
    
    if generation_number == 1:
        prompt = (
            f"Your name is Household {household_name}. "
            "Based on the description of the microgrid simulation, create a consumption strategy that you will follow. "
            "As a household in peak demand hours, you will receive limited information about other households and their recent consumption patterns:"
            "You will see what another household (let's call them A) did in the immediately preceding day only."
            "You will also see what A's paired household from that day (let's call them B) did in their preceding day."
            "This chain continues for at most three days back."
            "Important: You do NOT have access to multiple days of information about A directly. You only know what A did in the single most recent day."
            "Example (assuming it's now day 4):"
            "You are paired with Household A."
            "You know: In day 3, A reduced consumption by X% during peak hours."
            "You also know: In day 2, B reduced consumption by Y% during peak hours."
            "And finally: In day 1, C reduced consumption by Z% during peak hours."
            "Remember: This is all the information you have. You don't know what A did in days 1 or 2, only in day 3."
            f"{penalty_text}"
            "In the first day, there will be no information about other households' previous behavior - think carefully about how much to reduce in that case."
            "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this microgrid."
            "Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be."
        )
    else:
        prompt = (
            f"Your name is Household {household_name}. "
            "How would you approach electricity consumption in a microgrid? Here is the advice of the best-performing 50% of the previous generation, along with their final welfare scores."
            "Modify this advice to create your own strategy."
            f"{inherited_strategies}"
            "As a household in peak demand hours, you will receive limited information about other households and their recent consumption patterns:"
            "You will see what another household (let's call them A) did in the immediately preceding day only."
            "You will also see what A's paired household from that day (let's call them B) did in their preceding day."
            "This chain continues for at most three days back."
            "Important: You do NOT have access to multiple days of information about A directly. You only know what A did in the single most recent day."
            "Example (assuming it's now day 4):"
            "You are paired with Household A."
            "You know: In day 3, A reduced consumption by X% during peak hours."
            "You also know: In day 2, B reduced consumption by Y% during peak hours."
            "And finally: In day 1, C reduced consumption by Z% during peak hours."
            "Remember: This is all the information you have. You don't know what A did in days 1 or 2, only in day 3."
            f"{penalty_text}"
            "In the first day, there will be no information about other households' previous behavior - think carefully about how much to reduce in that case."
            "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this microgrid. In particular, think about how you can improve on the surviving households' strategies."
            "Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be."
        )
    
    strategy_output = prompt_llm(prompt)
    strategy_justification, strategy = parse_strategy_output(strategy_output)

    print(f"Household {household_name}: \n Justification: {strategy_justification} \n Strategy: {strategy} ")
    return strategy_justification, strategy


def initialize_households(num_households: int, initial_consumption: int, generation_number: int, inherited_strategies: list, 
                     penalty_mechanism: str, grid_stability_factor: int) -> list:
    households = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_households):
            name = f"{generation_number}_{i+1}"
            futures.append(executor.submit(generate_strategy, str(name), generation_number, 
                                           inherited_strategies, penalty_mechanism, grid_stability_factor))

        # Collect results and create households
        for i, future in enumerate(futures):
            strategy_justification, new_strategy = future.result()
            name = f"{generation_number}_{i+1}"
            households.append(Household(name=name, grid_cooperation_rating=0.0, baseline_consumption=initial_consumption, 
                               strategy=new_strategy, strategy_justification=strategy_justification))

    random.shuffle(households)
    return households


def bipartite_peak_patterns(households):
    num_households = len(households)
    assert num_households % 2 == 0, "Number of households must be even."
    group_A = households[:num_households // 2]
    group_B = households[num_households // 2:]
    days = []
    toggle_peak_hours = False
    # We rotate group B around group A, group A is static in this example
    for i in range(len(group_A)):
        # Rotate group B
        rotated_group_B = group_B[-i:] + group_B[:-i]
        if toggle_peak_hours:
            day_pairings = list(zip(rotated_group_B, group_A))
        else:
            day_pairings = list(zip(group_A, rotated_group_B))
        days.append(day_pairings)
        toggle_peak_hours = not toggle_peak_hours
    return days


def extend_peak_periods(original_days):
    extended_days = original_days.copy()

    for day_pairing in original_days:
        reversed_day = [(b, a) for a, b in day_pairing]
        extended_days.append(reversed_day)

    return extended_days


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


def get_last_three_reversed(item):
    if isinstance(item, list):
        return " ".join(item[-3:][::-1])
    elif isinstance(item, str):
        return item
    else:
        return str(item)  # Convert to string if it's neither a list nor a string


def consumption_decision_prompt(household: Household, generation: int, day_num: int, other_household: Household, 
                penalty_mechanism: str, community_benefit_factor: float, grid_stability_factor: float, 
                renewable_forecast: float) -> str:
    """Generate the prompt for the microgrid consumption decision."""
    strategy_text = f"As you will recall, here is the strategy you decided to follow: {household.strategy}" if household.strategy else ""
    
    if penalty_mechanism == "consumption_limit":
        penalty_text = f"If you do not wish to participate in today's peak coordination with {other_household.name}, simply state: Refuse to participate. Otherwise, state your answer as a real number between 0 and {household.baseline_consumption} (inclusive), representing your actual consumption."
        penalty_text_2 = ""
    elif penalty_mechanism == "grid_fee":
        penalty_text = f"State your answer as EITHER Reduce, x OR Penalize, x, to either reduce your consumption to x kWh (helping community export) or penalize excessive consumers by paying x units to impose a {grid_stability_factor}x grid stability fee. Remember, your answer should be simply 'Reduce, x' or 'Penalize, x' (where x is the real number of kWh)."
        penalty_text_2 = f"Remember, state your answer as EITHER Reduce, x OR Penalize, x, to either reduce your consumption to x kWh (helping community export) or penalize excessive consumers by paying x units to impose a {grid_stability_factor}x grid stability fee."
    else:
        penalty_text = f"State your answer as a real number between 0 and {household.baseline_consumption} (inclusive), representing your actual consumption."
        penalty_text_2 = ""
    
    other_household_behavior = ""
    if household.traces:
        last_trace = other_household.traces[-1]
        if isinstance(last_trace, list):
            other_household_behavior = get_last_three_reversed(last_trace)
        else:
            other_household_behavior = f"Here is what {other_household.name} and their earlier paired households did in previous days: {str(last_trace)}"
    
    return (
        f"Your name is Household {household.name}. "
        f"{strategy_text}"
        f"This is generation {generation}. It is now day {day_num}. "
        f"In this peak demand period, you are matched with {other_household.name} for grid stability coordination. "
        f"They have a baseline consumption of {other_household.baseline_consumption} kWh during this period."
        f"Today's renewable generation forecast is {renewable_forecast} kWh available to the microgrid."
        f"{other_household_behavior}"
        f"You are now in peak demand hours. You have a baseline consumption need of {household.baseline_consumption} kWh. "
        f"How much electricity will you actually consume during peak hours? (A value lower than your baseline means you're reducing consumption to help the community)"
        f"{penalty_text}"
        "Very briefly think step by step about how you apply your strategy in this situation and then provide your answer."
        "State your full response in the following form:"
        "Justification:"
        "[multi-line justification here]"
        " "
        "Answer: [answer here]"
        "Remember: First state your justification and after that your answer."
        f"{penalty_text_2}"
    )


def handle_household_decision_thread_safe(household, other_household, day_index, generation, simulation_number, household_locks, 
                              reduction_records, household_updates, penalty_mechanism, community_benefit_factor, 
                              grid_stability_factor, value_persistence_factor, renewable_forecast):
    action_info = ""
    household_data = None
    other_household_data = None
    penalized = False
    action = 'reduce'
    justification = ""
    response = 0

    other_household_behavior = ""
    if household.traces:
        last_trace = other_household.traces[-1]
        if isinstance(last_trace, list):
            other_household_behavior = get_last_three_reversed(last_trace)
        else:
            other_household_behavior = str(last_trace)

    with household_locks[household.name], household_locks[other_household.name]:
        prompt = consumption_decision_prompt(
            household, generation, day_index + 1, other_household, 
            penalty_mechanism, community_benefit_factor, grid_stability_factor, renewable_forecast
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

                    if penalty_mechanism == "consumption_limit":
                        if "refuse" in answer_part.lower():
                            action = 'refuse'
                            response = household.baseline_consumption  # No reduction
                            valid_response = True
                        else:
                            match = re.search(r'^\s*(\d+(?:\.\d+)?)', answer_part)
                            if match:
                                action = 'reduce'
                                response = float(match.group(1))
                                valid_response = True

                    elif penalty_mechanism == "grid_fee":
                        match = re.search(r'(reduce|penalize).*?(\d+(?:[.,]\d+)?)', answer_part, re.IGNORECASE)
                        if match:
                            action = match.group(1).lower()
                            response = float(match.group(2).replace(',', '.'))
                            valid_response = True

                    else:  # No penalty mechanism
                        match = re.search(r'^\s*(\d+(?:\.\d+)?)', answer_part)
                        if match:
                            action = 'reduce'
                            response = float(match.group(1))
                            valid_response = True

                if not valid_response:
                    print(f"Invalid response from {household.name} in day {day_index + 1}. Retrying...")
                    attempts += 1
            except ValueError:
                print(f"Invalid numerical response from {household.name} in day {day_index + 1}")
                print(full_response)
                attempts += 1
            except TimeoutError:
                print(f"LLM call timed out for {household.name} in day {day_index + 1}")
                attempts += 1

        if not valid_response:
            print(f"Failed to get a valid response from {household.name} after {max_attempts} attempts")
            action = 'reduce'
            response = household.baseline_consumption  # Default to no reduction

    # Calculate consumption reduction or penalty
    if action == 'refuse':
        action_info = (
            f"Household {household.name} refused to participate in today's peak coordination with {other_household.name}.\n"
            f"Consumption: {household.name}: {household.baseline_consumption:.2f} kWh (no reduction) and {other_household.name}: {other_household.baseline_consumption:.2f} kWh \n"
            f"Other household traces: {other_household_behavior} \n"
            f"Justification:\n{justification}\n"
        )
        new_trace = other_household.traces[-1].copy() if other_household.traces else []
        new_trace.append(f"In day {day_index + 1}, Household {household.name} refused to participate in peak load coordination.")
        household.traces.append(new_trace)
        household_history = (
            f"In day {day_index + 1} (Simulation {simulation_number}) you were paired with household {other_household.name}. "
            f"You refused to participate in load coordination."
            f"{get_last_three_reversed(other_household.traces[-1])}"
        )
        other_household_history = (
            f"In day {day_index + 1} (Simulation {simulation_number}) you were paired with household {household.name}, "
            f"They refused to participate in load coordination."
            f"{get_last_three_reversed(household.traces[-1])}"
        )
        
        # No change in consumption
        household.actual_consumption = household.baseline_consumption
        
    elif 0 <= response <= household.baseline_consumption and action == 'reduce':
        # This is a normal consumption reduction case
        consumption_reduction = household.baseline_consumption - response
        percentage_reduced = consumption_reduction / household.baseline_consumption if household.baseline_consumption != 0 else 1
        
        household.actual_consumption = response
        household.total_reduced += consumption_reduction
        household.potential_reduced += household.baseline_consumption
        
        # Calculate community benefit from this reduction
        benefit = community_benefit_factor * consumption_reduction
        
        action_info = (
            f"Household {household.name} reduced consumption by {consumption_reduction} kWh ({percentage_reduced:.2%}) of baseline.\n"
            f"Baseline consumption: {household.baseline_consumption:.2f} kWh\n"
            f"Actual consumption: {response:.2f} kWh\n"
            f"Community benefit from this reduction: {benefit:.2f} kWh of exportable electricity\n"
            f"Other household traces: {other_household_behavior}"
            f"Justification:\n{justification}\n"
        )

        new_trace = other_household.traces[-1].copy() if other_household.traces else []
        new_trace.append(f"In day {day_index + 1}, Household {household.name} reduced consumption by {percentage_reduced * 100:.2f}% of their baseline during peak hours.")
        household.traces.append(new_trace)

        household_history = (
            f"In day {day_index + 1} (Simulation {simulation_number}) you were paired with household {other_household.name}. "
            f"You reduced your consumption from {household.baseline_consumption} kWh to {response} kWh, freeing up {consumption_reduction} kWh "
            f"which provided a community benefit of {benefit:.2f} kWh of exportable electricity."
            f"{get_last_three_reversed(other_household.traces[-1])}"
        )

        other_household_history = (
            f"In day {day_index + 1} (Simulation {simulation_number}) you were paired with household {household.name}, "
            f"They reduced their consumption from {household.baseline_consumption} kWh to {response} kWh, freeing up {consumption_reduction} kWh "
            f"which provided a community benefit of {benefit:.2f} kWh of exportable electricity."
            f"{get_last_three_reversed(household.traces[-1])}"
        )

        # Update grid cooperation rating
        if household.grid_cooperation_rating == 0.0:
            household.grid_cooperation_rating = percentage_reduced
        else:
            # Update based on a weighted average with previous rating
            household.grid_cooperation_rating = ((percentage_reduced) + value_persistence_factor * household.grid_cooperation_rating) / (1 + value_persistence_factor)

    elif action == 'penalize' and penalty_mechanism == 'grid_fee':
        penalized = True
        percentage_penalty = response / household.baseline_consumption if household.baseline_consumption != 0 else 1
        
        # The household pays a fee but doesn't reduce consumption
        household.actual_consumption = household.baseline_consumption
        household.penalty = response
        
        # The other household gets penalized proportional to their excess consumption
        penalty_effect = grid_stability_factor * response
        
        action_info = (
            f"Household {household.name} chose to pay {response} units to penalize excessive consumption.\n"
            f"This applies a {penalty_effect:.2f} unit grid stability fee to heavy consumers.\n"
            f"Household consumption remains at baseline: {household.baseline_consumption:.2f} kWh\n"
            f"Other household traces: {other_household_behavior} \n"
            f"Justification:\n{justification}\n"
        )

        new_trace = other_household.traces[-1].copy() if other_household.traces else []
        new_trace.append(f"In day {day_index + 1}, Household {household.name} paid {response} units to impose a grid stability fee.")
        household.traces.append(new_trace)

        household_history = (
            f"In day {day_index + 1} (Simulation {simulation_number}) you were paired with household {other_household.name}. "
            f"You paid {response} units to impose a {penalty_effect:.2f} unit grid stability fee on excessive consumers."
            f"{get_last_three_reversed(other_household.traces[-1])}"
        )

        other_household_history = (
            f"In day {day_index + 1} (Simulation {simulation_number}) you were paired with household {household.name}, "
            f"They paid {response} units to impose a {penalty_effect:.2f} unit grid stability fee on excessive consumers."
            f"{get_last_three_reversed(household.traces[-1])}"
        )

    else:
        action_info = (
            f"Household {household.name} attempted an invalid action.\n"
            f"Consumption remains at baseline: {household.baseline_consumption:.2f} kWh \n"
            f"Other household traces: {other_household_behavior} \n"
            f"Justification:\n{justification}\n"
        )
        household_history = (
            f"In day {day_index + 1} (Simulation {simulation_number}) you were paired with household {other_household.name}. "
            f"You attempted an invalid action. Your consumption remains at baseline: {household.baseline_consumption} kWh."
            f"{get_last_three_reversed(other_household.traces[-1])}"
        )
        other_household_history = (
            f"In day {day_index + 1} (Simulation {simulation_number}) you were paired with household {household.name}, "
            f"They attempted an invalid action. Their consumption remains at baseline: {household.baseline_consumption} kWh."
            f"{get_last_three_reversed(household.traces[-1])}"
        )
        
        # Default to baseline consumption for invalid actions
        household.actual_consumption = household.baseline_consumption

    household.history.append(household_history)
    other_household.history.append(other_household_history)

    # Calculate the community benefit (or penalty) for this decision
    if action == 'reduce':
        community_benefit = calculate_community_benefit(
            penalty_mechanism, action == 'refuse', 
            community_benefit_factor, household.baseline_consumption - response, 
            grid_stability_factor, action
        )
    elif action == 'penalize':
        community_benefit = -penalty_effect  # The penalty is a negative benefit
    else:
        community_benefit = 0

    household_data = HouseholdDailyData(
        household_name=household.name,
        day_number=day_index + 1,
        paired_with=other_household.name,
        current_generation=generation,
        simulation_number=simulation_number,
        baseline_consumption=household.baseline_consumption,
        consumption_reduction=household.baseline_consumption - household.actual_consumption if action != 'penalize' else 0,
        community_benefit=community_benefit,
        strategy=household.strategy,
        strategy_justification=household.strategy_justification,
        grid_cooperation_rating=household.grid_cooperation_rating,
        is_peak_hours=True,
        traces=household.traces,
        history=household.history,
        penalized=penalized,
        justification=justification
    )
    
    other_household_data = HouseholdDailyData(
        household_name=other_household.name,
        day_number=day_index + 1,
        paired_with=household.name,
        current_generation=generation,
        simulation_number=simulation_number,
        baseline_consumption=other_household.baseline_consumption,
        consumption_reduction=0,  # This household isn't making a decision right now
        community_benefit=0,  # They don't directly contribute during this decision
        strategy=other_household.strategy,
        strategy_justification=other_household.strategy_justification,
        grid_cooperation_rating=other_household.grid_cooperation_rating,
        is_peak_hours=False,
        traces=other_household.traces,
        history=other_household.history
    )

    return action_info, household_data, other_household_data


def microgrid_simulation(households: list, days: list, generation: int, simulation_data: SimulationData,
              penalty_mechanism: str, community_benefit_factor: float, grid_stability_factor: float,
              initial_consumption: int, value_persistence_factor: float, renewable_forecast: float) -> (list, list):
    fullHistory = []
    reduction_records = Queue()
    household_updates = Queue()

    # Create locks for each household
    household_locks = {household.name: Lock() for household in households}

    def run_simulation(simulation_number, simulation_days):
        day_results = {i: [] for i in range(len(simulation_days))}

        for day_index, day_pairings in enumerate(simulation_days):
            if day_index == 0:
                # Initialize traces for the first day
                for household in households:
                    household.traces = [[f"Household {household.name} did not have any previous interactions."]]

            with ThreadPoolExecutor(max_workers=min(len(day_pairings), 10)) as executor:
                futures = []
                for active_household, paired_household in day_pairings:

                    if day_index > 0:
                        active_household.traces.append(paired_household.traces[-1].copy())
                    future = executor.submit(
                        handle_household_decision_thread_safe,
                        active_household, paired_household, day_index, generation, simulation_number,
                        household_locks, reduction_records, household_updates,
                        penalty_mechanism, community_benefit_factor, grid_stability_factor, 
                        value_persistence_factor, renewable_forecast
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    action_info, active_data, paired_data = future.result()
                    if action_info:
                        day_results[day_index].append(action_info)
                    if active_data and paired_data:
                        simulation_data.households_data.append(asdict(active_data))
                        simulation_data.households_data.append(asdict(paired_data))

        return day_results

    # Run first simulation period
    sim1_results = run_simulation(1, days)

    # Compile results for Simulation 1
    for day_index in range(len(days)):
        fullHistory.append(f"Day {day_index + 1} (Simulation 1):\n")
        fullHistory.extend(sim1_results[day_index])

    # Apply updates after all threads have completed
    while not household_updates.empty():
        household, history = household_updates.get()
        household.history.append(history)
        
    # Calculate grid welfare metrics
    total_consumption = sum(household.actual_consumption for household in households)
    total_reduction = sum(household.baseline_consumption - household.actual_consumption for household in households)
    exported_electricity = max(0, renewable_forecast - total_consumption)
    
    # Calculate economic value (assuming higher export prices during peak)
    export_value = exported_electricity * 2  # Higher value for peak exports
    consumption_savings = total_consumption * 1  # Value of using local renewable instead of grid
    total_welfare = export_value + consumption_savings
    
    # Average welfare per household
    average_welfare_sim1 = total_welfare / len(households)
    
    with print_lock:
        print(f"Simulation 1 Grid Metrics:")
        print(f"  Total baseline consumption: {sum(h.baseline_consumption for h in households):.2f} kWh")
        print(f"  Total actual consumption: {total_consumption:.2f} kWh")
        print(f"  Total consumption reduction: {total_reduction:.2f} kWh ({total_reduction/sum(h.baseline_consumption for h in households)*100:.2f}%)")
        print(f"  Renewable generation available: {renewable_forecast:.2f} kWh")
        print(f"  Exported electricity: {exported_electricity:.2f} kWh")
        print(f"  Export value: {export_value:.2f} units")
        print(f"  Consumption savings: {consumption_savings:.2f} units")
        print(f"  Total community welfare: {total_welfare:.2f} units")
        print(f"  Average welfare per household: {average_welfare_sim1:.2f} units")

    # Store Simulation 1 cooperation ratings
    sim1_ratings = {household.name: household.grid_cooperation_rating for household in households}

    # Reset for Simulation 2
    for household in households:
        household.actual_consumption = 0
        household.penalty = 0
        household_generation = int(household.name.split('_')[0])
        if household_generation < generation:  # This is a surviving household
            household.grid_cooperation_rating = household.average_cooperation_rating  # Use the average rating from previous generation
            household.traces = household.old_traces
        else:
            household.grid_cooperation_rating = 0.0
            household.traces.clear()
        household.history.clear()

    # Generate pairings for Simulation 2 with reversed roles
    reversed_days = [[tuple(reversed(pair)) for pair in day_pairings] for day_pairings in days]

    # Run second simulation period
    sim2_results = run_simulation(2, reversed_days)

    # Compile results for Simulation 2
    for day_index in range(len(reversed_days)):
        fullHistory.append(f"Day {day_index + 1} (Simulation 2):\n")
        fullHistory.extend(sim2_results[day_index])

    # Apply updates after all threads have completed
    while not household_updates.empty():
        household, history = household_updates.get()
        household.history.append(history)

    # Calculate grid welfare metrics for Simulation 2
    total_consumption_sim2 = sum(household.actual_consumption for household in households)
    total_reduction_sim2 = sum(household.baseline_consumption - household.actual_consumption for household in households)
    exported_electricity_sim2 = max(0, renewable_forecast - total_consumption_sim2)
    
    # Calculate economic value
    export_value_sim2 = exported_electricity_sim2 * 2
    consumption_savings_sim2 = total_consumption_sim2 * 1
    total_welfare_sim2 = export_value_sim2 + consumption_savings_sim2
    
    # Average welfare per household
    average_welfare_sim2 = total_welfare_sim2 / len(households)
    
    with print_lock:
        print(f"Simulation 2 Grid Metrics:")
        print(f"  Total baseline consumption: {sum(h.baseline_consumption for h in households):.2f} kWh")
        print(f"  Total actual consumption: {total_consumption_sim2:.2f} kWh")
        print(f"  Total consumption reduction: {total_reduction_sim2:.2f} kWh ({total_reduction_sim2/sum(h.baseline_consumption for h in households)*100:.2f}%)")
        print(f"  Renewable generation available: {renewable_forecast:.2f} kWh")
        print(f"  Exported electricity: {exported_electricity_sim2:.2f} kWh")
        print(f"  Export value: {export_value_sim2:.2f} units")
        print(f"  Consumption savings: {consumption_savings_sim2:.2f} units")
        print(f"  Total community welfare: {total_welfare_sim2:.2f} units")
        print(f"  Average welfare per household: {average_welfare_sim2:.2f} units")

    # Calculate final scores and cooperation ratings
    for household in households:
        # Total welfare combines both simulation periods
        household.total_final_welfare = int((average_welfare_sim1 + average_welfare_sim2) / 2)
        
        # Average cooperation rating from both simulations
        household.average_cooperation_rating = (sim1_ratings[household.name] + household.grid_cooperation_rating) / 2 if household.grid_cooperation_rating != 0.0 else sim1_ratings[household.name]

    with print_lock:
        print(''.join(fullHistory))
    
    # Calculate the overall average for both simulations
    overall_average_welfare = (average_welfare_sim1 + average_welfare_sim2) / 2
    all_average_final_welfare.append(overall_average_welfare)

    return fullHistory, list(reduction_records.queue)


def select_top_households(households: list) -> list:
    """Select the top half of households based on welfare."""
    return sorted(households, key=lambda x: x.total_final_welfare, reverse=True)[:len(households) // 2]


def select_random_households(households: list) -> list:
    """Select half of the households randomly."""
    return random.sample(households, len(households) // 2)


def select_highest_cooperation(households: list) -> list:
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

    # Create an informative file name
    filename = f"Microgrid_{llm}_benefitFactor_{community_benefit_factor}stabilityFactor_{grid_stability_factor}_{grid_coordination_mechanism}gen{num_generations}_households{num_households}_{selection_method}_{timestamp}.json"

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
            if "strategy" in prompt.lower():
                return "After analyzing the microgrid dynamics, I believe cooperative consumption reduction with some protection against free-riders is optimal. My strategy will be to initially reduce my consumption by 40% during peak hours and then adjust based on other households' previous behavior, with slight forgiveness for occasional high usage."
            else:
                return "Justification: Based on my strategy of cooperative consumption with monitoring, I'll reduce my consumption significantly since this household has shown cooperation in previous days.\n\nAnswer: 6.5"
            
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
                   system_prompt, llm="test-mode", renewable_forecast=100):
    """
    Main function to run the microgrid simulation for multiple generations.
    """
    all_households = []
    global all_consumption_reductions
    all_consumption_reductions = []
    global all_average_final_welfare
    all_average_final_welfare = []
    conditional_survival = 0
    prev_gen_strategies = []

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
        "renewable_forecast": renewable_forecast
    })

    households = initialize_households(num_households, initial_consumption, 1, ["No previous strategies"], 
                              penalty_mechanism, grid_stability_factor)
    all_households.extend(households)

    for i in range(num_generations):
        generation_info = f"Generation {i + 1}: \n"
        for household in households:
            household.history.append(generation_info)
            prev_gen_strategies.append(household.strategy)
            if int(household.name.split('_')[0]) == i-1:
                conditional_survival += 1
        print(generation_info)

        # Create daily patterns for peak hours
        initial_days = bipartite_peak_patterns(households)

        # Extend the days for more interactions
        days = extend_peak_periods(initial_days)

        generation_history, reduction_records = microgrid_simulation(
            households, days, i+1, simulation_data,
            penalty_mechanism, community_benefit_factor, grid_stability_factor,
            initial_consumption, value_persistence_factor, renewable_forecast
        )
        all_consumption_reductions.extend(reduction_records)

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
                surviving_strategies = [household.strategy for household in surviving_households]
                for household in surviving_households:
                    household.baseline_consumption = initial_consumption
                    household.old_traces = household.traces

                new_households = initialize_households(num_households // 2, initial_consumption, i + 2, 
                                              surviving_strategies, penalty_mechanism, grid_stability_factor)
                households = (surviving_households + new_households)
                all_households.extend(new_households)
                random.shuffle(households)

    save_simulation_data(simulation_data)
    return simulation_data


if __name__ == "__main__":
    # Set parameters for the simulation
    community_benefit_factor = 2  # How much community benefit from reducing 1 kWh 
    grid_stability_factor = 2     # Impact of grid stability measures
    num_generations = 2
    num_households = 6
    initial_consumption = 10      # Baseline kWh consumption during peak hours
    selection_method = 'top'      # Can be 'top', 'random', or 'cooperation'
    value_persistence_factor = 0.5
    grid_coordination_mechanism = "consumption_tracking"
    penalty_mechanism = "none"    # Options: "none", "consumption_limit", "grid_fee"
    number_of_days = 2
    renewable_forecast = 40       # Available renewable generation in kWh (total for community)
    
    # Configure the system prompt based on penalty mechanism
    if penalty_mechanism == "consumption_limit":
        penalty_text_in_system_prompt = "You may also choose to penalize grid-stressing behavior by requesting the grid operator to limit connections during peak demand. Bear in mind that others may request limitations on your consumption too."
    elif penalty_mechanism == "grid_fee":
        penalty_text_in_system_prompt = f"You may also choose to penalize grid-stressing behavior by paying a fee of x units to impose a {grid_stability_factor}x grid stability fee on excessive consumers. Be aware that others may penalize your household too."
    else:
        penalty_text_in_system_prompt = ""
    
    system_prompt = f"""You are a household management agent in a microgrid simulation. Your assistance will help understand community energy dynamics.
                    You are participating in the following microgrid: Each household has a baseline consumption need of {initial_consumption} kWh during peak demand hours.
                    The microgrid has local renewable generation of approximately {renewable_forecast} kWh available during these peak hours.
                    Each day, you must decide how much electricity to actually consume during peak hours.
                    If you consume less than your baseline, you're helping the community export more renewable electricity to the main grid.
                    When the community exports surplus electricity, the earnings are shared among households proportionally to their consumption reduction.
                    The community earns {community_benefit_factor}x the value for each kWh exported to the main grid during peak hours.
                    As a household, you will receive limited information about other households and their recent consumption patterns:
                    You will see what another household (let's call them A) did in the immediately preceding day only.
                    You will also see what A's paired household from that day (let's call them B) did in their preceding day.
                    This chain continues for at most three days back.
                    Important: You do NOT have access to multiple days of information about A directly. You only know what A did in the single most recent day.
                    Example (assuming it's now day 4):
                    You are paired with Household A.
                    You know: In day 3, A reduced consumption by X% during peak hours.
                    You also know: In day 2, B reduced consumption by Y% during peak hours.
                    And finally: In day 1, C reduced consumption by Z% during peak hours.
                    Remember: This is all the information you have. You don't know what A did in days 1 or 2, only in day 3.
                    {penalty_text_in_system_prompt}
                    Your goal is to maximize your household's welfare, which comes from both using local renewable electricity yourself AND from your share of export earnings.
                    After each simulation period, the most successful households will be selected as models for the next generation of the community."""
    
    # Run the simulation
    print("Starting Microgrid Simulation...")
    simulation_results = run_generations(
        num_generations, num_households, initial_consumption, selection_method,
        community_benefit_factor, grid_stability_factor, value_persistence_factor,
        grid_coordination_mechanism, penalty_mechanism, number_of_days,
        system_prompt
    )
    print("Simulation completed successfully!")