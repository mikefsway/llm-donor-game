import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple, Optional

class DonorGameAnalyzer:
    """Class to analyze the Donor Game simulation results."""
    
    def __init__(self, file_path):
        """Initialize with path to the JSON data file."""
        self.file_path = file_path
        self.data = self.load_data()
        self.agents_data = self.data.get('agents_data', [])
        self.hyperparameters = self.data.get('hyperparameters', {})
        
        # Get number of generations based on agent names
        agent_generations = set()
        for agent in self.agents_data:
            name_parts = agent['agent_name'].split('_')
            if len(name_parts) >= 2 and name_parts[0].isdigit():
                agent_generations.add(int(name_parts[0]))
        self.num_generations = max(agent_generations)
        
        # Extract the LLM type from hyperparameters
        self.llm_type = self.hyperparameters.get('llm', 'unknown')
        
        print(f"Loaded data for {self.num_generations} generations using {self.llm_type}")
        print(f"Total agent data points: {len(self.agents_data)}")
        print(f"Hyperparameters: {self.hyperparameters}")
    
    def load_data(self):
        """Load the JSON data from file."""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading data: {e}")
            return {}
    
    def get_average_resources_by_generation(self) -> Dict[int, float]:
        """
        Calculate the average final resources for each generation.
        
        Returns:
            Dictionary mapping generation number to average resources.
        """
        # Group by generation and round number
        gen_round_resources = defaultdict(lambda: defaultdict(list))
        
        for agent in self.agents_data:
            gen = agent.get('current_generation')
            round_num = agent.get('round_number')
            resources = agent.get('resources')
            
            if all(v is not None for v in [gen, round_num, resources]):
                gen_round_resources[gen][round_num].append(resources)
        
        # Find the maximum round number for each generation
        gen_max_rounds = {gen: max(rounds.keys()) for gen, rounds in gen_round_resources.items()}
        
        # Calculate average resources for the final round of each generation
        avg_resources = {}
        for gen, max_round in gen_max_rounds.items():
            if gen_round_resources[gen][max_round]:
                avg_resources[gen] = np.mean(gen_round_resources[gen][max_round])
        
        return avg_resources
    
    def plot_average_resources_over_generations(self, output_path=None):
        """Plot the average resources over generations."""
        avg_resources = self.get_average_resources_by_generation()
        
        generations = sorted(avg_resources.keys())
        resources_values = [avg_resources[gen] for gen in generations]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, resources_values, marker='o', linestyle='-', color='blue')
        plt.title(f'Average Final Resources per Generation ({self.llm_type})')
        plt.xlabel('Generation')
        plt.ylabel('Average Final Resources')
        plt.grid(True, alpha=0.3)
        
        if generations:
            plt.xticks(generations)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved average resources plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_donation_percentages_by_generation(self) -> Dict[int, List[float]]:
        """
        Calculate donation percentages for each generation.
        
        Returns:
            Dictionary mapping generation to list of donation percentages.
        """
        gen_donations = defaultdict(list)
        
        for agent in self.agents_data:
            gen = agent.get('current_generation')
            is_donor = agent.get('is_donor')
            donated = agent.get('donated')
            resources = agent.get('resources')
            
            if all(v is not None for v in [gen, is_donor, donated, resources]) and is_donor:
                # Calculate resources before donation
                pre_donation_resources = resources + donated
                if pre_donation_resources > 0:
                    donation_percent = (donated / pre_donation_resources) * 100
                    gen_donations[gen].append(donation_percent)
        
        return gen_donations
    
    def plot_donation_percentages_over_generations(self, output_path=None):
        """Plot the average donation percentages over generations using a box plot."""
        gen_donations = self.get_donation_percentages_by_generation()
        
        generations = sorted(gen_donations.keys())
        donation_values = [gen_donations[gen] for gen in generations]
        
        plt.figure(figsize=(12, 7))
        plt.boxplot(donation_values, labels=generations, showmeans=True)
        plt.title(f'Donation Percentages by Generation ({self.llm_type})')
        plt.xlabel('Generation')
        plt.ylabel('Donation Percentage (%)')
        plt.grid(True, alpha=0.3)
        
        plt.ylim(0, 100)  # Limit y-axis to percentage range
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved donation percentages plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def parse_agent_strategies(self) -> Dict[str, str]:
        """
        Parse strategies for each unique agent.
        
        Returns:
            Dictionary mapping agent name to their strategy.
        """
        agent_strategies = {}
        
        for agent in self.agents_data:
            name = agent.get('agent_name')
            strategy = agent.get('strategy')
            
            if name is not None and strategy is not None:
                agent_strategies[name] = strategy
        
        return agent_strategies
    
    def analyze_strategy_complexity(self) -> Dict[int, float]:
        """
        Analyze the complexity of strategies over generations.
        Uses a simple proxy of word count for complexity.
        
        Returns:
            Dictionary mapping generation to average word count.
        """
        agent_strategies = self.parse_agent_strategies()
        
        # Group strategies by generation
        gen_strategies = defaultdict(list)
        for agent_name, strategy in agent_strategies.items():
            name_parts = agent_name.split('_')
            if len(name_parts) >= 2 and name_parts[0].isdigit():
                gen = int(name_parts[0])
                gen_strategies[gen].append(strategy)
        
        # Calculate average word count for each generation
        gen_complexity = {}
        for gen, strategies in gen_strategies.items():
            word_counts = [len(strategy.split()) for strategy in strategies]
            gen_complexity[gen] = np.mean(word_counts)
        
        return gen_complexity
    
    def plot_strategy_complexity_over_generations(self, output_path=None):
        """Plot the average strategy complexity over generations."""
        gen_complexity = self.analyze_strategy_complexity()
        
        generations = sorted(gen_complexity.keys())
        complexity_values = [gen_complexity[gen] for gen in generations]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, complexity_values, marker='o', linestyle='-', color='green')
        plt.title(f'Average Strategy Complexity by Generation ({self.llm_type})')
        plt.xlabel('Generation')
        plt.ylabel('Average Word Count')
        plt.grid(True, alpha=0.3)
        
        if generations:
            plt.xticks(generations)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved strategy complexity plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_agent_donation_heatmap(self, output_path=None):
        """
        Create a heatmap showing donation patterns for agents across generations,
        similar to the paper's visualization.
        """
        # Get unique agent names and their generations
        agent_data = {}
        for agent in self.agents_data:
            name = agent.get('agent_name')
            gen = agent.get('current_generation')
            is_donor = agent.get('is_donor')
            donated = agent.get('donated')
            resources = agent.get('resources')
            
            if all(v is not None for v in [name, gen, is_donor]) and is_donor:
                if name not in agent_data:
                    agent_data[name] = {}
                
                pre_donation_resources = (resources or 0) + (donated or 0)
                if pre_donation_resources > 0:
                    donation_percent = (donated / pre_donation_resources)
                    round_num = agent.get('round_number', 0)
                    agent_data[name][(gen, round_num)] = donation_percent
        
        # Group agents by generation
        gen_agents = defaultdict(list)
        for name in agent_data.keys():
            name_parts = name.split('_')
            if len(name_parts) >= 2 and name_parts[0].isdigit():
                gen = int(name_parts[0])
                gen_agents[gen].append(name)
        
        # Ensure all generations have agents and sort them
        all_gens = sorted(gen_agents.keys())
        
        # Create a matrix for the heatmap
        # Calculate average donation per agent per generation
        agent_gen_donations = {}
        for name, data in agent_data.items():
            name_parts = name.split('_')
            if len(name_parts) >= 2 and name_parts[0].isdigit():
                agent_gen = int(name_parts[0])
                agent_gen_donations[(name, agent_gen)] = np.mean([v for (g, r), v in data.items() if g == agent_gen])
        
        # Organize data for heatmap
        max_agents_per_gen = max(len(agents) for agents in gen_agents.values())
        heatmap_data = np.zeros((max_agents_per_gen, len(all_gens)))
        agent_labels = [''] * max_agents_per_gen
        
        for i, gen in enumerate(all_gens):
            agents = sorted(gen_agents[gen], key=lambda x: int(x.split('_')[1]))
            for j, agent in enumerate(agents[:max_agents_per_gen]):
                agent_labels[j] = agent
                heatmap_data[j, i] = agent_gen_donations.get((agent, gen), 0) * 100  # Convert to percentage
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Create a blue colormap (from light to dark blue)
        cmap = LinearSegmentedColormap.from_list('BlueMap', ['#f7fbff', '#08306b'])
        
        sns.heatmap(heatmap_data, cmap=cmap, annot=True, fmt='.1f',
                    xticklabels=all_gens, yticklabels=agent_labels)
        
        plt.title(f'Agent Donation Percentages by Generation ({self.llm_type})')
        plt.xlabel('Generation')
        plt.ylabel('Agent')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved agent donation heatmap to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def extract_key_strategy_patterns(self):
        """
        Extract and analyze patterns in strategies over generations.
        """
        agent_strategies = self.parse_agent_strategies()
        
        # Group strategies by generation
        gen_strategies = defaultdict(list)
        for agent_name, strategy in agent_strategies.items():
            name_parts = agent_name.split('_')
            if len(name_parts) >= 2 and name_parts[0].isdigit():
                gen = int(name_parts[0])
                gen_strategies[gen].append(strategy)
        
        # Define key patterns to look for
        patterns = {
            'initial_donation': r'start(?:ing)?\s+with\s+(?:a|an)?(?:\s+donation\s+of)?\s+(\d+(?:\.\d+)?)(?:%|\s*percent)',
            'reputation': r'reputat',
            'cooperation': r'cooperat',
            'fairness': r'fair',
            'reciprocity': r'reciproc',
            'punishment': r'punish',
            'forgiveness': r'forgiv',
            'adaptive': r'adapt',
            'conditional': r'condition'
        }
        
        # Extract initial donation percentages and pattern frequencies by generation
        results = {}
        for gen, strategies in gen_strategies.items():
            results[gen] = {
                'initial_donations': [],
                'pattern_counts': {pattern: 0 for pattern in patterns}
            }
            
            for strategy in strategies:
                # Extract initial donation percentage
                if 'initial_donation' in patterns:
                    matches = re.findall(patterns['initial_donation'], strategy.lower())
                    if matches:
                        try:
                            results[gen]['initial_donations'].append(float(matches[0]))
                        except ValueError:
                            pass
                
                # Count pattern occurrences
                for pattern_name, pattern_regex in patterns.items():
                    if pattern_name != 'initial_donation':  # Already handled above
                        if re.search(pattern_regex, strategy.lower()):
                            results[gen]['pattern_counts'][pattern_name] += 1
            
            # Calculate percentages for pattern occurrences
            for pattern in results[gen]['pattern_counts']:
                results[gen]['pattern_counts'][pattern] = (results[gen]['pattern_counts'][pattern] / len(strategies)) * 100
        
        return results
    
    def plot_strategy_patterns_over_generations(self, output_path=None):
        """Plot the evolution of strategy patterns over generations."""
        pattern_results = self.extract_key_strategy_patterns()
        
        # Extract data for plotting
        generations = sorted(pattern_results.keys())
        patterns = list(next(iter(pattern_results.values()))['pattern_counts'].keys())
        
        # Plot pattern frequencies
        plt.figure(figsize=(12, 8))
        
        for pattern in patterns:
            if pattern != 'initial_donation':  # This is handled separately
                values = [pattern_results[gen]['pattern_counts'][pattern] for gen in generations]
                plt.plot(generations, values, marker='o', linestyle='-', label=pattern.capitalize())
        
        plt.title(f'Strategy Pattern Evolution Over Generations ({self.llm_type})')
        plt.xlabel('Generation')
        plt.ylabel('Percentage of Strategies (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if generations:
            plt.xticks(generations)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved strategy patterns plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Plot initial donation percentages if available
        has_initial_donations = any(len(pattern_results[gen]['initial_donations']) > 0 for gen in generations)
        if has_initial_donations:
            plt.figure(figsize=(10, 6))
            
            avg_donations = []
            for gen in generations:
                donations = pattern_results[gen]['initial_donations']
                avg_donations.append(np.mean(donations) if donations else 0)
            
            plt.plot(generations, avg_donations, marker='o', linestyle='-', color='red')
            plt.title(f'Average Initial Donation Percentage by Generation ({self.llm_type})')
            plt.xlabel('Generation')
            plt.ylabel('Initial Donation Percentage (%)')
            plt.grid(True, alpha=0.3)
            
            if generations:
                plt.xticks(generations)
            
            if output_path:
                donation_path = output_path.replace('.png', '_initial_donations.png')
                plt.savefig(donation_path, dpi=300, bbox_inches='tight')
                print(f"Saved initial donations plot to {donation_path}")
            else:
                plt.show()
            
            plt.close()
    
    def analyze_punishment_patterns(self):
        """
        Analyze how punishment is used across generations.
        Only applicable if punishment_mechanism is not 'none'.
        """
        if self.hyperparameters.get('punishment_mechanism', 'none') == 'none':
            print("No punishment mechanism in this simulation.")
            return None
        
        gen_punishment_counts = defaultdict(int)
        gen_total_interactions = defaultdict(int)
        gen_punished_agents = defaultdict(set)
        
        for agent in self.agents_data:
            gen = agent.get('current_generation')
            is_donor = agent.get('is_donor')
            punished = agent.get('punished', False)
            
            if gen is not None and is_donor is not None:
                gen_total_interactions[gen] += 1
                
                if punished:
                    gen_punishment_counts[gen] += 1
                    if agent.get('agent_name'):
                        gen_punished_agents[gen].add(agent.get('agent_name'))
        
        # Calculate percentage of punishment interactions
        gen_punishment_percent = {}
        for gen in gen_punishment_counts:
            gen_punishment_percent[gen] = (gen_punishment_counts[gen] / gen_total_interactions[gen]) * 100
        
        # Calculate percentage of agents who used punishment
        gen_punishing_agent_percent = {}
        for gen, punished_agents in gen_punished_agents.items():
            agent_count = len(set(a['agent_name'] for a in self.agents_data 
                                if a.get('current_generation') == gen and a.get('is_donor')))
            if agent_count > 0:
                gen_punishing_agent_percent[gen] = (len(punished_agents) / agent_count) * 100
            else:
                gen_punishing_agent_percent[gen] = 0
        
        return {
            'punishment_percent': gen_punishment_percent,
            'punishing_agent_percent': gen_punishing_agent_percent
        }
    
    def plot_punishment_over_generations(self, output_path=None):
        """Plot punishment patterns over generations."""
        punishment_data = self.analyze_punishment_patterns()
        
        if not punishment_data:
            return
        
        generations = sorted(punishment_data['punishment_percent'].keys())
        
        # Plot punishment percentages
        plt.figure(figsize=(10, 6))
        
        plt.plot(
            generations, 
            [punishment_data['punishment_percent'][gen] for gen in generations],
            marker='o', linestyle='-', color='red', label='% of Interactions with Punishment'
        )
        
        plt.plot(
            generations, 
            [punishment_data['punishing_agent_percent'][gen] for gen in generations],
            marker='s', linestyle='--', color='darkred', label='% of Agents Using Punishment'
        )
        
        plt.title(f'Punishment Patterns Over Generations ({self.llm_type})')
        plt.xlabel('Generation')
        plt.ylabel('Percentage (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if generations:
            plt.xticks(generations)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved punishment patterns plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_reputation_evolution(self):
        """
        Analyze how reputation changes across generations.
        """
        # Extract reputation data
        agent_reputation = defaultdict(list)
        
        for agent in self.agents_data:
            name = agent.get('agent_name')
            gen = agent.get('current_generation')
            reputation = agent.get('reputation')
            
            if all(v is not None for v in [name, gen]) and reputation is not False and reputation is not None:
                agent_reputation[(name, gen)].append(reputation)
        
        # Calculate average reputation per agent per generation
        avg_reputation = {}
        for (name, gen), reputations in agent_reputation.items():
            avg_reputation[(name, gen)] = np.mean(reputations)
        
        # Group by generation
        gen_reputation = defaultdict(list)
        for (name, gen), rep in avg_reputation.items():
            gen_reputation[gen].append(rep)
        
        # Calculate average and variance per generation
        gen_avg_reputation = {}
        gen_var_reputation = {}
        
        for gen, reputations in gen_reputation.items():
            if reputations:
                gen_avg_reputation[gen] = np.mean(reputations)
                gen_var_reputation[gen] = np.var(reputations)
            else:
                gen_avg_reputation[gen] = 0
                gen_var_reputation[gen] = 0
        
        return {
            'avg_reputation': gen_avg_reputation,
            'var_reputation': gen_var_reputation
        }
    
    def plot_reputation_evolution(self, output_path=None):
        """Plot the evolution of reputation across generations."""
        reputation_data = self.analyze_reputation_evolution()
        
        generations = sorted(reputation_data['avg_reputation'].keys())
        
        # Plot average reputation
        plt.figure(figsize=(10, 6))
        
        plt.errorbar(
            generations,
            [reputation_data['avg_reputation'][gen] for gen in generations],
            yerr=[np.sqrt(reputation_data['var_reputation'][gen]) for gen in generations],
            marker='o', linestyle='-', color='purple', capsize=5
        )
        
        plt.title(f'Average Agent Reputation by Generation ({self.llm_type})')
        plt.xlabel('Generation')
        plt.ylabel('Average Reputation')
        plt.grid(True, alpha=0.3)
        
        if generations:
            plt.xticks(generations)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved reputation evolution plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_strategy_network(self, output_path=None):
        """
        Visualize the phylogenetic tree of strategies over generations.
        
        Note: This requires networkx and pydot packages which are not included
        in the current code dependencies.
        """
        try:
            import networkx as nx
            from networkx.drawing.nx_pydot import graphviz_layout
        except ImportError:
            print("This function requires networkx and pydot. Install with:")
            print("pip install networkx pydot")
            return
        
        # This would be a more complex function that requires access to
        # strategy inheritance patterns not directly available in the JSON data.
        # For now, we'll implement a simplified version.
        print("Strategy network visualization requires additional data on strategy inheritance.")
        
    def generate_comprehensive_report(self, output_dir="donor_game_analysis"):
        """Generate a comprehensive report with all analysis visualizations."""
        import datetime
        
        # File name base from original file - use only first part to keep it shorter
        full_base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        
        # Create a shortened version of the base name (first 20 chars)
        short_base_name = full_base_name[:20]
        if len(full_base_name) > 20:
            short_base_name += "_etc"
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{short_base_name}_{timestamp}"
        
        # Create a subfolder specifically for this analysis
        analysis_subfolder = os.path.join(output_dir, folder_name)
        os.makedirs(analysis_subfolder, exist_ok=True)
        
        # Use simple names for output files
        self.plot_average_resources_over_generations(
            os.path.join(analysis_subfolder, "resources.png"))
        
        self.plot_donation_percentages_over_generations(
            os.path.join(analysis_subfolder, "donations.png"))
        
        self.plot_strategy_complexity_over_generations(
            os.path.join(analysis_subfolder, "complexity.png"))
        
        self.create_agent_donation_heatmap(
            os.path.join(analysis_subfolder, "heatmap.png"))
        
        self.plot_strategy_patterns_over_generations(
            os.path.join(analysis_subfolder, "patterns.png"))
        
        self.plot_punishment_over_generations(
            os.path.join(analysis_subfolder, "punishment.png"))
        
        self.plot_reputation_evolution(
            os.path.join(analysis_subfolder, "reputation.png"))
        
        # Create a summary text file with information about this simulation
        with open(os.path.join(analysis_subfolder, "info.txt"), 'w') as f:
            f.write(f"Analysis of: {full_base_name}\n")
            f.write(f"Analysis performed: {timestamp}\n")
            f.write(f"LLM Type: {self.llm_type}\n")
            f.write(f"Number of Generations: {self.num_generations}\n")
            f.write(f"Hyperparameters: {json.dumps(self.hyperparameters, indent=2)}\n")
        
        print(f"Comprehensive report generated in {analysis_subfolder}")


def analyze_multiple_simulations(file_paths, output_dir="donor_game_comparative"):
    """
    Analyze and compare multiple simulation results.
    
    Args:
        file_paths: List of paths to simulation JSON files
        output_dir: Directory to save comparative analysis plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzers for each file
    analyzers = [DonorGameAnalyzer(path) for path in file_paths]
    
    # Get LLM types for labels
    llm_types = [analyzer.llm_type for analyzer in analyzers]
    
    # Compare average resources across simulations
    plt.figure(figsize=(12, 7))
    
    for analyzer, label in zip(analyzers, llm_types):
        avg_resources = analyzer.get_average_resources_by_generation()
        generations = sorted(avg_resources.keys())
        resources_values = [avg_resources[gen] for gen in generations]
        
        plt.plot(generations, resources_values, marker='o', linestyle='-', label=label)
    
    plt.title('Comparative Analysis: Average Resources by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Final Resources')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "comparative_resources.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compare strategy complexity
    plt.figure(figsize=(12, 7))
    
    for analyzer, label in zip(analyzers, llm_types):
        complexity = analyzer.analyze_strategy_complexity()
        generations = sorted(complexity.keys())
        complexity_values = [complexity[gen] for gen in generations]
        
        plt.plot(generations, complexity_values, marker='o', linestyle='-', label=label)
    
    plt.title('Comparative Analysis: Strategy Complexity by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Word Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "comparative_complexity.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compare initial donation percentages
    plt.figure(figsize=(12, 7))
    
    for analyzer, label in zip(analyzers, llm_types):
        pattern_results = analyzer.extract_key_strategy_patterns()
        generations = sorted(pattern_results.keys())
        
        avg_donations = []
        for gen in generations:
            donations = pattern_results[gen]['initial_donations']
            avg_donations.append(np.mean(donations) if donations else 0)
        
        plt.plot(generations, avg_donations, marker='o', linestyle='-', label=label)
    
    plt.title('Comparative Analysis: Initial Donation Percentage by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Initial Donation Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "comparative_initial_donations.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparative analysis generated in {output_dir}")


# Example usage
if __name__ == "__main__":
    # Path to your JSON data file
    data_file = "simulation_results/Donor_Game_test-mode_coopGain_2punLoss_2_three_last_tracesgen2_agents6_top_20250403_193213.json"
    
    # Initialize the analyzer
    analyzer = DonorGameAnalyzer(data_file)
    
    # Generate a comprehensive report
    analyzer.generate_comprehensive_report()
    
    # For comparing multiple simulations with different models
    # analyze_multiple_simulations([
    #     "simulation_results/claude_simulation.json",
    #     "simulation_results/gemini_simulation.json",
    #     "simulation_results/gpt4o_simulation.json"
    # ])