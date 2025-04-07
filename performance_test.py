# performance_test.py
import os
import re
import glob
from statistics import mean, stdev
import time

def get_most_recent_log():
    """Find the most recently created log file that matches our pattern"""
    log_files = glob.glob("pacman_*.log")
    if not log_files:
        return None
    # Sort by creation time (most recent first)
    return max(log_files, key=os.path.getctime)

def extract_statistics(log_file):
    """Extract all statistics from a log file"""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return None
        
    stats = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Extract average score
        avg_score_match = re.search(r'Average Score: ([0-9.-]+)', content)
        if avg_score_match:
            stats['average_score'] = float(avg_score_match.group(1))
        
        # Extract win rate
        win_rate_match = re.search(r'Win Rate:\s+(\d+)/(\d+) \(([\d.]+)\)', content)
        if win_rate_match:
            stats['wins'] = int(win_rate_match.group(1))
            stats['total_games'] = int(win_rate_match.group(2))
            stats['win_rate'] = float(win_rate_match.group(3))
        
        # Extract all individual scores
        scores_match = re.search(r'Scores:\s+([\d., -]+)', content)
        if scores_match:
            scores_str = scores_match.group(1)
            try:
                stats['scores'] = [float(s.strip()) for s in scores_str.split(',')]
            except:
                stats['scores'] = []
        
        if not stats:
            print(f"No statistics found in {log_file}")
            return None
            
        return stats

def run_tests(agents=None, games_per_agent=5, ghosts=2, ghost_type='RandomGhost', quiet=True, depth=2, eval_fn=None):
    """Run tests for the specified agents and return statistics"""
    
    if agents is None:
        # Default to testing only ReflexAgent
        agents = [
            'ReflexAgent'
        ]
    
    results = {}
    
    for agent in agents:
        print(f"\nTesting {agent} against {ghost_type} (count: {ghosts})...")
        agent_results = []
        
        for i in range(games_per_agent):
            print(f"  Running game {i+1}/{games_per_agent}...")
            
            # Construct the command based on the agent
            cmd = f"python3 pacman.py -p {agent} -k {ghosts} -g {ghost_type}"

            print(cmd)
            
            # Add depth parameter for advanced agents
            if agent in ['MinimaxAgent', 'AlphaBetaAgent', 'ExpectimaxAgent']:
                cmd += f" -a depth={depth}"
            
            # Add evaluation function if specified
            if eval_fn and agent in ['MinimaxAgent', 'AlphaBetaAgent', 'ExpectimaxAgent']:
                cmd += f",evalFn={eval_fn}"
                
            if quiet:
                cmd += " -q"
            
            # Run the game
            os.system(cmd)
            
            # Allow time for file to be written
            time.sleep(0.5)
            
            # Get the most recent log file
            log_file = get_most_recent_log()
            if log_file:
                stats = extract_statistics(log_file)
                if stats:
                    agent_results.append(stats)
                    print(f"  Score: {stats['average_score']:.1f}, Win Rate: {stats.get('win_rate', 0):.2f}")
                else:
                    print("  No statistics found in log file")
            else:
                print("  No log file found")
        
        if agent_results:
            # Aggregate results for this agent
            avg_scores = [r['average_score'] for r in agent_results if 'average_score' in r]
            win_rates = [r['win_rate'] for r in agent_results if 'win_rate' in r]
            
            results[agent] = {
                'avg_score': mean(avg_scores) if avg_scores else 0,
                'win_rate': mean(win_rates) if win_rates else 0,
                'std_dev_score': stdev(avg_scores) if len(avg_scores) > 1 else 0,
                'games': len(agent_results)
            }
    
    return results

def print_results(results):
    """Print the results in a nicely formatted table"""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*70)
    print("PACMAN AGENT PERFORMANCE COMPARISON")
    print("="*70)
    print(f"{'Agent':<20} {'Avg Score':<15} {'Win Rate':<15} {'Std Dev':<15}")
    print("-"*70)
    
    # Sort agents by average score (descending)
    sorted_agents = sorted(results.keys(), key=lambda a: results[a]['avg_score'], reverse=True)
    
    for agent in sorted_agents:
        r = results[agent]
        print(f"{agent:<20} {r['avg_score']:>12.2f}   {r['win_rate']:>12.2f}   {r['std_dev_score']:>12.2f}")
    
    print("="*70)

if __name__ == "__main__":
    # 測試 ReflexAgent vs RandomGhost
    print("=== 測試 ReflexAgent vs RandomGhost ===")
    random_results = run_tests(
        agents=['ReflexAgent'], 
        games_per_agent=3,
        ghosts=2,
        ghost_type='RandomGhost'
    )
    print_results(random_results)
    
    # 測試 ReflexAgent vs SmarterRandomGhost
    print("\n=== 測試 ReflexAgent vs SmarterRandomGhost ===")
    smarter_random_results = run_tests(
        agents=['ReflexAgent'], 
        games_per_agent=3,
        ghosts=2,
        ghost_type='SmarterRandomGhost'
    )
    print_results(smarter_random_results)
    
    # 測試 ReflexAgent vs DirectionalGhost
    print("\n=== 測試 ReflexAgent vs DirectionalGhost ===")
    directional_results = run_tests(
        agents=['ReflexAgent'], 
        games_per_agent=3,
        ghosts=2,
        ghost_type='DirectionalGhost'
    )
    print_results(directional_results)