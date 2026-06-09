import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the data
df = pd.read_csv('name_game_LLM_feedback_MN_20260507_134935.csv')

print("="*80)
print("NAME GAME COORDINATION ANALYSIS")
print("="*80)
print(f"\nDataset: {len(df)} total rounds across {df['trial'].nunique()} trials")
print(f"Partner conditions: {df['partner_condition'].unique()}")
print(f"Rounds per trial: {df['round'].max()}")

# ============================================================================
# 1. OVERALL MATCHING RATES
# ============================================================================
print("\n" + "="*80)
print("1. OVERALL MATCHING RATES")
print("="*80)

overall_match_rate = df['matched'].mean()
print(f"\nOverall matching rate: {overall_match_rate:.2%}")

# Matching rate by partner condition
condition_matches = df.groupby('partner_condition')['matched'].agg(['mean', 'sum', 'count'])
condition_matches.columns = ['match_rate', 'matches', 'total_rounds']
print("\nMatching rate by partner condition:")
print(condition_matches)

# Matching rate by round
round_matches = df.groupby('round')['matched'].mean()
print("\nMatching rate by round:")
print(round_matches)

# ============================================================================
# 2. LETTER CHOICE DISTRIBUTION
# ============================================================================
print("\n" + "="*80)
print("2. LETTER CHOICE DISTRIBUTION")
print("="*80)

# Get all player choices
player_choices = []
for round_num in range(1, 15):
    round_data = df[df['round'] == round_num]
    for p in ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']:
        player_choices.extend(round_data[p].tolist())

choice_counts = pd.Series(player_choices).value_counts()
print("\nOverall letter distribution:")
print(choice_counts)

# Letter distribution by round phase
print("\nLetter distribution by phase:")
rounds_1_7 = df[df['round'] <= 7]
rounds_8_14 = df[df['round'] >= 8]

choices_1_7 = []
for p in ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']:
    choices_1_7.extend(rounds_1_7[p].tolist())

choices_8_14 = []
for p in ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']:
    choices_8_14.extend(rounds_8_14[p].tolist())

print("\nRounds 1-7 (J=10, others=0):")
print(pd.Series(choices_1_7).value_counts())

print("\nRounds 8-14 (J=10, M=15, N=15, others=0):")
print(pd.Series(choices_8_14).value_counts())

# ============================================================================
# 3. SWITCHING BEHAVIOR AT ROUND 8
# ============================================================================
print("\n" + "="*80)
print("3. SWITCHING BEHAVIOR AT ROUND 8")
print("="*80)

# Analyze what happens in round 8 (first round with M/N option)
round_8_data = df[df['round'] == 8].copy()
print(f"\nRound 8 letter choices across {len(round_8_data)} trials:")

for idx, row in round_8_data.iterrows():
    trial = row['trial']
    condition = row['partner_condition']
    choices = [row[f'p{i}'] for i in range(1, 7)]
    choice_dist = pd.Series(choices).value_counts().to_dict()
    matched = row['matched']

    print(f"  Trial {trial} ({condition}): {choice_dist} - {'MATCHED' if matched else 'NO MATCH'}")

# ============================================================================
# 4. CONVERGENCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. CONVERGENCE ANALYSIS")
print("="*80)

# Track convergence after mismatches
def analyze_convergence(trial_data):
    """Analyze how many rounds it takes to reconverge after a mismatch"""
    rounds_to_converge = []
    in_mismatch = False
    mismatch_start = None

    for idx, row in trial_data.iterrows():
        if not row['matched'] and not in_mismatch:
            in_mismatch = True
            mismatch_start = row['round']
        elif row['matched'] and in_mismatch:
            rounds_to_converge.append(row['round'] - mismatch_start)
            in_mismatch = False
            mismatch_start = None

    return rounds_to_converge

convergence_times = []
for trial in df['trial'].unique():
    trial_data = df[df['trial'] == trial].sort_values('round')
    conv_times = analyze_convergence(trial_data)
    convergence_times.extend(conv_times)

if convergence_times:
    print(f"\nAverage rounds to reconverge after mismatch: {np.mean(convergence_times):.2f}")
    print(f"Convergence time distribution: {pd.Series(convergence_times).value_counts().sort_index()}")
else:
    print("\nNo convergence events detected (all rounds matched or never reconverged)")

# ============================================================================
# 5. POINTS EARNED ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. POINTS EARNED ANALYSIS")
print("="*80)

# Final scores by trial
final_scores = df.groupby('trial').last()[['partner_condition', 'p1_total', 'p2_total',
                                            'p3_total', 'p4_total', 'p5_total', 'p6_total']]

avg_final_score = final_scores[['p1_total', 'p2_total', 'p3_total',
                                 'p4_total', 'p5_total', 'p6_total']].mean(axis=1)

print("\nFinal scores by trial:")
for trial in final_scores.index:
    condition = final_scores.loc[trial, 'partner_condition']
    score = avg_final_score.loc[trial]
    print(f"  Trial {trial} ({condition}): {score:.0f} points")

# Average by condition
print("\nAverage final score by partner condition:")
for condition in df['partner_condition'].unique():
    condition_trials = final_scores[final_scores['partner_condition'] == condition]
    avg_score = condition_trials[['p1_total', 'p2_total', 'p3_total',
                                   'p4_total', 'p5_total', 'p6_total']].mean().mean()
    print(f"  {condition}: {avg_score:.0f} points")

# Maximum possible score
max_score_J_only = 14 * 10  # 140 points if choosing J all 14 rounds
max_score_optimal = 7 * 10 + 7 * 15  # 175 points if J for 1-7, M/N for 8-14

print(f"\nMaximum possible scores:")
print(f"  J only (all 14 rounds): {max_score_J_only} points")
print(f"  Optimal (J in 1-7, M/N in 8-14): {max_score_optimal} points")

# ============================================================================
# 6. REASONING ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("6. REASONING PATTERNS")
print("="*80)

# Count reasoning entries
reasoning_cols = [f'p{i}_reasoning' for i in range(1, 7)]
total_reasoning = 0
for col in reasoning_cols:
    total_reasoning += df[col].notna().sum()

print(f"\nTotal reasoning entries: {total_reasoning} out of {len(df) * 6} possible")
print(f"Reasoning rate: {total_reasoning / (len(df) * 6):.2%}")

# Sample some reasoning from round 8
print("\nSample reasoning from Round 8 (decision point for M/N):")
round_8 = df[df['round'] == 8]
for idx, row in round_8.head(2).iterrows():
    print(f"\nTrial {row['trial']} ({row['partner_condition']}):")
    for i in range(1, 7):
        reasoning = row[f'p{i}_reasoning']
        if pd.notna(reasoning) and reasoning.strip():
            print(f"  Player {i} chose {row[f'p{i}']}: {reasoning[:150]}...")
            break

# ============================================================================
# 7. VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("7. GENERATING VISUALIZATIONS")
print("="*80)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Matching rate by round
ax1 = axes[0, 0]
round_matches.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Matching Rate by Round', fontsize=14, fontweight='bold')
ax1.set_xlabel('Round')
ax1.set_ylabel('Match Rate')
ax1.axvline(6.5, color='red', linestyle='--', alpha=0.5, label='Phase change (R7→R8)')
ax1.legend()
ax1.set_ylim([0, 1.1])

# Plot 2: Letter choices over time
ax2 = axes[0, 1]
letter_by_round = []
for round_num in range(1, 15):
    round_data = df[df['round'] == round_num]
    choices = []
    for p in ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']:
        choices.extend(round_data[p].tolist())

    j_pct = (pd.Series(choices) == 'J').mean()
    m_pct = (pd.Series(choices) == 'M').mean()
    n_pct = (pd.Series(choices) == 'N').mean()

    letter_by_round.append({'round': round_num, 'J': j_pct, 'M': m_pct, 'N': n_pct})

letter_df = pd.DataFrame(letter_by_round)
letter_df.plot(x='round', y=['J', 'M', 'N'], ax=ax2, marker='o')
ax2.set_title('Letter Choice Proportions Over Rounds', fontsize=14, fontweight='bold')
ax2.set_xlabel('Round')
ax2.set_ylabel('Proportion')
ax2.axvline(7.5, color='red', linestyle='--', alpha=0.5)
ax2.legend(['J (10 pts)', 'M (15 pts in R8+)', 'N (15 pts in R8+)'])

# Plot 3: Matching rate by condition
ax3 = axes[1, 0]
condition_matches['match_rate'].plot(kind='bar', ax=ax3, color='coral')
ax3.set_title('Matching Rate by Partner Condition', fontsize=14, fontweight='bold')
ax3.set_xlabel('Partner Condition')
ax3.set_ylabel('Match Rate')
ax3.set_ylim([0, 1.1])
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')

# Plot 4: Final scores by condition
ax4 = axes[1, 1]
score_by_condition = []
for condition in df['partner_condition'].unique():
    condition_trials = final_scores[final_scores['partner_condition'] == condition]
    scores = condition_trials[['p1_total', 'p2_total', 'p3_total',
                               'p4_total', 'p5_total', 'p6_total']].values.flatten()
    for score in scores:
        score_by_condition.append({'condition': condition, 'score': score})

score_df = pd.DataFrame(score_by_condition)
sns.boxplot(data=score_df, x='condition', y='score', ax=ax4)
ax4.set_title('Final Score Distribution by Condition', fontsize=14, fontweight='bold')
ax4.set_xlabel('Partner Condition')
ax4.set_ylabel('Final Score')
ax4.axhline(max_score_optimal, color='green', linestyle='--', alpha=0.5, label='Optimal (175)')
ax4.axhline(max_score_J_only, color='orange', linestyle='--', alpha=0.5, label='J-only (140)')
ax4.legend()
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('name_game_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved visualization to: name_game_analysis.png")

# ============================================================================
# 8. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("8. SUMMARY STATISTICS")
print("="*80)

summary_stats = {
    'Total trials': df['trial'].nunique(),
    'Total rounds': len(df),
    'Overall match rate': f"{df['matched'].mean():.2%}",
    'Match rate rounds 1-7': f"{rounds_1_7['matched'].mean():.2%}",
    'Match rate rounds 8-14': f"{rounds_8_14['matched'].mean():.2%}",
    'Most common letter (overall)': choice_counts.index[0],
    'Average final score': f"{avg_final_score.mean():.1f}",
    'Optimal score achieved': any(avg_final_score >= max_score_optimal)
}

print("\nSummary:")
for key, value in summary_stats.items():
    print(f"  {key}: {value}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
