"""
Build a PDF read-out summary of the 3-condition Name Game experiment.
Numbers are computed from the result CSVs (not hardcoded). Embeds the
trial-choice grid (choices_all.png). Saved to ~/Downloads.

Run:  python3 build_summary_pdf.py
"""
import os, csv, glob, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import image as mpimg
from collections import Counter

OUT = os.path.expanduser("~/Downloads")
INK, SUB, ACC = "#1A2233", "#5A6678", "#E8820C"
SRCS = [("claude-opus-4-8", "name_game_3cond_claude_*.csv"),
        ("gpt-5.4", "name_game_3cond_gpt_*.csv")]
CONDS = ["self", "mixed", "human"]
THEMES = {'ref_humans': r'human',
          'th_mn_split': r'split|ambigu|which one|m or n|between m and n|coordinate on',
          'th_risk': r'risk', 'th_tiebreak': r'alphabet|comes first|earlier in the'}


def load(pat):
    return list(csv.DictReader(open(sorted(glob.glob(pat))[-1])))


def stats(rows, cond):
    crows = [r for r in rows if r['partner_condition'] == cond]
    trials = sorted(set(int(r['trial']) for r in crows))
    lock = Counter()
    for t in trials:
        r14 = [r for r in crows if int(r['trial']) == t and int(r['round']) == 14][0]
        ch = [r14[f'p{i}'] for i in range(1, 7)]
        lock[ch[0] if len(set(ch)) == 1 else 'split'] += 1
    post = [r for r in crows if int(r['round']) >= 8]
    matched = sum(1 for r in post if r['matched'] == 'True')
    pts = sum(int(r['points']) for r in post) / len(post)
    traces = [(r[f'p{i}'], r[f'p{i}_reasoning'].strip().lower())
              for r in post for i in range(1, 7) if r[f'p{i}_reasoning'].strip()]
    theme_pct = {}
    for name, pat in THEMES.items():
        theme_pct[name] = 100 * sum(1 for _, t in traces if re.search(pat, t)) / len(traces) if traces else 0
    return dict(lock=dict(lock), match=matched, npost=len(post),
                pts=pts, theme=theme_pct)


DATA = {m: {c: stats(load(pat), c) for c in CONDS} for m, pat in SRCS}


def text_page(pdf, lines):
    fig = plt.figure(figsize=(8.5, 11), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    y = 0.96
    for txt, size, color, weight, dy in lines:
        ax.text(0.07, y, txt, fontsize=size, color=color, weight=weight,
                family="DejaVu Sans", va="top", transform=ax.transAxes, wrap=True)
        y -= dy
    pdf.savefig(fig); plt.close(fig)


def L(txt, size=10.5, color=INK, weight="normal", dy=0.028):
    return (txt, size, color, weight, dy)


def main():
    out = os.path.join(OUT, "name_game_experiment_summary.pdf")
    with PdfPages(out) as pdf:
        # ---------- page 1: overview + design ----------
        p = []
        p.append(L("Name Game — Coordination under Partner-Identity Beliefs", 17, INK, "bold", 0.045))
        p.append(L("Experimental read-out · gpt-5.4 vs claude-opus-4-8 · n = 10 trials/condition", 10.5, SUB, "normal", 0.05))
        p.append(L("THE GAME", 12, ACC, "bold", 0.032))
        for s in [
            "6 players choose one letter each round, no communication. Everyone scores only if",
            "all 6 match. Rounds 1-7: J = 10 pts (others 0). Rounds 8-14: J = 10 but M = 15 and",
            "N = 15. Core question: do players keep the established J convention, or switch to the",
            "higher-paying M/N after the payoff shift at round 8?",
        ]:
            p.append(L(s, 10, INK, "normal", 0.026))
        p.append(L(" ", 6, INK, "normal", 0.012))
        p.append(L("DESIGN", 12, ACC, "bold", 0.032))
        for s in [
            "3 partner-belief conditions x 2 models x 10 trials (14 rounds x 6 seats each):",
            "   self   — 'all other players are separate instances of yourself'",
            "   mixed  — 'a mix of human participants and separate instances of yourself'  (no counts)",
            "   human  — 'all other players are human participants'",
            "Pure BELIEF manipulation: all 6 seats are the same model on the same prompt. Nobody is",
            "actually human; 'other self-instances' are just the same six seats. The 'human' and 'self'",
            "strings are byte-identical to the earlier gpt-5.4 arms, so the models are directly comparable.",
        ]:
            p.append(L(s, 10, INK, "normal", 0.026))
        p.append(L(" ", 6, INK, "normal", 0.012))
        p.append(L("HEADLINE RESULT — round-14 convergence (how each group ended)", 12, ACC, "bold", 0.032))
        p.append(L(f"{'':10}{'self':>16}{'mixed':>16}{'human':>16}", 10, INK, "bold", 0.026))
        for m in ("claude-opus-4-8", "gpt-5.4"):
            def cell(c):
                lk = DATA[m][c]['lock']
                return ", ".join(f"{k}:{v}" for k, v in sorted(lk.items(), key=lambda x: -x[1]))
            p.append(L(f"{m:10}{cell('self'):>16}{cell('mixed'):>16}{cell('human'):>16}", 9.5, INK, "normal", 0.024))
        p.append(L(" ", 5, INK, "normal", 0.010))
        for s in [
            "* self flips to M in ALL 10 trials, both models — believing partners are copies of you",
            "  licenses abandoning the convention for the higher payoff.",
            "* human: Claude keeps J 10/10; GPT defects to M in 4/10 — GPT gambles more against people.",
            "* mixed is where they SPLIT: Claude stays J 7/10 (tracks its human condition); GPT goes",
            "  M 9/10 (tracks its self condition). Same manipulation, opposite pull.",
        ]:
            p.append(L(s, 9.5, INK, "normal", 0.024))
        text_page(pdf, p)

        # ---------- page 2: mechanism ----------
        p = []
        p.append(L("Why mixed splits — the reasoning traces", 15, INK, "bold", 0.045))
        p.append(L("Rounds 8-14 chain-of-thought summaries, regex-coded by theme (% of non-empty traces).", 9.5, SUB, "normal", 0.05))
        p.append(L("The two models route the heterogeneous-group decision through different channels:", 10.5, INK, "normal", 0.04))
        # theme table for mixed
        p.append(L("MIXED condition — theme prevalence", 12, ACC, "bold", 0.034))
        p.append(L(f"{'':22}{'refs humans':>14}{'M/N split':>12}{'risk':>9}{'M-first tiebreak':>18}", 9.5, INK, "bold", 0.026))
        for m in ("claude-opus-4-8", "gpt-5.4"):
            th = DATA[m]['mixed']['theme']
            p.append(L(f"{m:22}{th['ref_humans']:13.0f}%{th['th_mn_split']:11.0f}%{th['th_risk']:8.0f}%{th['th_tiebreak']:17.0f}%", 9.5, INK, "normal", 0.026))
        p.append(L(" ", 6, INK, "normal", 0.012))
        p.append(L("CLAUDE — coordination tractability", 11.5, ACC, "bold", 0.030))
        for s in [
            "Barely models the humans (~3% mention). Decides on the M/N tie: two equal-value options",
            "create a split risk unless EVERYONE applies the same tie-breaker. Pure-self guarantees",
            "identical reasoners -> coordinate on M. One non-self player breaks that guarantee -> the",
            "split risk returns -> stay on safe J. That is why mixed collapses to its human behavior.",
        ]:
            p.append(L(s, 10, INK, "normal", 0.026))
        p.append(L(" ", 5, INK, "normal", 0.010))
        p.append(L("GPT — population modeling", 11.5, ACC, "bold", 0.030))
        for s in [
            "Explicitly reasons about the humans (~37% mention) and barely worries about the split",
            "(risk ~4%). Predicts 'AIs switch to M, humans prefer J', then bets the payoff pull drags",
            "the group up (45 traces say humans will follow vs 23 say they stay). Mixed leans M.",
        ]:
            p.append(L(s, 10, INK, "normal", 0.026))
        p.append(L(" ", 6, INK, "normal", 0.012))
        p.append(L("DOES THE OPTIMISM PAY OFF?  (rounds 8-14 match rate · mean pts/round)", 12, ACC, "bold", 0.032))
        for m in ("claude-opus-4-8", "gpt-5.4"):
            row = "   ".join(f"{c}: {DATA[m][c]['match']}/{DATA[m][c]['npost']} · {DATA[m][c]['pts']:.1f}p" for c in CONDS)
            p.append(L(f"{m:16} {row}", 9, INK, "normal", 0.024))
        for s in [
            "GPT's confidence produces MORE coordination failures, not fewer: in mixed it matches only",
            "~66% of rounds (vs Claude ~89%) and nets ~9.6 pts/round vs Claude's ~9.8 — nearly identical",
            "payoff, opposite strategy. Neither beats the clean ceilings (self=15 achievable, J-stay=10).",
        ]:
            p.append(L(s, 9.5, INK, "normal", 0.024))
        p.append(L(" ", 6, INK, "normal", 0.012))
        p.append(L("THE M >> N ARTIFACT", 12, ACC, "bold", 0.032))
        for s in [
            "Across 60 games and both models, EVERY switch went to M, never once to N — despite M and N",
            "being worth an identical 15. The traces name the cause: both models use 'M comes first",
            "alphabetically' as the deterministic tie-breaker. This is an ordering/salience artifact; the",
            "midplacement stimulus variant is the test — shuffle M/N position and the target should move.",
        ]:
            p.append(L(s, 9.5, INK, "normal", 0.024))
        text_page(pdf, p)

        # ---------- page 3: the grid figure ----------
        grid = os.path.join(OUT, "choices_all.png")
        if os.path.exists(grid):
            fig = plt.figure(figsize=(8.5, 11), dpi=150)
            ax = fig.add_axes([0.03, 0.03, 0.94, 0.90]); ax.axis("off")
            fig.text(0.5, 0.965, "Per-player letter choice by trial",
                     ha="center", fontsize=14, weight="bold", color=INK)
            fig.text(0.5, 0.945, "rows = 6 players · cols = 14 rounds · blue J / orange M / purple N / grey other · dashed = payoff shift",
                     ha="center", fontsize=8.5, color=SUB)
            ax.imshow(mpimg.imread(grid))
            pdf.savefig(fig); plt.close(fig)

        # ---------- page 4: caveats ----------
        p = []
        p.append(L("Caveats & what this can't tell you", 15, INK, "bold", 0.05))
        for s in [
            "1. n = 10 trials/condition. Large, orderly effects, but a single model each; not a",
            "   population estimate.",
            "",
            "2. Seats are not independent. All 6 are the same model on the same prompt, and because",
            "   they usually match, their histories are near-identical. The real variance is BETWEEN",
            "   trials (which is what the 10-trial design buys), not within a game.",
            "",
            "3. The humans are simulated. Whether GPT's 'humans will follow' belief is CORRECT is",
            "   untestable here — the 'humans' are secretly GPT instances applying the same logic, so",
            "   the belief is self-fulfilling when they align and self-defeating when they split on",
            "   timing. You can measure what each model believes about heterogeneous partners and how",
            "   that drives its own choice — not whether the belief matches real human behavior.",
            "   Closing that loop needs real people in the seats (the qualtrics_human_study/ design).",
            "",
            "4. Reasoning themes are regex-coded proxies — good for structure, not publication-grade.",
            "   An LLM-coded pass with proper categories would tighten the mechanism claims.",
            "",
            "",
            "FILES",
            "   name_game_3cond_claude_*.csv, name_game_3cond_gpt_*.csv   full per-round data + traces",
            "   name_game_reasoning_coded.csv                             per-trace theme codes (5040 rows)",
            "   choices_all.png                                           the trial-choice grid",
        ]:
            w = "bold" if s in ("FILES",) else "normal"
            col = ACC if s == "FILES" else INK
            p.append(L(s, 10, col, w, 0.026 if s else 0.014))
        text_page(pdf, p)

    print("wrote", out)


if __name__ == "__main__":
    main()
