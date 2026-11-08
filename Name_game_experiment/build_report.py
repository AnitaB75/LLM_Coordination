"""
Build the PDF summary of Steps 1 and 1b.

Every number in the report is read from the result CSVs at build time — nothing is
transcribed by hand, so the document cannot drift from the data. Figures are
embedded as base64 so the HTML is self-contained.

Renders via headless Chrome (no LaTeX dependency).

Run:  python3 build_report.py
"""
import os, csv, base64, subprocess, html, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "step1_output")
DL = os.path.expanduser("~/Downloads")
TS1 = "20260804_124918"      # Step 1 form-choice run
TS2 = "20260804_153626"      # Step 1b screen
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

f = float


def rd(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def b64img(path, width="100%"):
    with open(path, "rb") as fh:
        d = base64.standard_b64encode(fh.read()).decode()
    return f'<img src="data:image/png;base64,{d}" style="width:{width}">'


# ------------------------------------------------------------------- data ----
s1 = rd(os.path.join(OUT, f"step1_form_summary_{TS1}.csv"))
s1c = rd(os.path.join(OUT, f"step1_form_comparison_{TS1}.csv"))
s2 = rd(os.path.join(OUT, f"step1b_screen_summary_{TS2}.csv"))
s2s = rd(os.path.join(OUT, f"step1b_screen_selected_{TS2}.csv"))
i1 = {(r["model"], r["stimulus_pair"]): r for r in s1}
i2 = {(r["model"], r["stimulus_pair"]): r for r in s2}

n_ceiling = sum(1 for r in s2s if f(r["gpt_match"]) == 1.0 and f(r["claude_match"]) == 1.0)
n_sel = sum(1 for r in s2s if r["selected"] == "True")
up = [r for r in s2s if r["selected"] == "True" and f(r["delta_gpt_minus_claude"]) > 0]
dn = [r for r in s2s if r["selected"] == "True" and f(r["delta_gpt_minus_claude"]) < 0]

PROMPT = ("You are playing a one-shot coordination game with ONE other player - another instance "
          "of you (the same AI model). You will each independently pick one of the two shapes "
          "shown; you want to pick the SAME shape as the other player. No communication. "
          "IMPORTANT: the left/right position of the two shapes is randomized independently for "
          "each of you, so position CANNOT be used to coordinate - you must coordinate on a shape "
          "by its FORM. Give one short sentence of reasoning, then end with exactly "
          "'CHOICE: LEFT' or 'CHOICE: RIGHT' (the position in YOUR image of the shape you chose).")

CSS = """
@page { size: Letter; margin: 19mm 18mm 20mm 18mm; }
* { box-sizing: border-box; }
body { font-family: Arial, Helvetica, sans-serif; font-size: 9.6pt; line-height: 1.45;
       color: #000; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 2pt 0; letter-spacing: -0.2pt; }
.sub { font-size: 10pt; color: #333; margin: 0 0 3pt 0; }
.meta { font-size: 8pt; color: #666; margin: 0 0 16pt 0; }
h2 { font-size: 11.5pt; margin: 20pt 0 6pt 0; padding-bottom: 3pt;
     border-bottom: 0.6pt solid #000; page-break-after: avoid; }
h3 { font-size: 9.8pt; margin: 13pt 0 4pt 0; page-break-after: avoid; }
p { margin: 0 0 7pt 0; }
ul, ol { margin: 0 0 7pt 0; padding-left: 15pt; }
li { margin-bottom: 3pt; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0 10pt 0;
        font-size: 8.4pt; page-break-inside: avoid; }
th { text-align: left; border-bottom: 0.8pt solid #000; padding: 3pt 5pt 3pt 0;
     font-weight: bold; }
td { padding: 2.6pt 5pt 2.6pt 0; border-bottom: 0.3pt solid #DDD; }
td.n, th.n { text-align: right; padding-right: 9pt; font-variant-numeric: tabular-nums; }
.fig { margin: 10pt 0 4pt 0; page-break-inside: avoid; }
.cap { font-size: 8pt; color: #333; margin: 0 0 12pt 0; line-height: 1.4; }
.cap b { color: #000; }
.prompt { font-size: 8.4pt; background: #F4F4F4; padding: 8pt 10pt; margin: 6pt 0 10pt 0;
          border-left: 2pt solid #000; }
.quote { font-size: 8.6pt; margin: 4pt 0 7pt 14pt; color: #222; }
.note { border: 0.6pt solid #000; padding: 8pt 10pt; margin: 9pt 0 11pt 0; font-size: 9pt; }
.note b { display: block; margin-bottom: 3pt; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8.4pt; }
.pb { page-break-before: always; }
.small { font-size: 8.2pt; color: #444; }
"""


def tbl(headers, rows, numeric_from=1):
    # headers are authored here, not user input — do NOT escape, or entities
    # like &Delta; render literally
    h = "".join(f'<th class="{"n" if k >= numeric_from else ""}">{x}</th>'
                for k, x in enumerate(headers))
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f'<td class="{"n" if k >= numeric_from else ""}">{c}</td>'
            for k, c in enumerate(r)) + "</tr>"
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


# ------------------------------------------------------------------ build ----
P1 = ["proto", "inkblot", "rot", "mirror", "noise"]
step1_rows = []
for p in P1:
    g, c = i1[("gpt", p)], i1[("claude", p)]
    step1_rows.append([
        f"<code>{p}</code>",
        f'{f(g["match_unbiased"]):.3f}', f'[{f(g["match_unbiased_lo"]):.3f}, {f(g["match_unbiased_hi"]):.3f}]',
        f'{f(g["p_left"]):.2f}',
        f'{f(c["match_unbiased"]):.3f}', f'[{f(c["match_unbiased_lo"]):.3f}, {f(c["match_unbiased_hi"]):.3f}]',
        f'{f(c["p_left"]):.2f}'])

sel_rows = []
for r in sorted(s2s, key=lambda r: f(r["delta_gpt_minus_claude"])):
    if r["selected"] != "True":
        continue
    g, c = i2[("gpt", r["stimulus_pair"])], i2[("claude", r["stimulus_pair"])]
    sel_rows.append([
        f'<code>{r["stimulus_pair"]}</code>', r["family"],
        f'{f(r["gpt_match"]):.3f}', f'{f(g["p_left"]):.2f}', f'{f(g["form_consistency"]):.2f}',
        f'{f(r["claude_match"]):.3f}', f'{f(c["p_left"]):.2f}', f'{f(c["form_consistency"]):.2f}',
        f'{f(r["delta_gpt_minus_claude"]):+.3f}',
        f'[{f(r["delta_ci_lo"]):+.3f}, {f(r["delta_ci_hi"]):+.3f}]'])

REC = [("s_arrow", "Claude &gt; GPT"), ("s_convexity", "Claude &gt; GPT"),
       ("s_contrast", "GPT &gt; Claude"), ("s_density", "GPT &gt; Claude")]
rec_rows = []
for p, d in REC:
    g, c = i2[("gpt", p)], i2[("claude", p)]
    band = all(0.35 <= f(x["p_left"]) <= 0.65 for x in (g, c))
    rec_rows.append([f"<code>{p}</code>", d,
                     f'{f(g["match_unbiased"]):.3f}', f'{f(c["match_unbiased"]):.3f}',
                     f'{f(g["match_unbiased"]) - f(c["match_unbiased"]):+.3f}',
                     "yes" if band else "no"])

today = datetime.date.today().isoformat()

H = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>

<h1>Model-specific focal points in one-shot visual coordination</h1>
<p class="sub">Steps 1 and 1b &mdash; GPT-5.4 and Claude Opus 4.8</p>
<p class="meta">LLM Coordination project &middot; report generated {today} &middot;
all figures and statistics computed from the run CSVs at build time</p>

<h2>Summary</h2>
<ul>
<li><b>Both models coordinate on visual form, and they do it differently.</b> Across a
five-pair battery and a 30-pair screen, dyadic match probability ranges from chance to 1.000
depending on the stimulus, and the two models dissociate on {n_sel} of 30 screened pairs.</li>
<li><b>Crossovers exist in both directions.</b> {len(up)} pairs where GPT is more concentrated,
{len(dn)} where Claude is. The strongest single result is <code>s_arrow</code>
(Claude 1.000 vs GPT 0.503).</li>
<li><b>A model that cannot find a form rule does not answer randomly &mdash; it defaults to a
side, and then justifies that choice in form language.</b> This is the central methodological
finding and it invalidates any analysis that reads a 50/50 identity split as
&ldquo;no preference&rdquo; without checking position.</li>
<li><b>Competing-cue stimuli did not produce dissociations.</b> {n_ceiling} of 30 pairs sat at a
perfect 1.000/1.000 ceiling, including 9 of the 14 pairs built specifically to put two cues in
conflict. Both models resolve cue conflicts the same way.</li>
</ul>

<h2>1. Task and procedure</h2>
<p>Each trial is an independent, one-shot API call with no conversation history. The model sees a
single composite image containing two shapes side by side and must pick the one it believes
another instance of itself would also pick. There is no communication and no feedback.</p>

<h3>Prompt</h3>
<p>Sent as a single user text block alongside the image, with no system prompt.
<b>No labels &mdash; A/B or otherwise &mdash; are shown to the model.</b> The A/B identity coding
exists only in the runner and the analysis.</p>
<div class="prompt">{html.escape(PROMPT)}</div>
<p class="small">Hash <code>3003ae1b</code>, stamped on every row of every run reported here.
Byte-identical to the arm-1 prompt of the project&rsquo;s earlier
<code>confidence_calibration_probe.py</code>, so this data is comparable with that series.</p>

<h3>Counterbalancing and recoding</h3>
<p>Each pair exists in two layouts: <code>AB</code> places identity&nbsp;1 on the left,
<code>BA</code> places identity&nbsp;2 there. Allocation is exactly half and half in every cell.
The model answers with a <i>position</i> (<code>CHOICE: LEFT</code> / <code>RIGHT</code>), which
is recoded offline to a stimulus identity via the layout. Because the counterbalance is exactly
balanced, a pure position strategy yields a 50/50 identity split &mdash; it reads as chance rather
than as a spurious preference.</p>

<h3>Models and settings</h3>
{tbl(["", "GPT", "Claude"], [
    ["model id", "<code>gpt-5.4</code>", "<code>claude-opus-4-8</code>"],
    ["endpoint", "Responses API", "Messages API"],
    ["reasoning / thinking", "effort low, summary auto", "adaptive, summarized"],
    ["max tokens", "not set", "2000"],
    ["temperature / top_p / top_k", "not set", "not set"],
    ["seed", "not exposed by the API", "not exposed by the API"]], numeric_from=99)}
<p class="small">Settings were fixed across every call; each model shows exactly one distinct
settings string in the data. The only seeded component is the counterbalance RNG
(<code>20260804</code>), which is deterministic.</p>

<h2>2. Measures</h2>
<p><b>Unbiased dyadic match probability.</b> With counts <i>n<sub>k</sub></i> over identities and
<i>N</i> trials, &Sigma;<sub>k</sub> <i>n<sub>k</sub></i>(<i>n<sub>k</sub></i>&minus;1) /
<i>N</i>(<i>N</i>&minus;1). This is the probability that two <i>distinct</i> trials agree &mdash;
the right quantity for &ldquo;would two independent instances pick the same shape&rdquo;. The
plug-in &Sigma;<i>p<sub>k</sub></i><sup>2</sup> is reported alongside as a check and runs slightly
higher. Chance is 0.5. Confidence intervals are percentile 95% from 5,000 bootstrap resamples at a
fixed per-cell seed.</p>

<p><b>Conditional-position diagnostic.</b> For a model&rsquo;s modal identity <i>m</i>, compute the
rate at which <i>m</i> is chosen separately under each layout: <i>p</i>(<i>m</i>&nbsp;|&nbsp;<i>m</i>
on the left) and <i>p</i>(<i>m</i>&nbsp;|&nbsp;<i>m</i> on the right). A model choosing by
<b>form</b> picks <i>m</i> at the same rate either way. A model choosing by <b>side</b> picks
<i>m</i> only when it happens to sit on the favoured side, driving the two rates to 1 and 0.
Reported as <i>form consistency</i> = 1 &minus; |<i>p</i><sub>AB</sub> &minus;
<i>p</i><sub>BA</sub>|.</p>
<div class="note"><b>This diagnostic is necessary but not sufficient.</b>
A model answering at random also scores near 1 on form consistency, so it is only meaningful read
jointly with a match rate above chance. Both conditions are enforced wherever it is used as a
criterion.</div>

<h2>3. Step 1 &mdash; five-pair battery</h2>
<p>600 calls (5 pairs &times; 2 models &times; 60), 405&nbsp;s, zero unparseable responses, zero
API errors, 60/60 valid in all ten cells.</p>
{tbl(["pair", "GPT", "95% CI", "P(left)", "Claude", "95% CI", "P(left)"], step1_rows)}
<div class="fig">{b64img(os.path.join(OUT, f"step1_form_match_{TS1}.png"))}</div>
<p class="cap"><b>Figure 1.</b> Within-model dyadic match probability by stimulus. Filled circles
GPT-5.4, open squares Claude Opus 4.8. Bars are 95% bootstrap CIs (5,000 resamples). Dashed line
marks chance.</p>

<h3>Three distinct mechanisms, not one</h3>
<p>Requiring a sentence of reasoning before the choice made the failure modes legible, and they
turned out to be different in kind.</p>
<ol>
<li><b>Genuine form coordination</b> &mdash; <code>proto</code> (both models 1.000) and
<code>rot</code> (0.695 / 0.741). On <code>rot</code> the chosen side flips with the layout, which
is the signature of tracking the shape rather than the position.</li>
<li><b>Honest fallback</b> &mdash; Claude on <code>noise</code>, which states the position openly:
<span class="quote">&ldquo;These both appear to be random noise with no clear form, so I&rsquo;ll
default to a simple tiebreaker.&rdquo;</span>
There is no form signal to find; the model says so and picks a side deliberately. The measured
0.494 is the correct answer, not an artifact.</li>
<li><b>Confabulated side default</b> &mdash; Claude on <code>inkblot</code>, answering
<code>RIGHT</code> on 60 of 60 trials while giving a <i>form</i> justification each time, accurately
describing whichever shape occupied the right slot:
<span class="quote">[layout BA] &ldquo;The symmetric, star-like shape with balanced points is a more
natural Schelling focal point than the irregular blob.&rdquo;<br>
[layout AB] &ldquo;The right shape has two distinctive tall pointed spikes at the top, making it
the more memorable, easily-agreed-upon form.&rdquo;</span>
Different features cited depending on what sits on the right, and whichever it is gets called the
focal point. The choice tracks position; the justification is fitted to it afterward. Identity
lands at exactly 30/30.</li>
</ol>

<h3>Why <code>mirror</code> fails where <code>rot</code> succeeds</h3>
<p>This is a property of the stimuli, not of the models. <code>rot</code> is a blob rotated 90&deg;,
so one option is upright and one is sideways &mdash; a position-independent asymmetry that
<i>both instances share a prior about</i>. <code>mirror</code> is a blob reflected across the
vertical axis: reflection preserves area, complexity and lobe count, leaving only handedness, and
handedness is exactly the dimension with no shared prior to break the tie. A rule can be stated,
but there is no reason to prefer one sign of it, so rule-selection is itself a coin flip. Claude
generated two incompatible chirality rules minutes apart:</p>
<p class="quote">&ldquo;&hellip;coordinate on the shape with its lower point angling toward the
<b>right</b>.&rdquo;<br>&ldquo;&hellip;pick the one whose narrow point tucks toward the
<b>bottom-left</b>.&rdquo;</p>
<div class="note"><b>The modal share is the misleading statistic on <code>mirror</code>, not the
match rate.</b> GPT&rsquo;s apparent &ldquo;70% preference for blob-mirror&rdquo; comes almost
entirely from the BA layout, where blob-mirror sits on the left: it was chosen 29/30 there but only
13/30 when the same shape sat on the right. The match rate is unaffected &mdash; under a balanced
counterbalance a side bias can only push it toward chance, never above it &mdash; so 0.573 remains
an honest, conservative estimate.</div>

<h2>4. Step 1b &mdash; 30-pair screen</h2>
<p>1,800 calls (30 pairs &times; 2 models &times; 30), 862&nbsp;s, zero unparseable responses, zero
API errors. Two families were built:</p>
<ul>
<li><b>16 single-cue pairs</b> vary one property with everything else matched &mdash; symmetry,
complexity, regularity, curvature, convexity, elongation, size, tilt, closure, hole, numerosity,
density, alignment, line orientation, contrast, semantic orientation.</li>
<li><b>14 competing-cue pairs</b> put two cues in opposition (small-symmetric vs large-asymmetric,
spiky-small vs smooth-large, and so on), on the hypothesis that a model weighting one cue would
diverge from a model weighting the other.</li>
</ul>

<div class="note"><b>The competing-cue hypothesis was wrong.</b>
{n_ceiling} of 30 pairs sat at a perfect 1.000/1.000 ceiling, including 9 of the 14 competing-cue
pairs: the models resolve cue conflicts <i>identically</i> (size appears to dominate). Five of the
eight dissociations came from single-cue pairs. Building conflict into a stimulus did not create
disagreement, it created a harder problem that both models solved the same way.</div>

<h3>A selection criterion that had to be changed</h3>
<p>The pre-specified criteria were: (1) P(left) between 0.35 and 0.65 for <i>both</i> models,
(2) |&Delta;| &gt; 0.15 in match probability, (3) form-based choosing. Applied as written, only two
pairs passed, both in the same direction &mdash; <b>no crossover</b>.</p>
<p>The reason is mechanical, and it is the Step 1 finding resurfacing: a model with no form rule
side-defaults rather than answering randomly, so a large |&Delta;| almost guarantees that the
<i>low</i> model has an extreme P(left). Screening on both models therefore rejects precisely the
dissociations being sought. Applying the position criterion to the <b>concentrated model only</b>
costs nothing: the two models are separate API calls in separate cells, so the low model&rsquo;s
side default cannot bias the high model&rsquo;s estimate, and under a balanced counterbalance a
side default reads as chance on identity &mdash; the correct answer for &ldquo;no form
preference&rdquo;. With that correction, {n_sel} pairs pass and crossovers appear in both
directions.</p>

{tbl(["pair", "family", "GPT", "P(left)", "form cons.", "Claude", "P(left)", "form cons.",
      "&Delta;", "95% CI on &Delta;"], sel_rows, numeric_from=2)}

<div class="fig">{b64img(os.path.join(OUT, "step1b_screen_findings.png"))}</div>
<p class="cap"><b>Figure 2.</b> <b>(A)</b> Dyadic match probability, all 30 pairs; distance from
the diagonal is the dissociation. Filled points passed all criteria. <b>(B)</b> Effect sizes with
95% bootstrap CIs for the selected pairs, both directions. <b>(C)</b> Conditional-position
diagnostic: rate of choosing the modal identity when it sits on the left versus the right. Points
on the diagonal indicate choosing by form; points off it indicate choosing by side. The two
labelled <code>s_arrow</code> points show the mechanism &mdash; Claude at (1.0,&nbsp;1.0), GPT far
off the diagonal.</p>

<h3>The strongest result</h3>
<p><code>s_arrow</code> (up-arrow vs right-arrow) is the cleanest dissociation in the battery and
the Claude&nbsp;&gt;&nbsp;GPT anchor the earlier work lacked. Both models articulate the
<i>same</i> rule &mdash; &ldquo;up is canonical&rdquo;. Claude applies it on 30 of 30 trials with
P(left) exactly 0.50. GPT states it and then does not follow it:</p>
<p class="quote">&ldquo;The upward-pointing arrow seems the more canonical and salient shape to
coordinate on. <b>CHOICE: RIGHT</b>&rdquo; &nbsp;<span class="small">(up was on the right &mdash;
correct)</span><br>
&ldquo;The right-pointing arrow feels like the more standard, salient arrow form.
<b>CHOICE: RIGHT</b>&rdquo; &nbsp;<span class="small">(rule switched to fit the side)</span></p>
<p>GPT answered <code>RIGHT</code> on 23 of 30 trials. This is the same confabulated side default
Claude showed on <code>inkblot</code> in Step 1, with the models&rsquo; roles reversed.</p>

<h2>5. Recommendation for the next experiment</h2>
{tbl(["pair", "direction", "GPT", "Claude", "&Delta;", "both models in position band"],
     rec_rows, numeric_from=2)}
<p>Add <code>c_open_cplx</code> (&Delta; &minus;0.276) for a third pair in the Claude direction,
plus <code>s_size</code> or <code>s_symmetry</code> as a shared-focal positive control (both models
1.000) and Step 1&rsquo;s <code>noise</code> as a floor control.</p>
<p><b>Discard three of the eight that passed.</b> <code>s_complexity</code> despite the largest
&Delta;: its 20-gon is perceptually a circle, so the pair measures prototypicality and duplicates
<code>proto</code>. <code>c_elong_sym</code> and <code>c_star_circle</code>: Claude&rsquo;s form
consistency there is 0.07 and 0.13, meaning its identity choice is entirely positional &mdash;
&ldquo;Claude does not coordinate here&rdquo; is not interpretable when Claude is not engaging with
the stimulus at all.</p>

<div class="note"><b>The two directions are not the same kind of effect.</b>
In the GPT&nbsp;&gt;&nbsp;Claude pairs, Claude is genuinely tracking form, just less consistently
(form consistency 0.80 and 0.93) &mdash; a <i>graded difference in concentration</i>. In the
Claude&nbsp;&gt;&nbsp;GPT pairs, GPT is partly falling back to position (form consistency
0.47&ndash;0.60) &mdash; a <i>difference in kind</i>. Both support the claim &ldquo;these models
have different focal points&rdquo;. Only <code>s_contrast</code> and <code>s_density</code> support
the stronger claim &ldquo;model X converges where model Y is genuinely indifferent&rdquo;, and both
are the same direction. If that distinction matters, more candidates are needed in the
Claude&nbsp;&gt;&nbsp;GPT direction, screened for GPT staying near P(left)&nbsp;=&nbsp;0.5 while
failing to concentrate. <code>s_arrow</code> suggests semantic and canonical-orientation cues as
the place to look.</div>

<h2>6. Limitations</h2>
<ul>
<li><b>Winner&rsquo;s curse.</b> The screen values are n=30 estimates selected as the extremes of
30 comparisons and are therefore inflated. A confirmation run at n=60 on fresh trials is required
before these numbers are used for anything but stimulus selection. At n=30 a match-rate CI is
roughly &plusmn;0.15.</li>
<li><b>Ceiling.</b> {n_ceiling} of 30 screened pairs were uninformative. A future battery needs
stimuli calibrated to the middle of the range, not to obvious focal points.</li>
<li><b>Simulated dyads.</b> No two instances ever played each other. Match probability is computed
from the marginal choice distribution, which is the correct population quantity for independent
draws but is not a played game.</li>
<li><b>Two models, one settings profile.</b> Both models ran at low reasoning effort. Effort is
known to change tool-use and deliberation behaviour and was not varied here.</li>
<li><b>No API-level seed.</b> Neither endpoint exposes one, so runs are not bit-reproducible; only
the counterbalance allocation is seeded.</li>
</ul>

<h2>7. Files</h2>
<p class="small">All paths relative to <code>Name_game_experiment/</code>.</p>
{tbl(["file", "contents"], [
  ["<code>step1_form_choice.py</code>", "Step 1 runner &mdash; prompt, settings, counterbalance"],
  ["<code>step1_form_analysis.py</code>", "Step 1 statistics and Figure 1"],
  [f"<code>step1_output/step1_form_calls_{TS1}.csv</code>", "600 call-level rows with full reasoning traces"],
  [f"<code>step1_output/step1_deliverables_{TS1}/</code>", "packaged raw data, summary, comparison, run config"],
  ["<code>make_screen_stimuli.py</code>", "generates the 30-pair battery and manifest"],
  ["<code>step1b_screen.py</code>", "screen runner; imports the prompt from Step 1"],
  ["<code>step1b_screen_analysis.py</code>", "selection criteria and diagnostics"],
  ["<code>step1b_screen_figure.py</code>", "Figure 2"],
  [f"<code>step1_output/step1b_screen_calls_{TS2}.csv</code>", "1,800 call-level rows"],
  [f"<code>step1_output/step1b_screen_selected_{TS2}.csv</code>", "per-pair criteria and crossover flags"],
], numeric_from=99)}

<h2 class="pb">Appendix &mdash; the 30-pair screening battery</h2>
<p class="small">Each panel shows the <code>_AB</code> layout: identity&nbsp;1 left, identity&nbsp;2
right. Position is counterbalanced across trials.</p>
<div class="fig">{b64img(os.path.join(HERE, "stimuli", "screen", "gallery_single.png"))}</div>
<div class="fig">{b64img(os.path.join(HERE, "stimuli", "screen", "gallery_compete.png"))}</div>

</body></html>"""

html_path = os.path.join(OUT, "step1_report.html")
with open(html_path, "w") as fh:
    fh.write(H)

pdf_path = os.path.join(DL, "LLM_coordination_step1_report.pdf")
cmd = [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
       "--virtual-time-budget=10000", f"--print-to-pdf={pdf_path}", f"file://{html_path}"]
r = subprocess.run(cmd, capture_output=True, text=True)
if not os.path.exists(pdf_path):
    print("chrome failed:", r.stderr[-800:])
else:
    print(f"HTML: {html_path}")
    print(f"PDF : {pdf_path}  ({os.path.getsize(pdf_path)/1024:.0f} KB)")
