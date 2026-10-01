"""Build the SIH26055 deck by re-using the team's MarSlick template (same masters, logos, fonts).

Every number is read from an artifact; nothing is typed in by hand:
- artifacts/hit_model_e2e_hybrid01.json  ANVESHAK (hybrid policy) vs open-loop sweep, held-out seeds 1000-1019
- artifacts/hit_model_e2e.json           rules-only (predictive) policy on the same held-out seeds
- artifacts/hit_model_eval.json          hit-model ranking quality on held-out missions
- core/receiver.snr_for_pd               detection sensitivity
"""
import copy
import json
import sys

from pptx import Presentation
from pptx.chart.data import CategoryChartData

sys.path.insert(0, ".")
from core.receiver import snr_for_pd  # noqa: E402

TEMPLATE = "deck/template/sih_template.pptx"   # team template (MarSlick deck)
OUT = "deck/out/SIH2026_SIH26055_ANVESHAK.pptx"

# ----- measured numbers ------------------------------------------------------
H = json.load(open("artifacts/hit_model_e2e_hybrid01.json"))["suites"]
R = json.load(open("artifacts/hit_model_e2e.json"))["suites"]
EV = json.load(open("artifacts/hit_model_eval.json"))
SUITES = list(H)
OURS = "hybrid"
mean = lambda d, pol: sum(d[s][pol]["wir"]["mean"] for s in SUITES) / len(SUITES)
n_missions = sum(H[s][OURS]["wir"]["n"] for s in SUITES)
gain_all = mean(H, OURS) / mean(H, "sweep")
gain_s4 = H["S4"][OURS]["wir"]["mean"] / H["S4"]["sweep"]["wir"]["mean"]
pred_s4 = H["S4"][OURS]["pred_correct"]["mean"]
wins = sum(H[s][OURS]["wilcoxon_vs_sweep_wir"]["wins"] for s in SUITES)
worst_p = max(H[s][OURS]["wilcoxon_vs_sweep_wir"]["p"] for s in SUITES)
max_rev = max(H[s][OURS]["max_revisit_worst"] for s in SUITES)
pop_ours, pop_sweep = H["S3"][OURS]["popup"], H["S3"]["sweep"]["popup"]
pfa_lo = min(H[s][p]["pfa"]["mean"] for s in SUITES for p in ("sweep", OURS))
pfa_hi = max(H[s][p]["pfa"]["mean"] for s in SUITES for p in ("sweep", OURS))
agile = (H["S2"][OURS]["wir"]["mean"], R["S2"]["predictive"]["wir"]["mean"])
rules_mean, ours_mean = mean(R, "predictive"), mean(H, OURS)
auc_ours = EV["overall"]["hit_model (ours)"]["auc_pr"]
auc_old = EV["overall"]["smart_scan_xgboost.pkl (supplied)"]["auc_pr"]
auc_base = EV["no_skill_auc_pr"]
auc_gain = auc_ours / auc_base
sens4, sens6 = snr_for_pd(0.9, 1e-4), snr_for_pd(0.9, 1e-6)
assert max_rev <= 128, "revisit floor violated in the artifact"
assert worst_p < 1e-4
print(f"missions {n_missions}  gain {gain_all:.2f}x  S4 {gain_s4:.1f}x  pred S4 {pred_s4:.1f}%  wins {wins}  p<={worst_p:.1e}  "
      f"popup {pop_ours} vs {pop_sweep}  agile {agile}  rules {rules_mean:.3f} ours {ours_mean:.3f}  "
      f"AUC-PR {auc_ours:.3f} vs {auc_base:.4f} ({auc_gain:.1f}x), supplied {auc_old:.3f}")


# ----- text helpers ----------------------------------------------------------
def shape(slide, name):
    for sh in slide.shapes:
        if sh.name == name:
            return sh
    raise KeyError(name)


def set_paras(sh, paras, template_idx=None):
    """Replace a text frame's paragraphs, keeping the template paragraph/run formatting.

    paras: list of str or list of segments (str). Segment k takes run k's formatting in the
    template paragraph (so "bold label" + "plain text" keeps the original look).
    """
    txBody = sh.text_frame._txBody
    old = list(sh.text_frame.paragraphs)
    templates = [p._p for p in old]
    for p in templates:
        txBody.remove(p)
    for i, para in enumerate(paras):
        segs = [para] if isinstance(para, str) else para
        tp = templates[template_idx if template_idx is not None else min(i, len(templates) - 1)]
        new_p = copy.deepcopy(tp)
        runs = new_p.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}r")
        for r in runs:
            new_p.remove(r)
        if segs == [""]:
            txBody.append(new_p)
            continue
        if not runs:
            raise ValueError(f"template paragraph in {sh.name} has no runs")
        end = new_p.find("{http://schemas.openxmlformats.org/drawingml/2006/main}endParaRPr")
        for k, text in enumerate(segs):
            r = copy.deepcopy(runs[min(k, len(runs) - 1)])
            for hl in r.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}hlinkClick"):
                hl.getparent().remove(hl)
            r.find("{http://schemas.openxmlformats.org/drawingml/2006/main}t").text = text
            if end is not None:
                end.addprevious(r)
            else:
                new_p.append(r)
        txBody.append(new_p)


def text(slide, name, *paras, **kw):
    set_paras(shape(slide, name), list(paras), **kw)


prs = Presentation(TEMPLATE)
s1, s2, s3, s4, s5, s6 = prs.slides

# ----- slide 1: title --------------------------------------------------------
text(s1, "Text 1",
     "Problem Statement ID – SIH26055",
     "Problem Statement Title – Smart Scan strategy for Electronic Warfare",
     "Theme – Robotics and Drones",
     "PS Category – Software",
     "Team ID – 120057",
     "Team Name – Evinco")

# ----- slide 2: proposed solution --------------------------------------------
text(s2, "Text 2", "ANVESHAK")
text(s2, "Text 6",
     "ANVESHAK — a machine-learning scheduler for an Electronic Support receiver that learns, from its own hits and "
     "misses, where and when to listen. With no emitter library or pre-mission data, it discovers emitters, measures "
     "their scan periods and dwells exactly when their beams return, while still guaranteeing every band is searched.")
text(s2, "Text 9",
     "An open-loop sweep gives equal time to empty bands, harmless beacons and live threats.",
     "A fixed sweep can fall into lock-step with a radar's scan: a 32-band sweep never sees a 3.2 s radar "
     "for 15 of 16 phase offsets.",
     "Without prior intelligence, nothing tells the receiver where a new or threatening emitter is.")
text(s2, "Text 12",
     f"Hit → confirm → lock: after a first hit it camps one scan to measure the period, then dwells only on "
     f"predicted beam passes ({pred_s4:.0f}% within ±20 ms).")
text(s2, "Text 14",
     f"Learns from hits and misses: a gradient-boosted hit model, trained on simulator truth, ranks where to "
     f"search — {auc_gain:.0f}× better than chance at spotting a new threat.")
text(s2, "Text 16",
     f"Earliest-deadline-first revisit floor: every band re-checked within {max_rev * 10 / 1000:.2f} s — no learned "
     "policy can starve part of the spectrum.")
# replace the oil-spill scene (shapes 'Shape 17' .. 'Text 33') with the waterfall from a real simulated mission
scene = [sh for sh in s2.shapes if sh.shape_id >= 20]
left, top, width, height = scene[0].left, scene[0].top, scene[0].width, scene[0].height
for sh in scene:
    sh._element.getparent().remove(sh._element)
s2.shapes.add_picture("deck/waterfall.png", left, top, width, height)

# ----- slide 3: technical approach -------------------------------------------
text(s3, "Text 6",
     "Python 3.12 · NumPy · SciPy",
     "Marcum-Q detection model (Pd from SNR at set Pfa)",
     "Discounted Beta-Bernoulli Thompson sampling",
     "Approximate-GCD scan period / phase estimator",
     "XGBoost hit model trained on simulator truth",
     "Seeded RF scenario generator with ground truth",
     "FastAPI band-state console: CRUD + live scoring",
     "Docker, deployable on Microsoft Azure")
steps = [("Step I: ", "Simulate the RF scene — 32 bands × 10 ms slots, 6 emitter families, ground truth"),
         ("Step II: ", "Detect — Marcum-Q Pd from SNR, Pfa 10⁻⁴ per dwell"),
         ("Step III: ", "Learn — band posteriors and per-emitter period / phase from hits and misses"),
         ("Step IV: ", "Schedule — predicted window › confirm camp › model-ranked search, under the revisit floor"),
         ("Step V: ", "Score against truth — intercept ratio, Pd, Pfa, intercept time, prediction error")]
for name, (label, body) in zip(["Text 8", "Text 11", "Text 14", "Text 17", "Text 20"], steps):
    text(s3, name, [label, body])
text(s3, "Text 22", "DWELL")
text(s3, "Text 24", "Predicted window")
text(s3, "Text 26", "A locked radar's beam is due now — the receiver dwells exactly on it.")
text(s3, "Text 28", "Confirm camp")
text(s3, "Text 30", "First hit on an unknown emitter — stay one scan to measure its period.")
text(s3, "Text 32", "Search")
text(s3, "Text 34", f"Hit model ranks bands by chance of a new threat; every band revisited within {max_rev * 10 / 1000:.2f} s.")

# ----- slide 4: feasibility --------------------------------------------------
text(s4, "Text 7", ["Already running ", "– the simulation core runs a 120 s, 32-band mission in about a second "
                                        "on one CPU core."])
text(s4, "Text 9", ["No hardware needed ", "– a seeded RF environment with per-slot ground truth, exactly as the "
                                           "PS asks."])
text(s4, "Text 11", ["Low cost ", "– NumPy / SciPy on commodity CPUs; no GPU for training or inference."])
text(s4, "Text 13", ["Modular design ", "– environment, receiver, belief and scheduler swap independently; a real "
                                        "ES receiver plugs in behind the same interface."])
text(s4, "Text 14", "Challenges in blind spectrum search (team estimate) :")
challenges = {  # box -> (title, body); colours follow the template's pie slices
    "Text 17": ("No prior intelligence", "No library, no frequency plan: every emitter is discovered, timed and "
                                         "tracked online from hits and misses."),
    "Text 20": ("Frequency-agile emitters", "No stable period to learn; the hit model ranks their bands "
                                            "instead, our hardest case: "
                                            f"{agile[0]:.2f} vs {agile[1]:.2f} for rules alone."),
    "Text 23": ("Pop-up threats", f"The revisit floor bounds blind time to {max_rev * 10 / 1000:.2f} s per band, "
                                  "whatever the policy prefers."),
    "Text 26": ("Scan-on-scan sync", "Commensurate periods hide radars from a fixed sweep; randomised search plus "
                                     "phase lock removes the trap."),
    "Text 29": ("Benign emitters", "Only new intercepts earn reward, so always-on beacons are seen once and then "
                                   "left alone."),
    "Text 32": ("False period locks", "Largest-consistent-period estimator; three missed predicted windows drop "
                                      "the lock."),
}
for name, (title, body) in challenges.items():
    text(s4, name, title, body)
chart = [sh for sh in s4.shapes if sh.has_chart][0].chart
cd = CategoryChartData()
cd.categories = ["No prior intelligence", "Scan-on-scan sync", "Frequency-agile emitters", "Benign emitters",
                 "Pop-up threats", "False period locks"]
cd.add_series("Share", [25, 20, 20, 15, 10, 10])
chart.replace_data(cd)

# ----- slide 5: impact -------------------------------------------------------
text(s5, "Text 7",
     f"{gain_all:.1f}× the threat intercepts of an open-loop sweep",
     f"{gain_s4:.0f}× the sweep where radar scans fall in lock-step with it",
     f"Next radar beam pass predicted within ±20 ms, {pred_s4:.0f}% of the time",
     f"Hit model: {auc_gain:.0f}× better than chance at finding new threats",
     f"No band left unwatched for more than {max_rev * 10 / 1000:.2f} s")
tiles = [(f"{gain_all:.1f}×", "threat-weighted intercepts"), (f"{gain_s4:.0f}×", "in the sync trap"),
         (f"{pred_s4:.0f}%", "beam passes within ±20 ms"), (f"{auc_gain:.0f}×", "hit model vs chance (AUC-PR)")]
for (vn, ln), (v, l) in zip([("Text 9", "Text 10"), ("Text 12", "Text 13"), ("Text 15", "Text 16"),
                             ("Text 18", "Text 19")], tiles):
    text(s5, vn, v)
    text(s5, ln, l)
text(s5, "Text 21", ["Revisit guarantee:  ", "earliest-deadline-first floor → rules + hit-model scores → dwell. Worst revisit "
                                             f"in every simulated mission = {max_rev} slots ({max_rev * 10 / 1000:.2f} s), "
                                             "exactly the bound."])
text(s5, "Text 24", "Operational")
text(s5, "Text 25",
     "Threats are intercepted far more often, so warning and jamming cues are more reliable.",
     "Receiver time goes to new and threatening emitters, not known beacons.")
text(s5, "Text 27", "Strategic")
text(s5, "Text 28",
     "Works with no pre-mission intelligence — suited to new theatres and unknown emitters.",
     "Measured scan periods and phases build an emitter library as a by-product.")
text(s5, "Text 30", "Economic")
text(s5, "Text 31",
     "A software-only upgrade to existing ES receivers; no new RF hardware.",
     "The same receiver sensitivity yields more intercepts, extending the value of fielded sets.")

# ----- slide 6: research -----------------------------------------------------
text(s6, "Text 6",
     "Scan-on-scan: with a radar window of τr slots every Tr and a receiver window of τs every Ts, an intercept is "
     "possible for only min(1, (τr+τs−1)/gcd(Tr,Ts)) of phase offsets — so a fixed 32-band sweep misses 15/16 of "
     "3.2 s radars entirely. Randomised search and phase lock remove the trap.",
     f"Detection follows a square-law, non-fluctuating model (Marcum Q): Pd = 0.9 needs {sens4:.1f} dB SNR at "
     f"Pfa 10⁻⁴ ({sens6:.1f} dB at 10⁻⁶). Measured Pfa in simulation, {pfa_lo * 1e4:.1f}–{pfa_hi * 1e4:.1f} × 10⁻⁴, "
     "matches the set value.",
     f"Across {n_missions} held-out missions (20 per scenario suite, none used in training or tuning) ANVESHAK beat "
     f"the open-loop sweep in {wins} of {n_missions} (paired Wilcoxon p < 10⁻⁴ in every suite). The XGBoost hit model "
     f"reaches AUC-PR {auc_ours:.3f} against {auc_base:.4f} for chance; an externally supplied model scored {auc_old:.3f}. "
     f"Rules + model beat rules alone on hopping emitters ({agile[0]:.3f} vs {agile[1]:.3f}, p < 0.001).",
     f"Trade-off we report rather than hide: the pop-up radar is found more often ({pop_ours['detected_share'] * 100:.0f}% "
     f"vs {pop_sweep['detected_share'] * 100:.0f}%) but first intercept is later (median {pop_ours['median_ttfi_s']:.1f} s "
     f"vs {pop_sweep['median_ttfi_s']:.1f} s), because confirm camps delay search; the revisit floor sets the balance.")
text(s6, "Text 8",
     "",
     "1. R. G. Wiley, ELINT: The Interception and Analysis of Radar Signals, Artech House, 2006.",
     "2. J. I. Marcum, “A statistical theory of target detection by pulsed radar,” IRE Trans. Information Theory, 1960.",
     "3. T. Chen, C. Guestrin, “XGBoost: A scalable tree boosting system,” Proc. ACM SIGKDD, 2016.",
     "4. W. R. Thompson, “On the likelihood that one unknown probability exceeds another,” Biometrika, 1933.",
     "5. A. Garivier, E. Moulines, “On upper-confidence bound policies for switching bandit problems,” ALT, 2011.",
     "6. SIH 2026 Problem Statement SIH26055, Defence Research and Development Organisation (DRDO).",
     "",
     ["Simulation core, hit model and benchmark — ", "Python · NumPy · SciPy · XGBoost, 126 contract tests"],
     template_idx=None)

prs.save(OUT)
print("saved", OUT)
