#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Black–Scholes Mastery: Terminal Game
------------------------------------
Features
- Levels: Concepts → Formula Build → Calculations → Theory Chain (Boss)
- Score & Lives
- Randomized questions
- Numeric auto-check with tolerances
- Fully offline; standard library only

Run:
    python black_scholes_mastery_game.py
"""

import math
import random
import sys
import time
import os
from textwrap import dedent

# ---------- Utils ----------

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "blue": "\033[34m",
    "grey": "\033[90m",
}

def c(s, color=None, bold=False):
    start = ""
    if color and color in ANSI:
        start += ANSI[color]
    if bold:
        start += ANSI["bold"]
    end = ANSI["reset"]
    return f"{start}{s}{end}"

def banner(text):
    line = "=" * (len(text) + 8)
    print(c(line, "cyan", True))
    print(c(f"=== {text} ===", "cyan", True))
    print(c(line, "cyan", True))

def pause(msg="Press Enter to continue..."):
    try:
        input(c(msg, "grey"))
    except EOFError:
        pass

def ask(prompt):
    try:
        return input(c(prompt + " ", "yellow"))
    except EOFError:
        return ""

def pretty_table(rows, header=None):
    cols = len(rows[0]) if rows else (len(header) if header else 0)
    widths = [0]*cols
    data = rows[:]
    if header:
        for j, h in enumerate(header):
            widths[j] = max(widths[j], len(str(h)))
    for r in data:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(str(cell)))
    def fmt_row(r):
        return " | ".join(str(cell).ljust(widths[j]) for j, cell in enumerate(r))
    out = []
    if header:
        out.append(fmt_row(header))
        out.append("-+-".join("-"*w for w in widths))
    for r in data:
        out.append(fmt_row(r))
    return "\n".join(out)

# ---------- Math core: Normal CDF & Black–Scholes ----------

def phi(x):
    "Standard normal PDF"
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def N(x):
    "Standard normal CDF using erf"
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def d1(S, K, r, sigma, T, t=0.0, q=0.0):
    tau = max(T - t, 0.0)
    if sigma <= 0 or tau <= 0 or S <= 0 or K <= 0:
        return float('nan')
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * math.sqrt(tau))

def d2(S, K, r, sigma, T, t=0.0, q=0.0):
    tau = max(T - t, 0.0)
    if sigma <= 0 or tau <= 0 or S <= 0 or K <= 0:
        return float('nan')
    return d1(S, K, r, sigma, T, t, q) - sigma * math.sqrt(tau)

def bs_call(S, K, r, sigma, T, t=0.0, q=0.0):
    tau = max(T - t, 0.0)
    if tau == 0:
        return max(S - K, 0.0)
    _d1 = d1(S, K, r, sigma, T, t, q)
    _d2 = d2(S, K, r, sigma, T, t, q)
    return S * math.exp(-q * tau) * N(_d1) - K * math.exp(-r * tau) * N(_d2)

def bs_put(S, K, r, sigma, T, t=0.0, q=0.0):
    tau = max(T - t, 0.0)
    if tau == 0:
        return max(K - S, 0.0)
    _d1 = d1(S, K, r, sigma, T, t, q)
    _d2 = d2(S, K, r, sigma, T, t, q)
    return K * math.exp(-r * tau) * N(-_d2) - S * math.exp(-q * tau) * N(-_d1)

def parity_call_from_put(put, S, K, r, T, t=0.0, q=0.0):
    tau = max(T - t, 0.0)
    return put + S * math.exp(-q * tau) - K * math.exp(-r * tau)

def approx_eq(a, b, tol=1e-4):
    if a == b:
        return True
    if any(map(lambda x: (x is None) or (isinstance(x, float) and math.isnan(x)), [a, b])):
        return False
    return abs(a - b) <= tol * (1.0 + max(abs(a), abs(b)))

# ---------- Game Content ----------

TERMS = {
    "S": "Current asset price",
    "K": "Strike price",
    "r": "Risk-free interest rate (continuously compounded)",
    "q": "Dividend yield (continuous)",
    "σ": "Volatility of returns (standard deviation)",
    "t": "Current time",
    "T": "Maturity time",
    "τ": "Time to maturity = T - t",
    "N(x)": "Standard normal cumulative distribution function",
    "φ(x)": "Standard normal density (pdf)",
    "d1": "Scaled drifted log-moneyness term used in BS formula",
    "d2": "d1 minus σ√τ",
}

PDE_TEXT = dedent(r"""
Black–Scholes PDE (no arbitrage, delta-hedging):
    ∂f/∂t + (1/2) σ^2 S^2 ∂²f/∂S² + (r - q) S ∂f/∂S - r f = 0

European call boundary/terminal conditions:
    f(S, T) = max(S - K, 0)
    f(0, t) = 0
    f(S, t) ~ S as S → ∞ (for call with q=0); with dividend yield q, asymptote S e^{-q τ}
""").strip()

CHAIN_STEPS = [
    "Start from risk-neutral dynamics dS_t = (r - q) S_t dt + σ S_t dW_t",
    "Derive the PDE via Itô's lemma and no-arbitrage (delta-hedging)",
    "Apply Feynman–Kac to represent the PDE solution as a conditional expectation",
    "Evaluate the expectation using lognormal distribution of S_T",
    "Arrive at closed-form Black–Scholes: C = S e^{-qτ} N(d1) - K e^{-rτ} N(d2)",
]

FORMULAE = [
    ("d1", "d1 = [ln(S/K) + (r - q + 0.5 σ^2) τ] / (σ√τ)"),
    ("d2", "d2 = d1 - σ√τ"),
    ("Call", "C = S e^{-qτ} N(d1) - K e^{-rτ} N(d2)"),
    ("Put",  "P = K e^{-rτ} N(-d2) - S e^{-qτ} N(-d1)"),
    ("Parity", "C - P = S e^{-qτ} - K e^{-rτ}"),
]

# ---------- Question Generators ----------

def gen_concept_mcq():
    key, meaning = random.choice(list(TERMS.items()))
    wrongs = random.sample([m for k, m in TERMS.items() if k != key], k=3)
    options = wrongs + [meaning]
    random.shuffle(options)
    correct_idx = options.index(meaning)
    q = f"What does '{key}' represent?"
    return q, options, correct_idx

def gen_formula_fill():
    # Randomly drop a token from one of the core formulae and ask to fill it
    name, full = random.choice(FORMULAE)
    tokens = [
        "ln", "S", "K", "r", "q", "σ", "τ", "N", "d1", "d2",
        "C", "P", "e", "√"
    ]
    # choose a token actually present
    present = [t for t in tokens if t in full]
    missing = random.choice(present) if present else "σ"
    masked = full.replace(missing, c("□", "yellow", True))
    prompt = f"Fill the missing token in the {name} formula:\n  {masked}\nYour answer:"
    return prompt, missing

def gen_calc_round():
    # Generate parameters and ask for call/put price or d1/d2
    S = round(random.uniform(50, 200), 2)
    K = round(random.uniform(50, 200), 2)
    sigma = round(random.uniform(0.05, 0.6), 4)
    r = round(random.uniform(0.0, 0.08), 4)
    q = round(random.uniform(0.0, 0.04), 4)
    tau = random.choice([0.25, 0.5, 1.0, 2.0])
    target = random.choice(["C", "P", "d1", "d2"])
    return {"S": S, "K": K, "sigma": sigma, "r": r, "q": q, "tau": tau, "target": target}

def compute_target(params):
    S = params["S"]; K = params["K"]; r = params["r"]; sigma = params["sigma"]; q = params["q"]; tau = params["tau"]
    if params["target"] == "C":
        return bs_call(S, K, r, sigma, T=tau, t=0.0, q=q)
    if params["target"] == "P":
        return bs_put(S, K, r, sigma, T=tau, t=0.0, q=q)
    if params["target"] == "d1":
        return d1(S, K, r, sigma, T=tau, t=0.0, q=q)
    if params["target"] == "d2":
        return d2(S, K, r, sigma, T=tau, t=0.0, q=q)
    return None

def format_params(params):
    rows = [
        ("S", params["S"]),
        ("K", params["K"]),
        ("r", params["r"]),
        ("q", params["q"]),
        ("σ", params["sigma"]),
        ("τ", params["tau"]),
    ]
    return pretty_table(rows, header=["Param", "Value"])

def gen_chain_shuffle():
    steps = CHAIN_STEPS[:]
    random.shuffle(steps)
    return steps

# ---------- Levels ----------

def level_concepts(score, lives):
    banner("Level 1 — Concepts")
    rounds = 6
    gained = 0
    for i in range(1, rounds+1):
        q, options, correct = gen_concept_mcq()
        print(c(f"\nQ{i}: {q}", "magenta", True))
        for idx, opt in enumerate(options):
            print(f"  {idx+1}. {opt}")
        ans = ask("Your choice (1-4):")
        try:
            ai = int(ans)-1
        except:
            ai = -1
        if ai == correct:
            print(c("✓ Correct!", "green", True))
            score += 10
            gained += 1
        else:
            print(c("✗ Wrong.", "red", True))
            print(c(f"Correct: {options[correct]}", "green"))
            lives -= 1
            if lives == 0:
                return score, lives
    print(c(f"\nLevel complete! +{gained*10} pts", "green", True))
    return score, lives

def level_formula_build(score, lives):
    banner("Level 2 — Formula Build")
    rounds = 5
    for i in range(1, rounds+1):
        prompt, missing = gen_formula_fill()
        print(c(f"\nQ{i}:", "magenta", True))
        print(prompt)
        ans = ask("Token (case-sensitive, e.g., σ, d1, N):").strip()
        if ans == missing:
            print(c("✓ Correct!", "green", True))
            score += 15
        else:
            print(c("✗ Wrong.", "red", True))
            print(c(f"Correct token: {missing}", "green"))
            lives -= 1
            if lives == 0:
                return score, lives
    print(c(f"\nLevel complete! +{5*15} pts", "green", True))
    return score, lives

def level_calculations(score, lives):
    banner("Level 3 — Calculations")
    rounds = 5
    for i in range(1, rounds+1):
        p = gen_calc_round()
        print(c(f"\nQ{i}: Compute {p['target']} given:", "magenta", True))
        print(format_params(p))
        start = time.time()
        ans = ask("Your numeric answer:").strip()
        try:
            val = float(ans)
        except:
            val = None
        correct = compute_target(p)
        elapsed = time.time() - start
        # dynamic tolerance: looser if answered quickly
        tol = 1e-3 if elapsed > 20 else 5e-3
        if val is not None and approx_eq(val, correct, tol=tol):
            print(c(f"✓ Correct! (true ≈ {correct:.6f})", "green", True))
            bonus = 10 if elapsed < 10 else 0
            score += 25 + bonus
            if bonus:
                print(c(f"Speed bonus +{bonus}!", "yellow", True))
        else:
            print(c("✗ Not quite.", "red", True))
            print(c(f"True value ≈ {correct:.6f}", "green"))
            lives -= 1
            if lives == 0:
                return score, lives
    print(c(f"\nLevel complete! +{5*25} pts (plus any bonuses)", "green", True))
    return score, lives

def level_boss(score, lives):
    banner("Boss Level — Theory Chain")
    steps_shuffled = gen_chain_shuffle()
    print("Put these steps in the correct logical order (enter a sequence like 3-1-4-2-5):\n")
    for i, s in enumerate(steps_shuffled, 1):
        print(c(f"{i}. {s}", "blue"))
    correct_order = [CHAIN_STEPS.index(s)+1 for s in steps_shuffled]
    ans = ask("\nYour order:").strip().replace(" ", "")
    try:
        pick = [int(x) for x in ans.replace(",", "-").split("-") if x]
    except:
        pick = []
    if pick == correct_order:
        print(c("✓ Perfect chain!", "green", True))
        score += 60
    else:
        print(c("✗ Chain out of order.", "red", True))
        print(c("Correct order should map to:", "green", True))
        mapping = " - ".join(str(i) for i in correct_order)
        print(c(f"{mapping}", "green"))
        lives -= 1
    return score, lives

# ---------- Game Flow ----------

def intro():
    banner("Black–Scholes Mastery")
    print(c("Your mission: prove you know the model inside-out — symbols, formulae, math, and theory.\n", "grey"))
    print(PDE_TEXT)
    print()
    pause()

def outro(score, lives):
    banner("Final Report")
    grade = "S" if score >= 250 else ("A" if score >= 210 else ("B" if score >= 170 else ("C" if score >= 130 else "Keep Training")))
    print(pretty_table([
        ("Score", score),
        ("Lives Left", lives),
        ("Rank", grade),
    ], header=["Metric", "Value"]))
    tips = [
        "Re-derive d1 and d2 from log(S_T/K) ~ Normal.",
        "Memorize put-call parity and test edge cases (τ→0, σ→0).",
        "Practice PDE → Feynman–Kac path until it's reflex.",
    ]
    print("\nStudy tips:")
    for t in tips:
        print(" - " + t)
    print("\n" + c("Replay to face new parameters & questions. Good luck!", "cyan", True))

def main():
    random.seed()
    intro()
    score = 0
    lives = 3

    # Levels
    score, lives = level_concepts(score, lives)
    if lives == 0:
        outro(score, lives); return
    score, lives = level_formula_build(score, lives)
    if lives == 0:
        outro(score, lives); return
    score, lives = level_calculations(score, lives)
    if lives == 0:
        outro(score, lives); return
    score, lives = level_boss(score, lives)

    outro(score, lives)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n" + c("Session aborted. See you next time.", "grey"))
