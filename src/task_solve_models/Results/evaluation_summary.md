# Evaluation Summary Report

This file tracks the performance of various LLMs on the standardized `evaluation_batch.json` (50 questions from XFinBench).

**Last Updated:** Tuesday, Jan 7, 2026

## 🏆 Leaderboard (Single Agent - Recalculated)

| Rank | Model | Accuracy | Correct | Time | Parameters (Est.) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Gemini 2.5 Pro** | **84.0%** | 42/50 | 17m 31s | ~800 Billion |
| 2 | **Gemini 3 Pro Preview** | **78.0%** | 39/50 | 25m 26s | ~1 Trillion |
| 3 | **Claude 4.5 Opus** | **76.0%** | 38/50 | 4m 51s | ~1.5 Trillion+ |
| 4 | **GPT-5.1** | **68.0%** | 34/50 | 2m 46s | ~2 Trillion+ |
| 5 | **GPT-4o** | **58.0%** | 29/50 | 3m 22s | ~1.7 Trillion |
| 6 | **Claude 3.5 Haiku** | **58.0%** | 29/50 | 3m 06s | ~20 Billion |
| 7 | **Qwen 2.5 (7B)** | **48.0%** | 24/50 | 24m 46s | 7 Billion (Local) |
| 8 | **Mistral (7B)** | **36.0%** | 18/50 | 22m 34s | 7 Billion (Local) |
| 9 | **DeepSeek-LLM (7B)** | **30.0%** | 15/50 | 24m 32s | 7 Billion (Local) |
| 10 | **Llama 3 (8B)** | **28.0%** | 14/50 | 14m 45s | 8 Billion (Local) |

---

## 🤝 Multi-Agent Results (Heterogeneous Team)
**Team**: ScA: `Gemini 2.5 Pro` | ScB: `Claude 4.5 Opus` | ScC: `GPT-5.1` | Mod: `Gemini 3 Pro Preview`

| Config | Rounds | Accuracy | Correct | Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Smart Voting (R0)** | **0** | **86.0%** | **43/50** | 63m 56s | **Hetero Champion** 🏆 |
| **Refinement (R1)** | **1** | **84.0%** | 42/50 | 69m 37s | Stable |

---

## 🏛️ Multi-Agent Results (Homogeneous Teams - v3)
Testing 3x copies of the same model + same model as Moderator.

| Team | Config | Accuracy | Correct | Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **3x Claude 4.5 Opus** | R1 (1 Round) | **86.0%** | 43/50 | 55m 12s | **Co-Champion** 🏆 |
| **3x Claude 4.5 Opus** | R0 (0 Rounds) | 82.0% | 41/50 | 52m 53s | Solid |
| **3x Gemini 2.5 Pro** | R0 (0 Rounds) | 78.0% | 39/50 | 75m 2s | Good |
| **3x Gemini 2.5 Pro** | R1 (1 Round) | 74.0% | 37/50 | 76m 7s | Regressed |
| **3x GPT-5.1** | R0 (0 Rounds) | 72.0% | 36/50 | 31m 56s | Fast |
| **3x GPT-5.1** | R1 (1 Round) | 70.0% | 35/50 | 31m 08s | Regressed |
| **3x GPT-4o** | R1 (1 Round) | 58.0% | 29/50 | 33m 54s | Debate helps |
| **3x GPT-4o** | R0 (0 Rounds) | 50.0% | 25/50 | 30m 44s | Baseline |
| **3x Claude 3.5 Haiku** | R1 (1 Round) | **62.0%** | 31/50 | 40m 4s | **Debate helps** |
| **3x Claude 3.5 Haiku** | R0 (0 Rounds) | 58.0% | 29/50 | 38m 22s | Matches single-agent |
| **3x Gemini 3 Pro Preview** | R1 (1 Round) | 50.0% | 25/50 | 124m 35s | Underperformed |
| **3x Gemini 3 Pro Preview** | R0 (0 Rounds) | 30.0% | 15/50 | 87m 33s | Poor |
| **3x Qwen 2.5 (7B)** | R1 (1 Round) | 38.0% | 19/50 | 716m 34s | Debate helps |
| **3x Qwen 2.5 (7B)** | R0 (0 Rounds) | 36.0% | 18/50 | 482m 30s | Below single-agent |
| **Local Team (7B/8B)** | R0 (0 Rounds) | 26.0% | 13/50 | 514m 45s | Struggling |
| **Local Team (7B/8B)** | R1 (1 Round) | 32.0% | 16/50 | 601m 51s | Improved |

---

## 📊 Key Insights (Updated)

### 1. The Power of the "Strategic Audit"
The breakthrough to **86%** was achieved by adding a mandatory "Audit" phase. Scientists now weight the Moderator's proposed strategy and identify logic gaps before execution. This forced critical thinking is consistently outperforming zero-shot reasoning.

### 2. Homogeneous Claude Matches Heterogeneous Peak
The **3x Claude 4.5 Opus** team (R1) successfully matched our **86%** record. This shows that for the most capable models, self-debate is just as powerful as cross-model debate.

### 3. Gemini 3 Pro Preview: Unexpected Underperformance
Surprisingly, **3x Gemini 3 Pro Preview** performed significantly worse than **3x Gemini 2.5 Pro** (50% vs 78% in R0). This suggests that the "Preview" version may have instability issues when used in a multi-agent context, or that its reasoning style doesn't benefit from self-consistency checks. The R1 (50%) was better than R0 (30%), showing that debate rounds help, but the overall performance is concerning.

### 4. Gemini 2.5 Pro: R0 > R1 Pattern
Interestingly, **3x Gemini 2.5 Pro** showed the same pattern as the heterogeneous team: **R0 (78%) > R1 (74%)**. This suggests that Gemini's extremely verbose reasoning style might lead to "over-thinking" in debate rounds, where models start second-guessing their initially correct logic.

### 5. GPT-5.1 Speed vs. Consistency
While GPT-5.1 is nearly 2x faster than Claude or Gemini, its performance in a team setting peaked at **72%**. Adding a second round of debate (R1) actually *lowered* its score to **70%**, suggesting that multiple GPT-5.1 instances might amplify each other's hallucinations or "lazy" patterns during refinement.

### 6. GPT-4o: Debate Round Improves Performance
**3x GPT-4o** shows a clear benefit from debate rounds: **R0 (50%) → R1 (58%)**, an **8 percentage point improvement**. This is the opposite pattern from GPT-5.1 and Gemini 2.5 Pro, where R1 regressed. This suggests GPT-4o benefits from the refinement process, possibly because its initial answers are less confident and benefit from cross-validation.

### 7. Local Models: Multi-Agent Helps
The local 7B/8B models showed a **6% improvement** from R0 (26%) to R1 (32%), suggesting that for weaker models, the debate round actually helps catch errors that the initial consensus missed.

### 8. Claude 3.5 Haiku: Multi-Agent Exceeds Single-Agent
**3x Claude 3.5 Haiku** demonstrates that multi-agent collaboration can improve performance: **R0 (58%) matches single-agent (58%)**, but **R1 (62%) exceeds it by 4 percentage points**. This shows that even smaller models (~20B parameters) can benefit from debate rounds, with the team achieving better results than individual performance. The debate round helped identify 2 additional correct answers (31 vs 29).

### 9. Qwen 2.5: Multi-Agent Underperforms Single-Agent
**3x Qwen 2.5 (7B)** shows an interesting pattern: **R0 (36%) and R1 (38%) both underperform single-agent (48%)** by 10-12 percentage points. However, the debate round (R1) still provides a **2 percentage point improvement** over R0 (38% vs 36%), suggesting that while multi-agent collaboration doesn't match individual performance for this local model, the debate process still helps identify 1 additional correct answer (19 vs 18). This may indicate that Qwen's reasoning style doesn't benefit as much from consensus-building, or that the local model setup introduces coordination overhead that outweighs the benefits.

---

## 🛠️ Next Steps
- [x] Benchmark **3x Claude**, **3x GPT-5.1**, **3x Gemini 2.5 Pro**, and **3x Gemini 3 Pro Preview**.
- [ ] Investigate why **Gemini 3 Pro Preview** underperformed so dramatically in multi-agent mode (30-50% vs 78% single-agent).
- [ ] Analyze the 7 failures in the 86% runs to see if they overlap across teams.
- [ ] Try a **Hybrid Pipeline**: Using GPT-5.1 for fast "Drafting" and Claude/Gemini 2.5 Pro for "Strategic Auditing."
