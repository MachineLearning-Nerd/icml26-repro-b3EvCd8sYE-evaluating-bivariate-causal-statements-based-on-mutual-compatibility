"""The gapminder causal system used in Sections 2.7 and 3.4.

Everything here is transcribed from Appendix D.2 of arXiv:2606.00278: the seven
country-level indicators, the exact variable descriptions handed to the models,
and the empirical correlation matrix of Table 1.  Because the paper prints the
correlation matrix in full, the LLM experiments can be reproduced exactly
without re-deriving it from the raw gapminder files.
"""

from __future__ import annotations

import numpy as np

VARIABLES = [
    "population density",
    "literacy rate",
    "daily income",
    "sanitation access",
    "smoking",
    "happiness score",
    "life expectancy",
]

DESCRIPTIONS = {
    "population density":
        "Average number of people per square kilometer of land in the given "
        "country",
    "literacy rate":
        "adult literacy rate is the percentage of people ages 15 and above who "
        "can, with understanding, read and write a short, simple statement on "
        "their everyday life",
    "daily income":
        "mean daily household per capita income or consumption expenditure in "
        "constant international dollars",
    "sanitation access":
        "percentage of people using at least basic sanitation services "
        "(improved sanitation facilities not shared with other households)",
    "smoking":
        "percentage of people over age 15 that smoke",
    "happiness score":
        "national average response to a happiness survey, with scores ranging "
        "from 0 (worst) to 100 (best)",
    "life expectancy":
        "average life expectancy at birth in years",
}

# Table 1 of Appendix D.2, in the order of VARIABLES.
CORRELATION = np.array([
    [1.000, 0.109, 0.708, 0.104, -0.018, 0.078, 0.128],
    [0.109, 1.000, 0.373, 0.798, 0.109, 0.526, 0.716],
    [0.708, 0.373, 1.000, 0.381, 0.019, 0.745, 0.424],
    [0.104, 0.798, 0.381, 1.000, 0.190, 0.656, 0.817],
    [-0.018, 0.109, 0.019, 0.190, 1.000, 0.103, 0.096],
    [0.078, 0.526, 0.745, 0.656, 0.103, 1.000, 0.737],
    [0.128, 0.716, 0.424, 0.817, 0.096, 0.737, 1.000],
])

# Table 2 of Appendix D.2 lists nine models accessed through Amazon Bedrock.
# Six of them are served, under the same weights, by the Hugging Face inference
# router and are therefore reproducible here.  ``params_b`` is the total
# parameter count in billions and is used as the paper-independent capacity
# proxy for the "higher-capacity models tend to score higher" comparison.
MODELS = [
    dict(paper_name="gpt-oss-120b", route="openai/gpt-oss-120b", params_b=120.0),
    dict(paper_name="gpt-oss-20b", route="openai/gpt-oss-20b", params_b=20.0),
    dict(paper_name="Gemma 3 27B IT", route="google/gemma-3-27b-it", params_b=27.0),
    dict(paper_name="Gemma 3 4B IT", route="google/gemma-3-4b-it", params_b=4.0),
    dict(paper_name="Qwen3 235B A22B 2507",
         route="Qwen/Qwen3-235B-A22B-Instruct-2507", params_b=235.0),
    dict(paper_name="Qwen3 Next 80B A3B",
         route="Qwen/Qwen3-Next-80B-A3B-Instruct", params_b=80.0),
]

# The paper's own model set (six of nine) is too small to test "higher-capacity
# models tend to achieve higher scores" with any power: a Spearman correlation
# on six points needs |rho| >= 0.83 to reach p < 0.05.  EXTENDED_MODELS widen the
# ladder to thirteen models spanning 4B to 1T parameters.  These are NOT the
# paper's models and are reported separately, as a better-powered supplementary
# test of the same qualitative claim -- never mixed into the Table 2 comparison.
EXTENDED_MODELS = [
    dict(paper_name="Qwen3 4B Instruct 2507",
         route="Qwen/Qwen3-4B-Instruct-2507", params_b=4.0),
    dict(paper_name="Llama 3.1 8B Instruct",
         route="meta-llama/Llama-3.1-8B-Instruct", params_b=8.0),
    dict(paper_name="Gemma 3 12B IT", route="google/gemma-3-12b-it", params_b=12.0),
    dict(paper_name="Qwen3 32B", route="Qwen/Qwen3-32B", params_b=32.0),
    dict(paper_name="Llama 3.3 70B Instruct",
         route="meta-llama/Llama-3.3-70B-Instruct", params_b=70.0),
    dict(paper_name="DeepSeek V3", route="deepseek-ai/DeepSeek-V3", params_b=671.0),
    dict(paper_name="Kimi K2 Instruct", route="moonshotai/Kimi-K2-Instruct",
         params_b=1000.0),
]

ALL_MODELS = MODELS + EXTENDED_MODELS

# Models from Table 2 that the Hugging Face router does not serve.  Listed so
# the coverage gap is explicit rather than silent.
MODELS_UNAVAILABLE = [
    "Claude Opus 4.5 (claude-opus-4-5-20251101-v1:0)",
    "Kimi K2 Thinking (kimi-k2-thinking)",
    "Mistral Large 3 (mistral-large-3-675b-instruct)",
    "Magistral Small 2509 (magistral-small-2509)",
]

# Generation parameters, Appendix D.2.
GEN_PARAMS = dict(max_tokens=2048, temperature=0.6, top_p=0.7)


def correlation_string() -> str:
    """The ``{corr}`` placeholder of the D.3 prompts."""
    header = "            " + "".join(f"{v[:12]:>14}" for v in VARIABLES)
    lines = [header]
    for i, v in enumerate(VARIABLES):
        lines.append(f"{v:>22} " + "".join(f"{CORRELATION[i, j]:>14.3f}"
                                           for j in range(len(VARIABLES))))
    return "\n".join(lines)


def variables_string() -> str:
    """The ``{var names, var desc}`` placeholder of the D.3 prompts."""
    return "\n".join(f"- {v}: {DESCRIPTIONS[v]}" for v in VARIABLES)


def permuted_correlation(order: list[str]) -> np.ndarray:
    """Correlation matrix rearranged into a model's proposed causal ordering."""
    idx = [VARIABLES.index(v) for v in order]
    return CORRELATION[np.ix_(idx, idx)]
