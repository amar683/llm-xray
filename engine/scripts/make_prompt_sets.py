"""Generate the built-in prompt sets (original text, no external sources).

Run from the engine folder:  uv run python scripts/make_prompt_sets.py
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).parent.parent / "xray" / "data" / "prompts"

LANGUAGES = ["Python", "JavaScript", "TypeScript", "Go", "Rust"]
CODING_TASKS = [
    "reverses a string",
    "checks whether a number is prime",
    "returns the n-th Fibonacci number",
    "removes duplicates from a list while keeping the original order",
    "counts how often each word appears in a text",
    "merges two sorted arrays into one sorted array",
    "checks that the brackets in a string are balanced",
    "converts a Roman numeral to an integer",
    "finds the longest common prefix of a list of strings",
    "runs a binary search on a sorted array",
]

CHAT_TOPICS = [
    "the water cycle", "black holes", "the Roman Empire", "jazz music", "vaccines",
    "volcanoes", "the stock market", "photosynthesis", "the Olympic Games", "coral reefs",
]  # fmt: skip
CHAT_FORMS = [
    "What is {x} and why does it matter?",
    "Can you explain {x} to a ten-year-old?",
    "What are some common misconceptions about {x}?",
    "How has our understanding of {x} changed over the last century?",
    "Give me three interesting facts about {x}.",
]

MATH_FORMS = [
    "A shop sells pens at {a} dollars each. How much do {b} pens cost?",
    "A train travels {c} km in {a} hours. What is its average speed?",
    "What is {a} percent of {c}?",
    "Solve for x: {a}x + {b} = {c}.",
    "A rectangle is {a} cm long and {b} cm wide. What are its area and perimeter?",
]

ITEMS = [
    "a reusable water bottle", "a noise-cancelling headset", "a standing desk",
    "a bicycle helmet", "a pocket notebook", "a camping tent", "an electric kettle",
    "a board game", "a desk lamp", "a pair of running shoes",
]  # fmt: skip
INSTRUCTION_FORMS = [
    "Write a two-sentence product description for {x}.",
    "Rewrite this more formally: 'hey, {x} is kinda great, you should get one'.",
    "Give step-by-step instructions for choosing {x}.",
    "Reply with only a JSON object with the keys name, category and price for {x}.",
    "Translate into French: 'I would like to buy {x}.'",
]


def build() -> dict[str, list[str]]:
    rng = random.Random(0)
    math = []
    for form in MATH_FORMS:
        for _ in range(10):
            a, b = rng.randint(2, 12), rng.randint(3, 40)
            math.append(form.format(a=a, b=b, c=a * rng.randint(11, 60) + b))
    return {
        "coding": [
            f"Write a {lang} function that {task}." for task in CODING_TASKS for lang in LANGUAGES
        ],
        "chat": [form.format(x=x) for x in CHAT_TOPICS for form in CHAT_FORMS],
        "math": math,
        "instruction": [form.format(x=x) for x in ITEMS for form in INSTRUCTION_FORMS],
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, prompts in build().items():
        assert len(prompts) == len(set(prompts)) == 50, name
        (OUT / f"{name}.jsonl").write_text(
            "".join(json.dumps({"prompt": p}) + "\n" for p in prompts)
        )
        print(f"{name}: {len(prompts)} prompts")
