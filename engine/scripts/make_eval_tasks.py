"""Generate the built-in eval tasks (questions with one exact answer).

Run from the engine folder:  uv run python scripts/make_eval_tasks.py
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).parent.parent / "xray" / "data" / "evals"

CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid",
    "Portugal": "Lisbon", "Japan": "Tokyo", "China": "Beijing", "Russia": "Moscow",
    "Egypt": "Cairo", "Greece": "Athens", "Ireland": "Dublin", "Austria": "Vienna",
    "Poland": "Warsaw", "Norway": "Oslo", "Sweden": "Stockholm", "Finland": "Helsinki",
    "Denmark": "Copenhagen", "Belgium": "Brussels", "Hungary": "Budapest", "Cuba": "Havana",
    "Peru": "Lima", "Chile": "Santiago", "Argentina": "Buenos Aires", "Mexico": "Mexico City",
    "Canada": "Ottawa", "Kenya": "Nairobi", "Thailand": "Bangkok", "Vietnam": "Hanoi",
    "South Korea": "Seoul", "Indonesia": "Jakarta", "Iran": "Tehran", "Iraq": "Baghdad",
    "Turkey": "Ankara", "Ukraine": "Kyiv", "Romania": "Bucharest", "Bulgaria": "Sofia",
    "Serbia": "Belgrade", "Croatia": "Zagreb", "the Czech Republic": "Prague",
    "the Netherlands": "Amsterdam", "the Philippines": "Manila", "Colombia": "Bogot",
    "Venezuela": "Caracas", "Ethiopia": "Addis Ababa", "Ghana": "Accra", "Nepal": "Kathmandu",
    "Bangladesh": "Dhaka", "Lebanon": "Beirut", "Syria": "Damascus", "Afghanistan": "Kabul",
}  # fmt: skip


THINGS = [
    "a blue backpack", "a wooden chair", "a steel water bottle", "a pair of sunglasses",
    "a paperback novel", "a tennis racket", "a coffee grinder", "a wool scarf",
    "a garden hose", "a wireless mouse", "a ceramic mug", "a yoga mat", "a kitchen knife",
    "a wall clock", "a pair of hiking boots", "a phone charger", "a chess set",
    "a rain jacket", "a frying pan", "a guitar strap", "a bath towel", "a bike lock",
    "a notebook", "a table fan", "a leather wallet", "a soccer ball", "a toaster",
    "a sleeping bag", "a picture frame", "a flashlight",
]  # fmt: skip
JSON_SHOTS = (
    'Item: a desk lamp\nJSON: {"name": "desk lamp", "category": "lighting", "price": 25}\n'
    'Item: a red bicycle\nJSON: {"name": "red bicycle", "category": "sports", "price": 180}\n'
)

# name, arguments, docstring, [(arguments, expected result), ...]
FUNCTIONS = [
    ("is_even", "n", "Return True if n is even.", [([4], True), ([7], False), ([0], True)]),
    ("square", "x", "Return x squared.", [([3], 9), ([-2], 4), ([0], 0)]),
    ("max_of_three", "a, b, c", "Return the largest of a, b and c.", [([1, 5, 3], 5), ([9, 2, 4], 9), ([-1, -2, -3], -1)]),
    ("reverse_string", "s", "Return s reversed.", [(["abc"], "cba"), ([""], ""), (["xy"], "yx")]),
    ("count_vowels", "s", "Return how many vowels (a, e, i, o, u) are in the lowercase string s.", [(["banana"], 3), (["sky"], 0), (["audio"], 4)]),
    ("factorial", "n", "Return n factorial for n >= 0.", [([0], 1), ([5], 120), ([3], 6)]),
    ("is_palindrome", "s", "Return True if s reads the same backwards.", [(["level"], True), (["abc"], False), (["a"], True)]),
    ("sum_list", "xs", "Return the sum of the numbers in xs.", [([[1, 2, 3]], 6), ([[]], 0), ([[-1, 1]], 0)]),
    ("abs_diff", "a, b", "Return the absolute difference between a and b.", [([3, 7], 4), ([7, 3], 4), ([2, 2], 0)]),
    ("last_element", "xs", "Return the last element of the non-empty list xs.", [([[1, 2, 3]], 3), ([["a"]], "a"), ([[0, 9]], 9)]),
    ("double_list", "xs", "Return a new list with every number in xs doubled.", [([[1, 2]], [2, 4]), ([[]], []), ([[-3]], [-6])]),
    ("clamp", "x, lo, hi", "Return x limited to the range lo to hi.", [([5, 0, 3], 3), ([-2, 0, 3], 0), ([2, 0, 3], 2)]),
    ("first_char", "s", "Return the first character of the non-empty string s.", [(["hello"], "h"), (["x"], "x"), (["ab"], "a")]),
    ("repeat_string", "s, n", "Return s repeated n times.", [(["ab", 2], "abab"), (["x", 0], ""), (["hi", 1], "hi")]),
    ("min_of_two", "a, b", "Return the smaller of a and b.", [([1, 2], 1), ([5, 3], 3), ([4, 4], 4)]),
    ("cube", "x", "Return x cubed.", [([2], 8), ([-1], -1), ([0], 0)]),
    ("is_positive", "n", "Return True if n is greater than zero.", [([3], True), ([0], False), ([-5], False)]),
    ("list_length", "xs", "Return the number of items in xs.", [([[1, 2, 3]], 3), ([[]], 0), ([["a"]], 1)]),
    ("to_upper", "s", "Return s in upper case.", [(["abc"], "ABC"), (["Hi"], "HI"), ([""], "")]),
    ("average", "xs", "Return the average of the non-empty list of numbers xs.", [([[2, 4]], 3.0), ([[5]], 5.0), ([[1, 2, 3]], 2.0)]),
]  # fmt: skip
CODE_SHOTS = (
    'def add(a, b):\n    """Return a plus b."""\n    return a + b\n\n'
    'def is_empty(xs):\n    """Return True if the list xs has no items."""\n    return len(xs) == 0\n\n'
)


def build() -> dict[str, list[dict]]:
    rng = random.Random(0)
    sums = []
    while len(sums) < 100:
        a, b = rng.randint(2, 60), rng.randint(2, 60)
        item = {"prompt": f"3 + 4 = 7\n15 + 22 = 37\n{a} + {b} =", "answer": str(a + b)}
        if item not in sums:
            sums.append(item)
    return {
        "capitals": [
            {"prompt": f"The capital of {country} is", "answer": city}
            for country, city in CAPITALS.items()
        ],
        "arithmetic": sums,
        "json": [
            {
                "prompt": f"{JSON_SHOTS}Item: {thing}\nJSON:",
                "check": "json",
                "keys": ["name", "category", "price"],
            }
            for thing in THINGS
        ],  # fmt: skip
        "code": [
            {
                "prompt": CODE_SHOTS + (stub := f'def {name}({args}):\n    """{doc}"""\n'),
                "check": "python",
                "entry": name,
                "stub": stub,
                "tests": tests,
            }
            for name, args, doc, tests in FUNCTIONS
        ],  # fmt: skip
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, items in build().items():
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
        print(f"{name}: {len(items)} items")
