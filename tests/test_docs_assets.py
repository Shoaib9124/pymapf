"""The published site's data files must be readable by a browser.

`.docs/` is served directly to the browser and the playground reads these JSON
files at runtime, so a file that Python can write but `JSON.parse` cannot read
is a broken page rather than a failing build.
"""

import json
import os

import pytest

DOCS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".docs"
)
ASSETS = os.path.join(DOCS, "assets")


def published_json():
    """Every .json under .docs/."""
    found = []
    for directory, _, names in os.walk(DOCS):
        found.extend(
            os.path.join(directory, name) for name in names if name.endswith(".json")
        )
    return sorted(found)


def reject_constant(name):
    raise ValueError("%s is not valid JSON" % name)


@pytest.mark.parametrize("path", published_json(), ids=os.path.basename)
def test_published_json_is_valid_json(path):
    with open(path, encoding="utf-8") as handle:
        json.load(handle, parse_constant=reject_constant)


def test_mappo_policy_checkpoint_is_present_and_shaped():
    path = os.path.join(ASSETS, "mappo-policy.json")
    assert os.path.exists(path), "Pre-trained MAPPO checkpoint must exist in .docs/assets/"
    with open(path, encoding="utf-8") as handle:
        checkpoint = json.load(handle, parse_constant=reject_constant)

    assert checkpoint["algorithm"] == "mappo"
    assert checkpoint["centralized_critic"] is True
    assert checkpoint["obs_dim"] == 327
    assert checkpoint["n_actions"] == 5
    assert isinstance(checkpoint["actor"], list) and len(checkpoint["actor"]) == 6
    assert isinstance(checkpoint["critic"], list) and len(checkpoint["critic"]) == 6


def test_every_asset_the_page_references_exists():
    with open(os.path.join(DOCS, "index.html"), encoding="utf-8") as handle:
        page = handle.read()

    referenced = set()
    for attribute in ('src="', 'href="', 'poster="'):
        start = 0
        while True:
            found = page.find(attribute, start)
            if found < 0:
                break
            start = found + len(attribute)
            value = page[start : page.index('"', start)]
            if value.startswith(("http", "#", "data:", "mailto")):
                continue
            referenced.add(value.split("#")[0])

    missing = [
        value
        for value in sorted(referenced)
        if value and not os.path.exists(os.path.join(DOCS, value))
    ]
    assert not missing, "index.html links files that do not exist in .docs: %s" % missing
