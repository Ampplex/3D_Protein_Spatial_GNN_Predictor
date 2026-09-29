from __future__ import annotations

from typing import Any
import requests

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"


def post_search(query: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    response = requests.post(SEARCH_URL, json=query, timeout=timeout)
    response.raise_for_status()
    return response.json()


def search_entries(query: dict[str, Any], limit: int) -> list[str]:
    payload = dict(query)
    payload.setdefault("return_type", "entry")
    payload.setdefault("request_options", {})
    payload["request_options"] = dict(payload["request_options"])
    payload["request_options"]["paginate"] = {
        "start": 0,
        "rows": limit,
    }
    data = post_search(payload)
    return [item["identifier"] for item in data.get("result_set", [])]


def get_entry(pdb_id: str, timeout: int = 60) -> dict[str, Any]:
    response = requests.get(DATA_URL.format(pdb_id=pdb_id), timeout=timeout)
    response.raise_for_status()
    return response.json()
