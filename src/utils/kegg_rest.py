"""
Minimal KEGG REST helper for reverse gene->drug linking.

Endpoints used:
- link: https://rest.kegg.jp/link/drug/hsa:{gene_id}
- get:  https://rest.kegg.jp/get/{drug_id}

We prefer to return a list of KEGG drug IDs (e.g., D00109). Detailed info can be
fetched via existing KEGG MCP or REST if needed.
"""

from typing import List
import urllib.request


def link_drugs_for_gene(kegg_gene_id: str, timeout: int = 20) -> List[str]:
    """
    Retrieve KEGG drug IDs linked to a KEGG human gene ID (e.g., 'hsa:1956').
    Returns list of 'Dxxxxx' identifiers.
    """
    url = f"https://rest.kegg.jp/link/drug/{kegg_gene_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "omnitarget/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return []

    drug_ids: List[str] = []
    for line in text.splitlines():
        # Expected format: hsa:1956\tdrug:Dxxxx
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        right = parts[1]
        if right.startswith("drug:"):
            did = right.replace("drug:", "").strip()
            if did and did not in drug_ids:
                drug_ids.append(did)
    return drug_ids


