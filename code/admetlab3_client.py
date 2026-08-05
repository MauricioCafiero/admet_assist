#!/usr/bin/env python3
"""
admetlab3_client.py — fetch ADMET properties for molecules from SMILES via ADMETlab 3.0.

Usage:
    python3 admetlab3_client.py --input molecules.smi --output admet.csv
    python3 admetlab3_client.py --input data.csv --smiles-column smiles --output admet.csv
    python3 admetlab3_client.py --smiles "CCO" "CC(=O)Oc1ccccc1C(=O)O" --output admet.csv

Notes on the API (verified 2026-08-04 against https://admetlab3.scbdd.com):
  * The documented JSON REST endpoint /api/single/admet is currently BROKEN server-side
    (KeyError "['BSEP'] not in index", HTTP 500 for every request). The batch /api/admet
    was removed (404) and /api/uploadfile returns null. So this client uses the *web*
    batch path that the site itself uses, which is fully functional and returns all
    119 ADMET endpoints as a CSV:
        1. GET  /server/screening           -> scrape a csrfmiddlewaretoken (no cookie needed)
        2. POST /server/screeningCal        -> multipart form (uploadfile=@smiles.txt) -> 302
        3. GET  /static/results/csv/<taskid>.csv  -> poll until ready, then download
  * Accepted upload extensions: .txt, .csv, .sdf (.smi is rejected as "Invalid file type").
  * Prediction is asynchronous; the CSV appears a few seconds after the 302.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Sequence

import requests

BASE_URL = "https://admetlab3.scbdd.com"
SCREENING_URL = f"{BASE_URL}/server/screening"
SUBMIT_URL = f"{BASE_URL}/server/screeningCal"
RESULT_PAGE_URL = f"{BASE_URL}/server/result/"  # /server/result/<taskid>
CSV_URL = f"{BASE_URL}/static/results/csv/"    # .../<taskid>.csv

# Regex to pull the CSRF token out of the screening form HTML.
_CSRF_RE = re.compile(r'csrfmiddlewaretoken" value="([^"]+)"')


class AdmetlabError(RuntimeError):
    """Raised when the ADMETlab 3.0 service rejects or fails a request."""


class AdmetlabClient:
    """Thin client for the ADMETlab 3.0 batch prediction (web) flow."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = 60.0,
        poll_interval: float = 3.0,
        poll_timeout: float = 600.0,
        max_retries: int = 3,
        backoff: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "admetlab3-client/1.0 (+https://admetlab3.scbdd.com)"}
        )

    # -- internals ---------------------------------------------------------

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """POST/GET with retry on 429/5xx/timeouts using exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise AdmetlabError(f"HTTP {resp.status_code} from {url}")
                return resp
            except (requests.RequestException, AdmetlabError) as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.backoff * (2 ** (attempt - 1)))
        raise AdmetlabError(f"request to {url} failed after {self.max_retries} tries: {last_exc}")

    def _get_csrf(self) -> str:
        resp = self._request("GET", f"{self.base_url}/server/screening")
        m = _CSRF_RE.search(resp.text)
        if not m:
            raise AdmetlabError("could not find csrfmiddlewaretoken on screening page")
        return m.group(1)

    def _submit(self, smiles_list: Sequence[str]) -> str:
        """Upload SMILES as a .txt file; return the taskid from the 302 Location."""
        token = self._get_csrf()
        # Server rejects .smi ("Invalid file type"); .txt works and is one-SMILES-per-line.
        text = "\n".join(s.strip() for s in smiles_list if s and s.strip()) + "\n"
        if not text.strip():
            raise AdmetlabError("no non-empty SMILES to submit")
        # requests sends this as a multipart part with the filename "smiles.txt".
        files = {"uploadfile": ("smiles.txt", text, "text/plain")}
        data = {
            "csrfmiddlewaretoken": token,
            "method": "1",          # 1 = file upload (batch); 2 = single textarea molecule
            "is_example": "0",
        }
        resp = self._request("POST", f"{self.base_url}/server/screeningCal",
                             data=data, files=files, allow_redirects=False)
        # The view is @csrf_exempt; a successful upload yields 302 -> /server/result/<taskid>.
        # A 200 means the form was re-rendered (e.g. invalid file type / empty input).
        if resp.status_code != 302:
            snippet = re.sub(r"\s+", " ", resp.text)[:300]
            raise AdmetlabError(
                f"submission did not redirect (HTTP {resp.status_code}); "
                f"check input. Body: {snippet!r}"
            )
        location = resp.headers.get("Location", "")
        taskid = location.rsplit("/", 1)[-1]
        if not taskid:
            raise AdmetlabError(f"no taskid in redirect Location header: {location!r}")
        return taskid

    def _wait_for_csv(self, taskid: str) -> str:
        """Poll the CSV URL until it returns 200; return the CSV text."""
        url = f"{self.base_url}/static/results/csv/{taskid}.csv"
        deadline = time.monotonic() + self.poll_timeout
        while time.monotonic() < deadline:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200 and resp.content:
                return resp.text
            time.sleep(self.poll_interval)
        raise AdmetlabError(
            f"CSV for task {taskid!r} not ready within {self.poll_timeout:.0f}s "
            f"(last HTTP {resp.status_code if 'resp' in dir() else '?'})."
        )

    # -- public API --------------------------------------------------------

    def predict(self, smiles_list: Sequence[str]) -> "list[dict[str, str]]":
        """Submit a batch of SMILES and return one dict per molecule (123 columns)."""
        unique = list(dict.fromkeys(s.strip() for s in smiles_list if s and s.strip()))
        if not unique:
            raise AdmetlabError("no SMILES provided")
        taskid = self._submit(unique)
        csv_text = self._wait_for_csv(taskid)
        return parse_admet_csv(csv_text)

    def predict_to_dataframe(self, smiles_list: Sequence[str]):
        """Like predict() but returns a pandas DataFrame. Requires pandas."""
        import pandas as pd
        rows = self.predict(smiles_list)
        return pd.DataFrame(rows)


def parse_admet_csv(csv_text: str) -> list[dict[str, str]]:
    """Parse the ADMETlab CSV (handles embedded newlines/quotes) into row dicts."""
    reader = csv.DictReader(io.StringIO(csv_text))
    return [dict(row) for row in reader]


# -- input loading ---------------------------------------------------------

def load_smiles(path: str | os.PathLike, smiles_column: str | None = None) -> list[str]:
    """Load SMILES from a .smi/.txt (one per line) or a .csv (column of SMILES)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".smi", ".txt", ".tsv"):
        with open(path) as f:
            return [line.split()[0] for line in f if line.strip() and not line.startswith("#")]
    if suffix == ".csv":
        import csv as _csv
        with open(path, newline="") as f:
            reader = _csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"{path}: empty CSV")
            col = smiles_column or next(
                (c for c in reader.fieldnames if c.lower() in ("smiles", "smile", "canonical_smiles")),
                reader.fieldnames[0],
            )
            return [row[col] for row in reader if row.get(col)]
    if suffix in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(path)
        col = smiles_column or next(
            (c for c in df.columns if str(c).lower() in ("smiles", "smile", "canonical_smiles")),
            df.columns[0],
        )
        return [str(v) for v in df[col].tolist() if v]
    raise ValueError(f"unsupported input extension: {suffix}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch ADMET properties from ADMETlab 3.0.")
    p.add_argument("--input", help="path to .smi/.txt/.csv/.xlsx with SMILES")
    p.add_argument("--smiles-column", help="column name for SMILES in a .csv/.xlsx")
    p.add_argument("--smiles", nargs="*", help="one or more SMILES strings on the CLI")
    p.add_argument("--output", default="admet_results.csv", help="output CSV path")
    p.add_argument("--base-url", default=BASE_URL, help="ADMETlab base URL")
    p.add_argument("--poll-interval", type=float, default=3.0, help="CSV poll interval (s)")
    p.add_argument("--poll-timeout", type=float, default=600.0, help="max wait for results (s)")
    p.add_argument("--max-retries", type=int, default=3, help="HTTP retries on 429/5xx")
    args = p.parse_args(argv)

    if args.input:
        smiles = load_smiles(args.input, args.smiles_column)
    elif args.smiles:
        smiles = list(args.smiles)
    else:
        p.error("provide --input or --smiles")

    print(f"Submitting {len(smiles)} molecule(s) to ADMETlab 3.0 ...", file=sys.stderr)
    client = AdmetlabClient(
        base_url=args.base_url,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
        max_retries=args.max_retries,
    )

    try:
        rows = client.predict(smiles)
    except AdmetlabError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Write CSV (preserves ADMETlab's native column order/headers).
    if rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"Wrote {len(rows)} molecule(s) x {len(rows[0])} columns -> {args.output}",
            file=sys.stderr,
        )
    else:
        print("WARNING: no rows returned", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())