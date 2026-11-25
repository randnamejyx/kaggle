"""Reliably submit this project's predictions to Kaggle.

The Kaggle CLI can wait forever at 0% on this server because
www.googleapis.com resolves to unreachable addresses. This script validates
the CSV, obtains an official Kaggle upload URL, finds a reachable Google
frontend, uploads with explicit timeouts, creates the submission, and waits
for the score.
"""

import argparse
import csv
import getpass
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

from kagglesdk import KaggleClient, KaggleEnv
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateSubmissionRequest,
    ApiListSubmissionsRequest,
    ApiStartSubmissionUploadRequest,
)


def read_ids(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames is None or "id" not in rows.fieldnames:
            raise ValueError(f"{path}: missing id column")
        return [row["id"] for row in rows]


def validate_submission(submission_path, test_path):
    with submission_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["id", "label"]:
            raise ValueError(
                f"{submission_path}: expected columns ['id', 'label'], "
                f"found {rows.fieldnames}"
            )
        submission_rows = list(rows)

    test_ids = read_ids(test_path)
    submission_ids = [row["id"] for row in submission_rows]
    if submission_ids != test_ids:
        raise ValueError(
            f"{submission_path}: IDs/order must exactly match {test_path}; "
            "do not use sample_submission.csv IDs"
        )
    if len(set(submission_ids)) != len(submission_ids):
        raise ValueError(f"{submission_path}: duplicate IDs")
    invalid_labels = sorted({row["label"] for row in submission_rows} - {"0", "1"})
    if invalid_labels:
        raise ValueError(f"{submission_path}: invalid labels {invalid_labels}")
    print(f"CSV validated: {len(submission_rows)} rows; IDs match {test_path}")


def configure_token(token_file):
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token and token_file.is_file():
        token = token_file.read_text(encoding="utf-8").strip()
        print(f"Using saved token from {token_file}")
    if not token:
        token = getpass.getpass("Kaggle API token (hidden): ").strip()
    if not token.startswith("KGAT_"):
        raise ValueError("Expected a current Kaggle token beginning with KGAT_")
    os.environ["KAGGLE_API_TOKEN"] = token


def reachable_google_ip(upload_host):
    candidates = []
    for lookup_host in ("storage.googleapis.com", upload_host):
        try:
            addresses = socket.getaddrinfo(lookup_host, 443, socket.AF_INET)
        except socket.gaierror:
            continue
        for address in addresses:
            ip = address[4][0]
            if ip not in candidates:
                candidates.append(ip)

    for ip in candidates:
        probe = subprocess.run(
            [
                "curl", "--silent", "--show-error",
                "--resolve", f"{upload_host}:443:{ip}",
                "--connect-timeout", "3", "--max-time", "6",
                "--output", "/dev/null", "--write-out", "%{http_code}",
                f"https://{upload_host}/",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip() != "000":
            print(f"Reachable upload frontend: {ip}")
            return ip
    raise RuntimeError(f"No reachable IP found for {upload_host}")


def upload_file(url, path, ip):
    host = urlsplit(url).hostname
    if not host:
        raise ValueError("Kaggle returned an invalid upload URL")
    escaped_url = url.replace("\\", "\\\\").replace('"', '\\"')
    escaped_path = str(path).replace("\\", "\\\\").replace('"', '\\"')
    config = "\n".join([
        f'url = "{escaped_url}"',
        f'upload-file = "{escaped_path}"',
        f'resolve = "{host}:443:{ip}"',
        "silent",
        "show-error",
        "connect-timeout = 5",
        "max-time = 90",
        "retry = 3",
        "retry-delay = 1",
        "retry-all-errors",
        'output = "/dev/null"',
        'write-out = "%{http_code}"',
        "",
    ])
    result = subprocess.run(
        ["curl", "--config", "-"],
        input=config,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    status = result.stdout.strip()[-3:]
    if result.returncode != 0 or status not in {"200", "201"}:
        detail = result.stderr.strip().replace(url, "[upload URL]")
        raise RuntimeError(
            f"upload failed: curl={result.returncode}, HTTP={status}: {detail[:500]}"
        )
    print(f"Upload complete: HTTP {status}")


def list_submissions(client, competition):
    request = ApiListSubmissionsRequest()
    request.competition_name = competition
    return client.competitions.competition_api_client.list_submissions(request).submissions


def submit(args):
    submission_path = args.file.resolve()
    validate_submission(submission_path, args.test.resolve())
    configure_token(args.token_file.expanduser())

    with KaggleClient(env=KaggleEnv.PROD) as client:
        start = ApiStartSubmissionUploadRequest()
        start.competition_name = args.competition
        start.file_name = submission_path.name
        start.content_length = submission_path.stat().st_size
        start.last_modified_epoch_seconds = int(submission_path.stat().st_mtime)
        slot = client.competitions.competition_api_client.start_submission_upload(start)
        host = urlsplit(slot.create_url).hostname
        print("Kaggle upload slot created")
        ip = reachable_google_ip(host)
        upload_file(slot.create_url, submission_path, ip)

        create = ApiCreateSubmissionRequest()
        create.competition_name = args.competition
        create.blob_file_tokens = slot.token
        create.submission_description = args.message
        result = client.competitions.competition_api_client.create_submission(create)
        print(result.message)

        deadline = time.monotonic() + args.wait
        while True:
            submissions = list_submissions(client, args.competition)
            latest = submissions[0]
            status = str(latest.status).split(".")[-1]
            print(
                f"ref={latest.ref} status={status} "
                f"public={latest.public_score or '-'} private={latest.private_score or '-'}"
            )
            if status not in {"PENDING", "RUNNING"}:
                if status != "COMPLETE":
                    raise RuntimeError(latest.error_description or f"submission ended as {status}")
                return
            if time.monotonic() >= deadline:
                print("Scoring is still pending; query it later with the Kaggle CLI.")
                return
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default="hate-speech-detection-in-english")
    parser.add_argument("--file", type=Path, default=Path("artifacts/submission.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path("~/.config/kaggle/token.secret"),
        help="Saved KGAT token file (default: ~/.config/kaggle/token.secret)",
    )
    parser.add_argument(
        "--message", default="word-char TF-IDF logistic regression; validated test IDs"
    )
    parser.add_argument("--wait", type=int, default=120, help="Scoring wait in seconds")
    args = parser.parse_args()
    try:
        submit(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Submission failed: {error}\n")


if __name__ == "__main__":
    main()
