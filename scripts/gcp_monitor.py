#!/usr/bin/env python3
"""GCP monitor — writes gcp-status.json consumed by the dashboard.

Targets the single VM `evo-reward-gpu` (search across candidate zones) plus
the `evo-reward-ckpts` GCS bucket. Cost = VM compute + Cloud NAT (24/7) +
stopped-disk + GCS storage. Billing-actual is optional (BQ export).

Parallels scripts/status.py for VM + GCS queries — the two are expected to
report the same numbers. Difference: status.py also SSHes for live training
telemetry; this monitor can't SSH from CI, so the `training` field is null
unless a progress.json has been written to GCS (see docs/monitoring.md).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml


# --- schema -----------------------------------------------------------------

@dataclasses.dataclass
class VMInfo:
    name: str
    status: str                    # RUNNING | TERMINATED | STOPPED | MISSING | STAGING | ...
    zone: str | None
    machine_type: str | None
    provisioning: str              # SPOT | STANDARD | UNKNOWN
    created_at: str | None
    last_started_at: str | None
    runtime_hours_current: float | None   # since last_started_at (only if RUNNING)
    hourly_rate_usd: float | None
    estimated_current_run_usd: float | None


@dataclasses.dataclass
class CheckpointState:
    bucket: str
    count: int
    latest_step: int | None
    latest_age_hours: float | None
    total_gb: float | None


@dataclasses.dataclass
class CostBreakdown:
    # Live estimate pieces — all USD.
    compute_current_run: float        # vm runtime × hourly rate (only while RUNNING)
    nat_since_active: float | None    # ((now - nat_active_since) h) × $0.044
    storage_current: float            # bucket size × $/GB-month × fraction-of-month-so-far
    live_estimate_total: float
    # From BigQuery billing export, if configured.
    billing_actual_usd: float | None
    billing_as_of: str | None
    month_to_date_usd: float | None


@dataclasses.dataclass
class WorkerError:
    stage: str                         # vm | checkpoints | billing
    message: str


# --- utilities --------------------------------------------------------------

def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_between(start_iso: str | None, end: dt.datetime | None = None) -> float | None:
    if not start_iso:
        return None
    try:
        start = parse_iso(start_iso)
        end = end or dt.datetime.now(dt.timezone.utc)
        return round((end - start).total_seconds() / 3600.0, 4)
    except (ValueError, TypeError):
        return None


def price_for(
    pricing: dict[str, dict[str, float]],
    machine_type: str | None,
    spot: bool,
) -> float | None:
    if not machine_type:
        return None
    row = pricing.get(machine_type)
    if not row:
        return None
    return row.get("spot" if spot else "ondemand")


def month_fraction_elapsed() -> float:
    """How far through the current month we are — for pro-rata storage cost."""
    now = dt.datetime.now(dt.timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # Approximate: 30.4375 days/month avg. Fine for a live estimate.
    return (now - start).total_seconds() / (30.4375 * 86400)


# --- GCP calls (lazy imports so --dry-run works without creds) --------------

def load_credentials():
    """Default ADC only — single-project setup. No key-based fallback needed."""
    return None


def describe_vm(
    project_id: str,
    vm_name: str,
    candidate_zones: list[str],
    credentials,
    pricing: dict[str, dict[str, float]],
) -> VMInfo:
    from google.api_core.exceptions import NotFound
    from google.cloud import compute_v1

    client = compute_v1.InstancesClient(credentials=credentials)
    found = None
    for zone in candidate_zones:
        try:
            inst = client.get(project=project_id, zone=zone, instance=vm_name)
            found = (inst, zone)
            break
        except NotFound:
            continue

    if not found:
        return VMInfo(
            name=vm_name, status="MISSING", zone=None,
            machine_type=None, provisioning="UNKNOWN",
            created_at=None, last_started_at=None,
            runtime_hours_current=None,
            hourly_rate_usd=None, estimated_current_run_usd=None,
        )

    inst, zone = found
    machine_type = (inst.machine_type or "").rsplit("/", 1)[-1] or None
    provisioning = getattr(inst.scheduling, "provisioning_model", "") or "STANDARD"
    spot = provisioning == "SPOT"
    rate = price_for(pricing, machine_type, spot)
    started = inst.last_start_timestamp or inst.creation_timestamp
    runtime = hours_between(started) if inst.status == "RUNNING" else None
    est = round(runtime * rate, 4) if (runtime is not None and rate is not None) else None
    return VMInfo(
        name=inst.name,
        status=inst.status,
        zone=zone,
        machine_type=machine_type,
        provisioning=provisioning,
        created_at=inst.creation_timestamp,
        last_started_at=inst.last_start_timestamp,
        runtime_hours_current=runtime,
        hourly_rate_usd=rate,
        estimated_current_run_usd=est,
    )


def probe_checkpoints(project_id: str, bucket: str, credentials) -> CheckpointState:
    """Count step_*.npz files under results/ and report freshness + total size."""
    from google.cloud import storage
    client = storage.Client(project=project_id, credentials=credentials)
    b = client.bucket(bucket)
    step_re = re.compile(r"step_(\d+)\.npz$")
    latest_step = -1
    latest_time: dt.datetime | None = None
    count = 0
    total_bytes = 0
    # Prefix scan keeps the list call bounded
    for blob in b.list_blobs(prefix="results/"):
        total_bytes += blob.size or 0
        m = step_re.search(blob.name)
        if not m:
            continue
        count += 1
        step = int(m.group(1))
        if step > latest_step:
            latest_step = step
            latest_time = blob.updated
    age_hours: float | None = None
    if latest_time is not None:
        age_hours = round(
            (dt.datetime.now(dt.timezone.utc) - latest_time).total_seconds() / 3600.0,
            3,
        )
    return CheckpointState(
        bucket=bucket,
        count=count,
        latest_step=latest_step if latest_step >= 0 else None,
        latest_age_hours=age_hours,
        total_gb=round(total_bytes / 1e9, 4),
    )


def query_billing(
    credentials,
    project_id: str,
    billing_table: str | None,
    billing_account_id: str | None,
) -> tuple[float | None, str | None, float | None]:
    if not billing_table or not billing_account_id:
        return None, None, None
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, credentials=credentials)
    sql = f"""
    SELECT
      SUM(cost) AS total_cost,
      MAX(export_time) AS max_export_time,
      SUM(CASE WHEN DATE(usage_start_time) >= DATE_TRUNC(CURRENT_DATE(), MONTH)
               THEN cost ELSE 0 END) AS mtd_cost
    FROM `{billing_table}`
    WHERE billing_account_id = @billing_account_id
      AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 35 DAY)
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("billing_account_id", "STRING", billing_account_id),
            ]
        ),
    )
    row = next(iter(job.result()), None)
    if not row:
        return None, None, None
    as_of = row["max_export_time"].isoformat() if row["max_export_time"] else None
    return (
        float(row["total_cost"] or 0),
        as_of,
        float(row["mtd_cost"] or 0),
    )


# --- orchestration ----------------------------------------------------------

def run(config: dict[str, Any]) -> dict[str, Any]:
    project_id = config["project_id"]
    vm_name = config["vm_name"]
    zones = config["candidate_zones"]
    pricing = config.get("pricing", {})
    gcs_cfg = config.get("gcs", {})
    infra = config.get("infra_costs", {})
    billing_cfg = config.get("billing", {}) or {}

    errors: list[WorkerError] = []
    creds = load_credentials()

    # VM
    try:
        vm = describe_vm(project_id, vm_name, zones, creds, pricing)
    except Exception as e:
        errors.append(WorkerError("vm", str(e)))
        vm = VMInfo(name=vm_name, status="UNKNOWN", zone=None, machine_type=None,
                    provisioning="UNKNOWN", created_at=None, last_started_at=None,
                    runtime_hours_current=None,
                    hourly_rate_usd=None, estimated_current_run_usd=None)

    # GCS checkpoints
    try:
        ckpt = probe_checkpoints(project_id, gcs_cfg["bucket"], creds)
    except Exception as e:
        errors.append(WorkerError("checkpoints", str(e)))
        ckpt = CheckpointState(bucket=gcs_cfg.get("bucket", ""), count=0,
                               latest_step=None, latest_age_hours=None, total_gb=None)

    # Billing
    billing_actual = billing_as_of = mtd = None
    try:
        billing_actual, billing_as_of, mtd = query_billing(
            creds, project_id,
            billing_cfg.get("export_table"),
            billing_cfg.get("account_id"),
        )
    except Exception as e:
        errors.append(WorkerError("billing", str(e)))

    # Cost rollup
    compute_current = vm.estimated_current_run_usd or 0.0
    nat_since: float | None = None
    nat_start = infra.get("nat_active_since")
    nat_rate = infra.get("nat_hourly_usd", 0.0)
    if nat_start and nat_rate:
        hrs = hours_between(nat_start)
        if hrs is not None:
            nat_since = round(hrs * nat_rate, 2)
    storage_cur = 0.0
    if ckpt.total_gb is not None:
        storage_cur = round(
            ckpt.total_gb * gcs_cfg.get("storage_usd_per_gb_month", 0.020)
            * month_fraction_elapsed(),
            4,
        )
    live_total = round(
        compute_current + (nat_since or 0.0) + storage_cur,
        2,
    )

    costs = CostBreakdown(
        compute_current_run=round(compute_current, 4),
        nat_since_active=nat_since,
        storage_current=storage_cur,
        live_estimate_total=live_total,
        billing_actual_usd=billing_actual,
        billing_as_of=billing_as_of,
        month_to_date_usd=mtd,
    )

    return {
        "updated_at": iso_now(),
        "project_id": project_id,
        "vm": dataclasses.asdict(vm),
        "checkpoints": dataclasses.asdict(ckpt),
        "costs": dataclasses.asdict(costs),
        "training": None,   # future: populated from gs://.../progress.json
        "errors": [dataclasses.asdict(e) for e in errors],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/gcp_monitor_config.yaml")
    parser.add_argument("--out", default="gcp-status.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Emit a synthetic payload without calling GCP. "
                             "Useful for local UI development.")
    args = parser.parse_args()

    if args.dry_run:
        payload = _synthetic_payload()
    else:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        payload = run(config)

    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} "
          f"(vm={payload['vm']['status']}, "
          f"ckpt={payload['checkpoints']['count']}, "
          f"errors={len(payload['errors'])})", file=sys.stderr)
    return 0


def _synthetic_payload() -> dict[str, Any]:
    return {
        "updated_at": iso_now(),
        "project_id": "evo-reward",
        "vm": {
            "name": "evo-reward-gpu", "status": "RUNNING", "zone": "us-central1-a",
            "machine_type": "g2-standard-8", "provisioning": "SPOT",
            "created_at": "2026-04-18T12:00:00+00:00",
            "last_started_at": "2026-04-20T02:00:00+00:00",
            "runtime_hours_current": 6.5,
            "hourly_rate_usd": 0.28, "estimated_current_run_usd": 1.82,
        },
        "checkpoints": {
            "bucket": "evo-reward-ckpts", "count": 41, "latest_step": 410_000,
            "latest_age_hours": 0.08, "total_gb": 2.31,
        },
        "costs": {
            "compute_current_run": 1.82,
            "nat_since_active": 10.56,   # ~240h × $0.044
            "storage_current": 0.03,
            "live_estimate_total": 12.41,
            "billing_actual_usd": 8.91,
            "billing_as_of": "2026-04-19T00:00:00+00:00",
            "month_to_date_usd": 41.30,
        },
        "training": None,
        "errors": [],
    }


if __name__ == "__main__":
    sys.exit(main())
