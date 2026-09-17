"""Automates the pipeline: runs `run_pipeline.py` every 5 minutes (APScheduler).

The first run happens immediately after startup. `max_instances=1` prevents
overlapping runs; a failing run is logged but does not stop the scheduler.
"""
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler

from run_pipeline import main as run_pipeline

INTERVAL_MINUTES = 5


def run_job() -> None:
    print(f"\n[scheduler] pipeline run started at {datetime.now():%Y-%m-%d %H:%M:%S}")
    try:
        run_pipeline()
    except Exception as exc:  # keep the scheduler alive
        print(f"[scheduler] pipeline run FAILED: {exc}")
    else:
        print(f"[scheduler] pipeline run finished at {datetime.now():%Y-%m-%d %H:%M:%S}")


def _next_run_hint(job) -> str:
    """APScheduler <= 3.10 keeps the next fire time on the Job object;
    in 3.11 the attribute was removed, so compute it from the trigger."""
    nrt = getattr(job, "next_run_time", None)                     # APScheduler <= 3.10
    if nrt is None:
        try:                                                      # APScheduler 3.11+
            nrt = job.trigger.get_next_fire_time(None, datetime.now(timezone.utc))
        except Exception:
            return "unknown"
    return f"≈ {nrt} UTC"


def main() -> None:
    run_job()  # run immediately once at startup

    scheduler = BlockingScheduler(timezone="UTC")
    job = scheduler.add_job(
        run_job,
        trigger="interval",
        minutes=INTERVAL_MINUTES,
        max_instances=1,
        id="pmldl-pipeline",
        name="Data → Model → Deployment pipeline",
    )
    print(f"\n[scheduler] pipeline scheduled every {INTERVAL_MINUTES} minutes "
          f"(next run {_next_run_hint(job)}). Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n[scheduler] stopped.")


if __name__ == "__main__":
    main()