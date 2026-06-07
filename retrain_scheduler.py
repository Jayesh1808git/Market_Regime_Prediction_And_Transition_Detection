"""
retrain_scheduler.py
Runs train.py every Sunday at 2am IST.
Pure Python — no cron, no supercronic needed.
"""
import subprocess
import time
import logging
from datetime import datetime, timezone, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

def should_retrain(now: datetime) -> bool:
    # Every Sunday at 02:00 IST
    return now.weekday() == 6 and now.hour == 2 and now.minute == 0

def run_retrain():
    logger.info("Starting scheduled retrain...")
    try:
        result = subprocess.run(
            ["python", "train.py", "--refresh-data"],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        if result.returncode == 0:
            logger.info("Retrain completed successfully.")
        else:
            logger.error(f"Retrain failed:\n{result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("Retrain timed out after 1 hour.")
    except Exception as e:
        logger.error(f"Retrain error: {e}")

if __name__ == "__main__":
    logger.info("Retrain scheduler started. Waiting for Sunday 02:00 IST...")
    last_run_date = None

    while True:
        now = datetime.now(IST)

        if should_retrain(now) and last_run_date != now.date():
            last_run_date = now.date()
            run_retrain()

        # Sleep 55 seconds — checks time every minute
        time.sleep(55)