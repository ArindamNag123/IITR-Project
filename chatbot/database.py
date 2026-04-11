import os
import redis
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --- FalkorDB Connection ---
try:
    falkor_db = redis.Redis(
        host=os.getenv("FALKORDB_HOST"),
        port=int(os.getenv("FALKORDB_PORT", 6379)),
        username=os.getenv("FALKORDB_USER"),
        password=os.getenv("FALKORDB_PASS"),
        decode_responses=True
    )
    # Simple ping to verify connection
    falkor_db.ping()
    logger.info("✅ Connected to FalkorDB")
except Exception as e:
    logger.error(f"❌ FalkorDB Connection Error: {e}")
    falkor_db = None

def get_db():
    return falkor_db
