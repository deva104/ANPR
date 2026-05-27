import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def _today_start_utc() -> datetime:
    now = _utc_now()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def get_all_vehicles() -> List[dict]:
    result = (
        supabase.table("vehicles")
        .select("*")
        .order("added_on", desc=True)
        .execute()
    )
    return result.data or []


def get_vehicle(plate_number: str) -> Optional[dict]:
    plate = plate_number.upper().strip()
    result = (
        supabase.table("vehicles")
        .select("*")
        .eq("plate_number", plate)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def add_vehicle(
    plate_number: str,
    owner_name: str,
    vehicle_type: str,
    valid_from=None,
    valid_until=None,
    purpose=None,
) -> dict:
    row = {
        "plate_number": plate_number.upper().strip(),
        "owner_name": owner_name,
        "vehicle_type": vehicle_type,
        "valid_from": valid_from,
        "valid_until": valid_until,
        "purpose": purpose,
    }
    result = supabase.table("vehicles").insert(row).execute()
    if not result.data:
        raise RuntimeError("Failed to insert vehicle")
    return result.data[0]


def update_vehicle(
    plate_number: str,
    owner_name: str,
    vehicle_type: str,
    valid_from=None,
    valid_until=None,
    purpose=None,
) -> dict:
    plate = plate_number.upper().strip()
    updates = {
        "owner_name": owner_name,
        "vehicle_type": vehicle_type,
        "valid_from": valid_from,
        "valid_until": valid_until,
        "purpose": purpose,
    }
    result = (
        supabase.table("vehicles")
        .update(updates)
        .eq("plate_number", plate)
        .execute()
    )
    if not result.data:
        raise RuntimeError(f"Vehicle not found: {plate}")
    return result.data[0]


def delete_vehicle(plate_number: str) -> bool:
    plate = plate_number.upper().strip()
    result = (
        supabase.table("vehicles")
        .delete()
        .eq("plate_number", plate)
        .execute()
    )
    return bool(result.data)


def log_access(
    plate_number: str,
    owner_name: str,
    vehicle_type: Optional[str],
    access_granted: bool,
    denial_reason: Optional[str] = None,
) -> dict:
    row = {
        "plate_number": plate_number.upper().strip(),
        "owner_name": owner_name,
        "vehicle_type": vehicle_type,
        "access_granted": access_granted,
        "denial_reason": denial_reason,
        "detected_on": _utc_now().isoformat(),
    }
    result = supabase.table("access_logs").insert(row).execute()
    if not result.data:
        raise RuntimeError("Failed to insert access log")
    return result.data[0]


def get_logs(limit: int = 50) -> List[dict]:
    result = (
        supabase.table("access_logs")
        .select("*")
        .order("detected_on", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def search_logs(plate_number: str) -> List[dict]:
    pattern = f"%{plate_number.strip()}%"
    result = (
        supabase.table("access_logs")
        .select("*")
        .ilike("plate_number", pattern)
        .order("detected_on", desc=True)
        .execute()
    )
    return result.data or []


def get_today_stats() -> dict:
    today_start = _today_start_utc().isoformat()
    now = _utc_now()
    now_iso = now.isoformat()
    soon_iso = (now + timedelta(days=2)).isoformat()

    allowed_res = (
        supabase.table("access_logs")
        .select("*", count="exact")
        .eq("access_granted", True)
        .gte("detected_on", today_start)
        .execute()
    )
    denied_res = (
        supabase.table("access_logs")
        .select("*", count="exact")
        .eq("access_granted", False)
        .gte("detected_on", today_start)
        .execute()
    )

    vehicles = supabase.table("vehicles").select("*").execute().data or []
    active_passes = 0
    expiring_soon = 0
    visitor_types = {"visitor", "relative"}

    for v in vehicles:
        vtype = (v.get("vehicle_type") or "").lower()
        if vtype not in visitor_types:
            continue
        valid_until = _parse_dt(v.get("valid_until"))
        if valid_until is None:
            continue
        if valid_until >= now:
            active_passes += 1
            if valid_until <= now + timedelta(days=2):
                expiring_soon += 1

    return {
        "allowed": allowed_res.count or 0,
        "denied": denied_res.count or 0,
        "active_passes": active_passes,
        "expiring_soon": expiring_soon,
    }


def verify_plate(plate_number: str) -> dict:
    plate = plate_number.upper().strip()
    vehicle = get_vehicle(plate)

    if vehicle is None:
        log_access(plate, "Unknown", None, False, "not_registered")
        return {
            "granted": False,
            "reason": "not_registered",
            "plate_number": plate,
            "owner_name": "Unknown",
            "vehicle_type": None,
        }

    owner_name = vehicle.get("owner_name", "Unknown")
    vehicle_type = vehicle.get("vehicle_type")
    vtype = (vehicle_type or "").lower()

    if vtype in ("owner", "renter"):
        log_access(plate, owner_name, vehicle_type, True, None)
        return {
            "granted": True,
            "reason": None,
            "plate_number": plate,
            "owner_name": owner_name,
            "vehicle_type": vehicle_type,
        }

    if vtype in ("visitor", "relative"):
        now = _utc_now()
        valid_until = _parse_dt(vehicle.get("valid_until"))
        if valid_until is None or valid_until < now:
            log_access(plate, owner_name, vehicle_type, False, "expired")
            return {
                "granted": False,
                "reason": "expired",
                "plate_number": plate,
                "owner_name": owner_name,
                "vehicle_type": vehicle_type,
                "valid_until": vehicle.get("valid_until"),
            }
        log_access(plate, owner_name, vehicle_type, True, None)
        return {
            "granted": True,
            "reason": None,
            "plate_number": plate,
            "owner_name": owner_name,
            "vehicle_type": vehicle_type,
            "valid_until": vehicle.get("valid_until"),
        }

    log_access(plate, owner_name, vehicle_type, False, "not_registered")
    return {
        "granted": False,
        "reason": "not_registered",
        "plate_number": plate,
        "owner_name": owner_name,
        "vehicle_type": vehicle_type,
    }
