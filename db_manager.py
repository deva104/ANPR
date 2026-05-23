from supabase import create_client, Client

# ==========================================
# ⚙️ CONFIGURATION (Your live Supabase project)
# ==========================================
SUPABASE_URL = "https://miqdestfirvcfmqlclqc.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1pcWRlc3RmaXJ2Y2ZtcWxjbHFjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NTcwMzQ3MCwiZXhwIjoyMDkxMjc5NDcwfQ.TxNsVFMXAGdAj7HNlJTL6LrGcFyV-TcFo3i7vHA-XmU"


class PlateDatabase:
    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def is_allowed(self, plate_number):
        """Returns (True, owner_name) if plate is registered & authorized, (False, None) if not."""
        plate = plate_number.upper().strip().replace(" ", "")
        try:
            # Query the vehicles table in your live Supabase database
            response = (
                self.client.table("vehicles")
                .select("status, resident_id")
                .eq("plate_number", plate)
                .execute()
            )
            if response.data:
                vehicle = response.data[0]
                # Check if vehicle status is set to 'authorized'
                if vehicle.get("status") == "authorized":
                    res_id = vehicle.get("resident_id")
                    if res_id:
                        # Find the resident profile ID
                        res_resp = (
                            self.client.table("residents")
                            .select("profile_id")
                            .eq("id", res_id)
                            .execute()
                        )
                        if res_resp.data and res_resp.data[0].get("profile_id"):
                            prof_id = res_resp.data[0].get("profile_id")
                            # Retrieve the resident's actual full name from profiles
                            prof_resp = (
                                self.client.table("profiles")
                                .select("full_name")
                                .eq("id", prof_id)
                                .execute()
                            )
                            if prof_resp.data:
                                return True, prof_resp.data[0].get("full_name")
                    return True, "Authorized Resident"
        except Exception as e:
            print(f"Error querying vehicle authorization on Supabase: {e}")
            
        return False, None

    def log_detection(self, plate_number, allowed, owner_name):
        """Log the vehicle detection log directly to Supabase entry_logs."""
        plate = plate_number.upper().strip().replace(" ", "")
        decision = "authorized" if allowed else "unknown"
        reason = f"Authorized access for {owner_name or 'Resident'}" if allowed else "Unknown vehicle detected at gate"
        
        try:
            # Insert scan directly into the entry_logs table
            self.client.table("entry_logs").insert(
                {
                    "plate_number_raw": plate_number.upper(),
                    "plate_number_normalized": plate,
                    "ocr_confidence": 100.0,
                    "source": "camera",
                    "decision": decision,
                    "reason": reason,
                }
            ).execute()
            print(f"Logged detection for {plate} (Allowed: {allowed}) successfully to Supabase.")
        except Exception as e:
            print(f"Error logging detection to Supabase: {e}")

    def add_plate(self, plate_number, owner_name):
        """Mock plate registration to maintain backward compatibility."""
        pass
