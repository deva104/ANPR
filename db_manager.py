from supabase import create_client, Client

SUPABASE_URL = "https://dzjqcozucensltywpcin.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR6anFjb3p1Y2Vuc2x0eXdwY2luIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU2NzQyMzksImV4cCI6MjA5MTI1MDIzOX0.-ixetpzJataSFH_0ahWCSxX3OahBqvv-lXNa-iuzals"


class PlateDatabase:
    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def is_allowed(self, plate_number):
        """Returns (True, owner_name) if plate allowed, (False, None) if not."""
        plate = plate_number.upper().strip().replace(" ", "")
        response = (
            self.client.table("allowed_plates")
            .select("owner_name")
            .eq("plate_number", plate)
            .execute()
        )
        if response.data:
            return True, response.data[0]["owner_name"]
        return False, None

    def log_detection(self, plate_number, allowed, owner_name):
        """Log every detection attempt to Supabase."""
        plate = plate_number.upper().strip().replace(" ", "")
        self.client.table("detection_log").insert(
            {
                "plate_number": plate,
                "allowed": allowed,
                "owner_name": owner_name,
            }
        ).execute()

    def add_plate(self, plate_number, owner_name):
        """Add a new allowed plate."""
        plate = plate_number.upper().strip().replace(" ", "")
        self.client.table("allowed_plates").insert(
            {
                "plate_number": plate,
                "owner_name": owner_name,
            }
        ).execute()
