import os
from supabase import create_client

SUPABASE_URL = "https://owcildvdeisxhskkpecw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im93Y2lsZHZkZWlzeGhza2twZWN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgzMzM0NTgsImV4cCI6MjA5MzkwOTQ1OH0.9jz6oc8vIW7lfohvqOF6tsxv9IGadWJFl8HS3pi2C34"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Test insert into vehicles
result = supabase.table("vehicles").insert({
    "plate_number": "TEST01",
    "owner_name": "Test User",
    "vehicle_type": "owner"
}).execute()
print("Insert result:", result.data)

# Test read
result = supabase.table("vehicles").select("*").execute()
print("All vehicles:", result.data)

# Test delete
result = supabase.table("vehicles").delete().eq(
    "plate_number", "TEST01"
).execute()
print("Deleted test row")

print("Supabase connection working correctly")
