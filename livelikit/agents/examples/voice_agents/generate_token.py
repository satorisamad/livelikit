import os
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

# Load environment variables from .env file
load_dotenv()

# Get LiveKit credentials from environment
api_key = os.getenv("LIVEKIT_API_KEY")
api_secret = os.getenv("LIVEKIT_API_SECRET")

if not api_key or not api_secret:
    raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in the .env file")

# Define the room and participant details
room_name = "test-room"
participant_identity = "test-user"

# Create an access token
token = AccessToken(api_key, api_secret)\
    .with_identity(participant_identity)\
    .with_name("Test User")\
    .with_grants(VideoGrants(
        room_join=True,
        room=room_name,
    )).to_jwt()

with open("token.txt", "w") as f:
    f.write(token)

print("Token has been written to token.txt")
