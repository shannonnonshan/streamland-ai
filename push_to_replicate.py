"""
Push model to Replicate using API
"""

import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

# Get credentials
api_token = os.getenv("REPLICATE_API_TOKEN")
username = os.getenv("REPLICATE_USERNAME")
model_name = "streamland-whisper"

if not api_token or not username:
    print("❌ Missing REPLICATE_API_TOKEN or REPLICATE_USERNAME in .env")
    sys.exit(1)

print(f"✓ Username: {username}")
print(f"✓ Model: {model_name}")
print(f"✓ Pushing to: r8.im/{username}/{model_name}")

# Create the model on Replicate if it doesn't exist
create_cmd = f"""
curl -X POST https://api.replicate.com/v1/models \\
  -H "Authorization: Token {api_token}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "name": "{model_name}",
    "owner": "{username}",
    "description": "Streamland Whisper - Audio Transcription Model",
    "visibility": "public"
  }}'
"""

print("\n📦 Creating model on Replicate...")
result = subprocess.run(create_cmd, shell=True, capture_output=True, text=True)
print(result.stdout)
if result.stderr and "already exists" not in result.stderr:
    print(result.stderr)

print("\n🚀 Model created/verified on Replicate!")
print(f"📍 Access at: https://replicate.com/{username}/{model_name}")
print(f"🔗 API: https://api.replicate.com/v1/models/{username}/{model_name}")

# Save API token for future use
env_content = f"""
# Replicate API
REPLICATE_API_TOKEN={api_token}
REPLICATE_USERNAME={username}
REPLICATE_MODEL={model_name}
REPLICATE_API_URL=https://api.replicate.com/v1/models/{username}/{model_name}
"""

with open(".replicate", "w") as f:
    f.write(env_content)

print("\n✅ Configuration saved to .replicate")
print(f"\n💡 To use the model via API:")
print(f"""
import replicate

output = replicate.run(
    "{username}/{model_name}:latest",
    input={{
        "audio": "https://example.com/audio.wav",
        "language": "en",
        "task": "transcribe"
    }}
)
print(output)
""")
