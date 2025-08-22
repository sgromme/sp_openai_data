from openai import OpenAI
from openai import APIStatusError, APIConnectionError, RateLimitError, OpenAIError
import os, time
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("OpenAI API Key loaded:", api_key is not None)

client = OpenAI(api_key=api_key)

prompt = (
    "Generate a CSV-style table of the weekly demand forecast for 3 products "
    "over 12 weeks with random but realistic values"
)

def call_with_retries(max_retries=4, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(model="gpt-5", input=prompt)

            # Prefer the unified text helper if available
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text

            # Fallbacks for other shapes (older/newer SDKs)
            if getattr(resp, "output", None):
                parts = []
                for item in resp.output or []:
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) in ("output_text", "text"):
                            parts.append(getattr(c, "text", ""))
                if parts:
                    return "\n".join(parts)

            if getattr(resp, "choices", None):
                # Legacy/chat-like shape
                return resp.choices[0].message.content

            # If we reach here, we didn’t find text in any expected place
            raise ValueError("No text found in response payload.")

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
        except APIConnectionError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
        except APIStatusError as e:
            # 4xx/5xx from the API – print and stop (usually not retryable except 429/5xx)
            raise RuntimeError(f"API status error {e.status_code}: {e.response}") from e
        except OpenAIError as e:
            # Generic SDK error – surface it
            raise RuntimeError(f"OpenAI error: {e}") from e

print("Response from OpenAI:")
try:
    text = call_with_retries()
    print(text)
except Exception as e:
    print(f"⚠️ Error: {e}")