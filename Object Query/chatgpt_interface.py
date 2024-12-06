# Helpful video: https://www.youtube.com/watch?v=lJJkBaO15Po
# Assuming the we have a label for the object
import logging
import os
from openai import OpenAI
from object_query import ObjectQuery  

logging.basicConfig(level=logging.INFO)

class ChatGPTInterface:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key for OpenAI is not set.")
        self.client = OpenAI(api_key=self.api_key)

    def get_gpt_response(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            content = response.choices[0].message.content.strip().lower()
            logging.info(f"Received response: {content}")
            
            if "high" in content:
                return "high"
            elif "medium" in content:
                return "medium"
            elif "low" in content:
                return "low"
            try:
                numeric_response = float(content)
                if 1 <= numeric_response <= 10:
                    return numeric_response
                else:
                    print(f"Unexpected numeric response: {content}")
                    return None
            except ValueError:
                print(f"Unexpected response: {content}")
                return None

        except Exception as e:
            logging.error(f"API call failed: {e}")
            return None


