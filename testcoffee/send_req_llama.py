import base64
from together import Together
import ast

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

class LlamaPrompt:
    def __init__(self, api_key="0e05bfaf5d5aaa8ed5fa476ff9aa303bb6667fa56c755db8ebee6aa4224ee1ae", image_path="image.png"):
        self.client = Together(api_key=api_key)
        self.image_path = image_path
        
    def prompt_llama(self, items):
        """
        Process a list of items using LLama model to categorize them into bags.
        
        Args:
            items: List of strings representing item names
            
        Returns:
            List of tuples in the format [("object_name", "bag_name"), ...]
        """
        # Encode the image
        b64 = encode_image(self.image_path)
        
        # Build the prompt with the provided items
        items_str = ", ".join(items)
        input_text = f"""
You have the objects: {items_str}.
You have two bags: 'food' and 'non-food'.
Determine an order to place each object into one of the bags so nothing gets damaged.
Respond **only** with a Python code block containing a single list of tuples:
```python
[("object_name", "bag_name"), ...]
"""

        # Call the API
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }]
        )

        # Parse the response
        output = response.choices[0].message.content
        python_index = output.lower().find("python")
        if python_index != -1:
            # Skip past the word "python"
            start_index = python_index + len("python")
            extracted_text = output[start_index:]
            code_block_index = extracted_text.find("```")
            if code_block_index != -1:
                parsed_output = extracted_text[:code_block_index]

                # Convert string representation to actual list of tuples
                try:
                    tuples_list = ast.literal_eval(parsed_output.strip())
                    return tuples_list
                except (SyntaxError, ValueError) as e:
                    print(f"Error converting to list of tuples: {e}")
                    return []
        
        return []

if __name__ == "__main__":
    llama = LlamaPrompt()
    result = llama.prompt_llama(["banana", "apple", "can", "cup", "orange", "bottle"])
    print(result)