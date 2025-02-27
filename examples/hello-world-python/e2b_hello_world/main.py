import base64
from dotenv import load_dotenv
from anthropic import Anthropic
from typing import List, Tuple
from e2b_code_interpreter import Sandbox, Result
from e2b_code_interpreter.models import Logs

from e2b_hello_world.model import MODEL_NAME, SYSTEM_PROMPT, tools
from e2b_hello_world.code_interpreter import code_interpret

# Load the .env file
load_dotenv()

client = Anthropic()

def chat(code_interpreter: Sandbox, user_message: str) -> Tuple[List[Result], Logs]:
    print(f"\n{'='*50}\nUser Message: {user_message}\n{'='*50}")

    message = client.messages.create(
        model=MODEL_NAME,
        system=SYSTEM_PROMPT,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_message}],
        tools=tools,
    )

    print(f"\n{'='*50}\nModel response: {message.content}\n{'='*50}")

    if message.stop_reason == "tool_use":
        tool_use = next(block for block in message.content if block.type == "tool_use")
        tool_name = tool_use.name
        tool_input = tool_use.input

        print(f"\n{'='*50}\nUsing tool: {tool_name}\n{'='*50}")

        if tool_name == "execute_python":
            return code_interpret(code_interpreter, tool_input["code"])

    return None, None

def main():
  user_message = "Estimate a distribution of height of men without using external data sources. Also print the median value."

  # Create the CodeInterpreter object and save it as code_interpreter
  with Sandbox() as code_interpreter:
    retries = 3
    attempt = 0

    while attempt < retries:
        code_interpreter_results, code_interpreter_logs = chat(code_interpreter, user_message)
        if code_interpreter_results is not None:
            break
        attempt += 1
        print(f"Result is None. Retrying... attempt {attempt}/{retries}")

    if code_interpreter_results is None:
        raise ValueError("Max retries reached and result is still None.")

    print(code_interpreter_logs)

    first_result= code_interpreter_results[0]
    print(first_result)

    # If we received a chart in PNG form, we can visualize it
    if first_result.png:
      # Decode the base64 encoded PNG data
      png_data = base64.b64decode(first_result.png)

      # Generate a unique filename for the PNG
      filename = f"chart.png"

      # Save the decoded PNG data to a file
      with open(filename, "wb") as f:
          f.write(png_data)

      print(f"Saved chart to {filename}")
