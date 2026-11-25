import json
import os
import sys
import time
import urllib.request

# 1. Environment configuration for Phoenix before import
os.environ["PHOENIX_WORKING_DIR"] = os.path.join(os.getcwd(), ".phoenix")
os.environ["PHOENIX_PORT"] = "6007"

import pandas as pd
import phoenix as px
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from phoenix.client import Client as phoenixClient
from phoenix.evals import LLM, ClassificationEvaluator, evaluate_dataframe
from phoenix.otel import register
from pydantic import BaseModel, Field

# 2. Launch Phoenix and instrument tracing
session = px.launch_app(use_temp_dir=False)
tracer_provider = register(
    project_name="live-api-multi-tool-agent",
)
GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# 3. Setup Gemini client
load_dotenv()
api_key = os.getenv("Gemini_Api_Key")
client = genai.Client(api_key=api_key)

px_client = phoenixClient()


# 4. Live API tools
def get_live_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Fetch real time currency exchange rate from a live API.

    Args:
     from_currency: 3 letter source currency code (e.g., 'USD' , 'EUR').
     to_currency : 3 letter target currency code (e.g., 'NPR' , 'GBP' , 'EUR')
    """
    base_curr = from_currency.upper()
    target_cur = to_currency.upper()
    url = f"https://open.er-api.com/v6/latest/{base_curr}"

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Python/3.13"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())

        if data.get("result") == "success":
            rate = data["rates"].get(target_cur)
            if rate:
                return json.dumps(
                    {
                        "base": base_curr,
                        "target": target_cur,
                        "rate": rate,
                        "latest_updated": data.get("time_last_update_utc"),
                    }
                )
            return json.dumps(
                {"error": f"Currency code '{target_cur}' not found."}
            )
        return json.dumps({"error": "Failed to retrieve rate data."})
    except Exception as e:
        return json.dumps({"error": f"Api request failed: {str(e)}"})


def get_current_weather(latitude: float, longitude: float) -> str:
    """Fetch real time current-weather metrics (temperature in Celsius and windspeed)
    from Open-Meteo API using geographical coordinates.

    Args:
        latitude: Latitude of the location (e.g., 27.7172 for kathmandu).
        longitude: Longitude of the location (e.g., 85.3240 for kathmandu).
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}&current_weather=true"
    )

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Python/3.13"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())

        if "current_weather" in data:
            weather = data["current_weather"]
            return json.dumps(
                {
                    "temperature_celsius": weather.get("temperature"),
                    "windspeed_kmh": weather.get("windspeed"),
                    "weather_code": weather.get("weather_code"),
                }
            )
        return json.dumps({"error": "Weather data unavailable."})
    except Exception as e:
        return json.dumps({"error": f"Weather API error: {str(e)} "})


def calculate_converted_amount(amount: float, rate: float) -> float:
    """Calculate converted money value based on amount and the exchange rate

    Args:
     amount: Base monetary amount
     rate : Exchange rate to multiply by
    """
    return amount * rate


def run_live_agent(prompt: str) -> str:
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            tools=[
                calculate_converted_amount,
                get_current_weather,
                get_live_exchange_rate,
            ],
            temperature=0.0,
        ),
    )
    response = chat.send_message(prompt)
    return response.text


class AgentFinalResponse(BaseModel):
    exchange_rate: float = Field(
        description="The live USD to NPR exchange rate."
    )
    converted_amount_npr: float = Field(
        description="The calculated NPR value for $250 USD"
    )
    kathmandu_temperature_c: float = Field(
        description="Current temperature in Kathmandu in Celsius"
    )
    Summary_notes: str = Field(
        description="A concise narrative summary of retrieved live data"
    )


TOOL_SELECTION_TEMPLATE = """
You are evaluating whether an AI agent selected the correct tool and passed valid arguments based on the user request.

[User Request and context]
{input}

[Agent response / Execution]
{output}

Evaluate the tool selection:
- Label "correct" if the tool name and arguments accurately fulfill the user's intent
- Label "incorrect" if the tool is wrong, unnecessary, or passed invalid/missing parameters

Respond ONLY in JSON format with two fields:
"label": "correct" or "incorrect"
"explanation": A single sentence justification
"""


def main():
    prompt = (
        "Check the live USD exchange rate. "
        "If I convert $250 USD to NPR today, how much NPR will I receive? "
        "Also, get the current temperature in Kathmandu (Latitude: 27.7172, Longitude: 85.3240)"
    )

    print("--- Running Live API multi-tool agent ---")
    print(f"Prompt: {prompt}\n")

    # 1. Execute agent
    result = run_live_agent(prompt)
    print("--- Agent Response ---")
    print(result)

    # Allow async OpenTelemetry exporters to flush spans
    time.sleep(2)

    # 2. Extract Spans
    spans_df = px_client.spans.get_spans_dataframe(
        project_name="live-api-multi-tool-agent"
    )

    if spans_df.empty:
        print("No trace spans found.")
        return

    print("Detected spans kind:", list(set(spans_df["span_kind"])))

    # Filter for tool spans and exclude prior evaluation spans
    tool_spans = spans_df.query("span_kind in ['TOOL', 'LLM']").copy()
    eval_spans = tool_spans[tool_spans["span_kind"] != "EVALUATOR"].copy()

    if eval_spans.empty:
        print("No TOOL/LLM spans found to evaluate.")
        return

    # Normalize input and output columns
    if "attributes.input.value" in eval_spans.columns:
        eval_spans["input"] = eval_spans["attributes.input.value"]

    if "attributes.output.value" in eval_spans.columns:
        eval_spans["output"] = eval_spans["attributes.output.value"]

    # Fill missing values to prevent template formatting key errors
    eval_spans["input"] = eval_spans["input"].fillna("N/A")
    eval_spans["output"] = eval_spans["output"].fillna("N/A")

    # Slice top 2 rows to respect Gemini free-tier rate limits
    sample_spans = eval_spans.head(2)

    # 3. Instantiate Evaluator & Run Classification
    print(f"\n--- Running Evaluation on {len(sample_spans)} spans ---")
    evaluator = ClassificationEvaluator(
        name="Tool Call Accuracy",
        prompt_template=TOOL_SELECTION_TEMPLATE,
        choices={"correct": 1.0, "incorrect": 0.0},
        llm=LLM(provider="google", model="gemini-2.5-flash"),
    )

    eval_results_df = evaluate_dataframe(
        dataframe=sample_spans,
        evaluators=[evaluator],
    )

    print("\n--- Evaluation Results ---")
    print(eval_results_df)

    # 4. Log Evaluation result back to Phoenix UI cleanly
    try:
        from phoenix.trace import log_evaluations

        log_evaluations(
            eval_name="Tool Call Accuracy", dataframe=eval_results_df
        )
        print(
            "\nSuccessfully logged evaluations to Phoenix UI (http://localhost:6007)!"
        )
    except Exception as e1:
        try:
            px_client.spans.log_span_annotations(eval_results_df)
            print("\nSuccessfully logged evaluations via span annotations!")
        except Exception as e2:
            print(f"\nLogging exception: {e2}")

    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
    input("\n press enter to exit")