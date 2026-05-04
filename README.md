# Energy Narrative Engine

An LLM-powered service that generates personalized energy email content for utility customers. It pulls billing, appliance itemization, weather, and time-of-use behavior data from the Bidgely APIs, derives insights, and uses an LLM to produce a structured email payload (subject line, greeting, energy story, breakdown, behavior summary, and tiered recommendation tips). The service runs as an SQS worker and publishes the generated payload as XML to a downstream queue.

## Features

- **SQS-driven worker** — polls an input queue for jobs, processes each one, and publishes results to an output queue.
- **Three notification types** — `regular`, `monthly_summary`, and `bill_projection`.
- **Parallel data fetching** — user, endpoints, and billing data are fetched concurrently with a `ThreadPoolExecutor`.
- **Tone selection** — the LLM picks between `casual`, `excited`, `formal`, and `attention` tones based on bill change, dominant appliance share, and appliance count.
- **Bill projection tool** — exposes a LangChain `@tool` so the LLM can fetch in-cycle projected bills on demand.
- **Behavior insights** — derives peak-hour, night-hour, and per-appliance usage patterns from `tbappdata` time-bucket data.
- **Weather context** — summarizes hot days, cold days, and average temperatures over the billing cycle.
- **Structured XML output** — the final payload is serialized to XML for the downstream notification system.

## Architecture

```
SQS input queue
      │
      ▼
┌──────────────────────────────────────────────────────┐
│  poll_forever → process_message → EnergyEmailAgent   │
│                                                      │
│   1. load_base_data        (user, endpoints, billing)│
│   2. load_dependent_data   (itemization, weather)    │
│   3. derive                (insights + behavior)     │
│   4. generate_sections     (LLM email content)       │
│                                                      │
│            │                                         │
│            ▼                                         │
│   build_email_payload → build_output_queue_message   │
└──────────────────────────────────────────────────────┘
      │
      ▼
SQS output queue (XML)
```

## Project Layout

This repo contains a single module:

- `energy_narrative_engine.py` — all logic: data models, Bidgely HTTP client, parsers, insight derivation, LLM agent, SQS worker.

Key building blocks inside the file:

- `BidgelyClient` — thin HTTP client for the Bidgely API (user, endpoints, billing, itemization, weather, bill projection).
- `AgentState` — dataclass that carries every intermediate artifact through the pipeline.
- `EnergyEmailAgent` — orchestrates data loading, insight derivation, and LLM section generation.
- `build_email_payload` / `build_output_queue_message` — assemble the final JSON payload and the XML message published to SQS.
- `poll_forever` — the SQS worker entry point.

## Requirements

- Python 3.9+
- AWS credentials with permission to read from / write to the configured SQS queues
- An OpenAI API key (or an Ollama instance if you switch `build_llm()` to the commented-out branch)
- Network access to the Bidgely API base URL

### Python dependencies

```
requests
boto3
langchain-core
langchain-openai
langchain-ollama
```

Install:

```bash
pip install requests boto3 langchain-core langchain-openai langchain-ollama
```

## Configuration

All configuration is via environment variables.

| Variable | Required | Description |
|---|---|---|
| `BASE_URL` | yes | Base URL of the Bidgely API (e.g. `https://productqaapi-external.bidgely.com`). |
| `BIDGELY_BEARER_TOKEN` | yes | Bearer token for the Bidgely API. |
| `SQS_QUEUE_URL` | yes | Input SQS queue URL the worker polls. |
| `OUTPUT_SQS_QUEUE_URL` | no | Downstream SQS queue URL. If unset, the worker logs the payload but does not publish. |
| `AWS_REGION` | no | AWS region for SQS clients. Default: `ap-south-1`. |
| `MODEL_NAME` | no | OpenAI model name. Default: `gpt-5.4-mini`. |
| `OLLAMA_BASE_URL` | no | Ollama URL if you switch to the local LLM branch. Default: `http://localhost:11434`. |
| `OLLAMA_MODEL` | no | Ollama model. Default: `llama3`. |
| `POLL_WAIT_SECONDS` | no | SQS long-poll seconds. Default: `20`. |
| `VISIBILITY_TIMEOUT` | no | SQS message visibility timeout. Default: `180`. |
| `MAX_MESSAGES` | no | Max messages per receive. Default: `1`. |
| `TBAPPDATA_FILE_PATH` | no | Path to the `tbappdata` time-bucket JSON file used for behavior analysis. Default: `/mnt/data/tbappdata`. Behavior insights are skipped if the file is missing. |
| `OPENAI_API_KEY` | yes (when using OpenAI) | Standard OpenAI auth env var consumed by `langchain-openai`. |

## Running

Set the required environment variables and start the worker:

```bash
export BASE_URL="https://productqaapi-external.bidgely.com"
export BIDGELY_BEARER_TOKEN="..."
export SQS_QUEUE_URL="https://sqs.ap-south-1.amazonaws.com/.../input-queue"
export OUTPUT_SQS_QUEUE_URL="https://sqs.ap-south-1.amazonaws.com/.../output-queue"
export OPENAI_API_KEY="..."

python energy_narrative_engine.py
```

The worker will long-poll the input queue, process each message, publish the XML result to the output queue, and delete the input message on success. Failed messages are left in the queue and retried after the visibility timeout.

## Input Message Format

Each SQS message body is a JSON object:

```json
{
  "user_id": "d2f6d611-4385-4536-82cd-26c19899ee86",
  "home_id": "1",
  "measurement_type": "ELECTRIC",
  "notification_type": "monthly_summary"
}
```

| Field | Required | Notes |
|---|---|---|
| `user_id` | yes | Bidgely user UUID. |
| `home_id` | yes | Home ID (string). |
| `measurement_type` | no | Default `ELECTRIC`. |
| `notification_type` | no | One of `regular`, `monthly_summary` (default), `bill_projection`. |

## Output Message Format

The worker publishes XML to `OUTPUT_SQS_QUEUE_URL` with the following top-level structure:

```xml
<genericEventData>
  <uuid>...</uuid>
  <hid>1</hid>
  <eventType>MonthlySummary | Regular | BillProjection</eventType>
  <userDeliveryModes>Email</userDeliveryModes>
  <billProjection>...</billProjection>           <!-- only for BillProjection -->
  <llmEmailContent>
    <subjectLine>...</subjectLine>
    <greetingText>...</greetingText>
    <tone>casual | excited | formal | attention</tone>
    <billCycleDetails>...</billCycleDetails>
    <itemizationDetails>
      <appliance>...</appliance>
      ...
    </itemizationDetails>
    <energyStory>...</energyStory>
    <energyBreakdown>...</energyBreakdown>
    <behaviorSummary>...</behaviorSummary>
    <recommendationTips>
      <thisWeek>...</thisWeek>
      <thisMonth>...</thisMonth>
      <thisYear>...</thisYear>
    </recommendationTips>
    <tokenUsage>...</tokenUsage>
  </llmEmailContent>
</genericEventData>
```

## Notification Types

- **`regular`** — standard email with energy story, breakdown, behavior summary, and recommendation tips.
- **`monthly_summary`** — same structure, framed as a monthly recap (default).
- **`bill_projection`** — additionally fetches an in-cycle bill projection via the `get_bill_projection` LangChain tool, computes projected vs. expected and projected vs. last-bill deltas, and embeds the result in the output XML.

## Behavior Insights

If a `tbappdata` JSON file is present at `TBAPPDATA_FILE_PATH`, the engine derives:

- Per-appliance hourly usage, peak-hour share (6–9 PM), night-hour share (12–5 AM), and dominant hour.
- Pattern flags such as *evening AC user*, *high always-on baseline*, *night owl usage*, and *evening-heavy appliance usage*.
- A natural-language peak-cost-awareness sentence.

These are passed to the LLM as `behavior_facts` so the generated email can reference real usage patterns.

## Switching to a Local LLM

`build_llm()` defaults to `ChatOpenAI`. To use a local Ollama model instead, swap the function body to the commented-out `ChatOllama(...)` block at the top of the function and set `OLLAMA_BASE_URL` / `OLLAMA_MODEL`.

## Notes & Caveats

- `BIDGELY_BEARER_TOKEN` is read from the environment; do not commit tokens to source control.
- The worker selects the latest billing cycle where `bidgelyGeneratedInvoice == false`. If no such cycle exists for a user, the run raises and the message is retried.
- `derive_bill_projection_facts` contains a placeholder estimation (first-15-days extrapolation) that is overridden when the live bill-projection API tool returns data; replace with real mid-cycle data when available.
- The hard-coded `BILLING_T0` / `BILLING_T1` constants define the billing window queried from Bidgely. Adjust if you need a different range.

## License
