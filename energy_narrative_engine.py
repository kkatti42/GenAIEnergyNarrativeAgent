import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


import requests
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import boto3
from botocore.exceptions import ClientError


# =============================================================================
# CONFIG
# =============================================================================

# from hackathon_with_bill_projection_tool import (
#     BidgelyClient,
#     EnergyEmailAgent,
#     build_email_payload,
#     NOTIFICATION_TYPE_REGULAR,
# )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4-mini")

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
OUTPUT_SQS_QUEUE_URL = os.getenv("OUTPUT_SQS_QUEUE_URL")
BASE_URL = os.getenv("BASE_URL")
BIDGELY_BEARER_TOKEN = os.getenv("BIDGELY_BEARER_TOKEN")

POLL_WAIT_SECONDS = int(os.getenv("POLL_WAIT_SECONDS", "20"))
VISIBILITY_TIMEOUT = int(os.getenv("VISIBILITY_TIMEOUT", "180"))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "1"))

#BASE_URL = "https://productqaapi-external.bidgely.com"
USER_ID = "d2f6d611-4385-4536-82cd-26c19899ee86"
HOME_ID = "1"
MEASUREMENT_TYPE = "ELECTRIC"

NOTIFICATION_TYPE_REGULAR = "regular"
NOTIFICATION_TYPE_MONTHLY_SUMMARY = "monthly_summary"
NOTIFICATION_TYPE_BILL_PROJECTION = "bill_projection"

BILLING_T0 = 1422776400
BILLING_T1 = 1844044198
ITEMIZATION_FROM_DATE = "2017-01-01"
ITEMIZATION_TO_DATE = "2026-12-01"
TBAPPDATA_FILE_PATH = os.getenv("TBAPPDATA_FILE_PATH", "/mnt/data/tbappdata")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
#BIDGELY_BEARER_TOKEN = os.getenv("BIDGELY_BEARER_TOKEN", "9840074e-fee0-49be-b3be-7510fb80aa0f")
BILL_PROJECTION_API_PATH_TEMPLATE = os.getenv(
    "BILL_PROJECTION_API_PATH_TEMPLATE",
    "/2.1/users/{user_id}/homes/{home_id}/billprojections",
)

PEAK_HOURS = {18, 19, 20}  # 6 PM to 9 PM based on hour buckets
NIGHT_HOURS = {0, 1, 2, 3, 4, 5}
TB_APP_ID_TO_CATEGORY = {
    3: "laundry",
    4: "cooking",
    7: "entertainment",
    8: "airConditioning",
    9: "spaceHeating",
    16: "refrigeration",
    18: "lighting",
    71: "alwaysOn",
    99: "other",
    1001: "other",
    1002: "other",
}

logging.basicConfig(level=logging.INFO)
logging.getLogger("openai").setLevel(logging.DEBUG)
logger = logging.getLogger("energy_email_agent")
# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class HeaderContent:
    subject_line: str
    greeting_text: str
    
@dataclass
class UserProfile:
    user_id: str
    full_name: Optional[str]
    zipcode: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    timezone: Optional[str]
    email: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EndpointInfo:
    endpoint_id: str
    measurement_type: Optional[str] = None
    profile: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BillingCycle:
    billing_start_ts: int
    billing_end_ts: int
    start_date: str
    end_date: str
    cost: float
    consumption_kwh: float
    bidgely_generated_invoice: bool
    estimation_type: Optional[str] = None
    user_type: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApplianceBreakdown:
    category: str
    label: str
    usage_kwh: float
    cost: float
    percentage: float
    cost_percentage: Optional[float] = None
    is_estimated: Optional[bool] = None


@dataclass
class ItemizationCycle:
    start_ts: int
    end_ts: int
    start_date: str
    end_date: str
    appliances: List[ApplianceBreakdown]
    total_usage_kwh: float
    total_cost: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeatherDay:
    ts: int
    min_temp: Optional[float]
    max_temp: Optional[float]
    avg_temp: Optional[float]


@dataclass
class WeatherSummary:
    start_ts: int
    end_ts: int
    total_days: int
    avg_temp_f: Optional[float]
    max_temp_f: Optional[float]
    min_temp_f: Optional[float]
    days_above_95f: int
    days_above_90f: int
    days_below_40f: int
    warm_days_80_plus: int
    raw_days: List[WeatherDay] = field(default_factory=list)


@dataclass
class DerivedInsights:
    bill_change_amount: Optional[float]
    bill_change_percent: Optional[float]
    usage_change_kwh: Optional[float]
    usage_change_percent: Optional[float]
    dominant_appliance: Optional[str]
    dominant_appliance_cost: Optional[float]
    dominant_appliance_share_percent: Optional[float]
    remaining_appliance_cost: Optional[float]
    remaining_appliance_share_percent: Optional[float]
    weather_signal: Optional[str]
    tone: str

@dataclass
class BillProjectionFacts:
    cycle_start_date: Optional[str]
    cycle_end_date: Optional[str]
    days_observed: Optional[int]
    days_remaining: Optional[int]
    observed_usage_kwh: Optional[float]
    observed_cost: Optional[float]
    avg_hourly_consumption_kwh: Optional[float]
    projected_remaining_usage_kwh: Optional[float]
    projected_total_usage_kwh: Optional[float]
    projected_total_cost: Optional[float]
    projected_vs_expected_amount: Optional[float]
    projected_vs_expected_percent: Optional[float]
    projected_vs_last_bill_amount: Optional[float]
    projected_vs_last_bill_percent: Optional[float]
    projected_dominant_appliance: Optional[str]
    projected_dominant_appliance_share_percent: Optional[float]
    weather_signal: Optional[str]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationContent:
    intrigue_html: str
    action_html: str
    insight_html: str


@dataclass
class RecommendationTips:
    this_week: RecommendationContent
    this_month: RecommendationContent
    this_year: RecommendationContent


@dataclass
class ApplianceBehaviorSummary:
    app_id: int
    label: str
    total_usage: float
    hourly_usage: Dict[int, float] = field(default_factory=dict)
    peak_usage: float = 0.0
    peak_share_percent: float = 0.0
    night_usage: float = 0.0
    night_share_percent: float = 0.0
    dominant_hour: Optional[int] = None


@dataclass
class BehaviorInsights:
    behavior_patterns: List[str] = field(default_factory=list)
    peak_cost_awareness_text: Optional[str] = None
    peak_window: str = "6 PM to 9 PM"
    total_usage: float = 0.0
    peak_usage: float = 0.0
    peak_share_percent: float = 0.0
    night_usage: float = 0.0
    night_share_percent: float = 0.0
    appliance_summaries: List[ApplianceBehaviorSummary] = field(default_factory=list)


@dataclass
class BehaviorPatternContent:
    title: str
    description: str


@dataclass
class BehaviorSummaryContent:
    patterns: List[BehaviorPatternContent] = field(default_factory=list)
    peak_cost_awareness: str = ""


@dataclass
class EmailSections:
    header: HeaderContent
    tone: str
    energy_story: str
    energy_breakdown: str
    recommendation_tips: RecommendationTips
    behavior_summary: BehaviorSummaryContent


@dataclass
class AgentState:
    user_profile: Optional[UserProfile] = None
    home_id: Optional[str] = None
    measurement_type: Optional[str] = None
    endpoint: Optional[EndpointInfo] = None
    selected_bill_cycle: Optional[BillingCycle] = None
    previous_bill_cycle: Optional[BillingCycle] = None
    itemization_cycle: Optional[ItemizationCycle] = None
    weather_summary: Optional[WeatherSummary] = None
    insights: Optional[DerivedInsights] = None
    behavior_insights: Optional[BehaviorInsights] = None
    bill_projection_facts_obj: Optional[BillProjectionFacts] = None
    sections: Optional[EmailSections] = None
    style_config: Optional[Dict[str, str]] = None
    notification_type: str = NOTIFICATION_TYPE_REGULAR

    billing_raw: Optional[Dict[str, Any]] = None
    user_raw: Optional[Dict[str, Any]] = None
    endpoints_raw: Optional[Dict[str, Any]] = None
    itemization_raw: Optional[Dict[str, Any]] = None
    weather_raw: Optional[Dict[str, Any]] = None
    llm_usage: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# HTTP CLIENT
# =============================================================================

class BidgelyAPIError(Exception):
    pass


class BidgelyClient:
    def __init__(self, base_url: str, bearer_token: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"bearer {bearer_token}",
            "Content-Type": "application/json",
        })

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        logger.info("GET %s params=%s", url, params)
        response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code != 200:
            raise BidgelyAPIError(
                f"GET {url} failed with status={response.status_code} body={response.text[:500]}"
            )
        return response.json()

    def fetch_user_details(self, user_id: str) -> Dict[str, Any]:
        return self.get(f"/v2.0/users/{user_id}")

    def fetch_user_endpoints(self, user_id: str) -> Dict[str, Any]:
        return self.get(f"/v2.0/users/{user_id}/endpoints")

    def fetch_billing_details(
        self,
        user_id: str,
        home_id: str,
        t0: int,
        t1: int,
        measurement_type: str,
    ) -> Dict[str, Any]:
        return self.get(
            f"/billingdata/users/{user_id}/homes/{home_id}/utilitydata",
            params={
                "t0": t0,
                "t1": t1,
                "measurementType": measurement_type,
            },
        )

    def fetch_itemization(
        self,
        user_id: str,
        endpoint_id: str,
        from_date: str,
        to_date: str,
        measurement_type: str,
    ) -> Dict[str, Any]:
        return self.get(
            f"/v2.0/users/{user_id}/endpoints/{endpoint_id}/itemizationDetails",
            params={
                "fromDate": from_date,
                "toDate": to_date,
                "extended": "true",
                "hybrid": "true",
                "mode": "month",
                "showFailedBillingCycles": "true",
                "showPercentage": "true",
                "round": "true",
                "measurementType": measurement_type,
            },
        )

    def fetch_weather(
        self,
        country_code: str,
        zipcode: str,
        t0: int,
        t1: int,
    ) -> Dict[str, Any]:
        return self.get(
            f"/weather/{country_code}/{zipcode}/data",
            params={
                "mode": "day",
                "t0": t0,
                "t1": t1,
            },
        )

    def fetch_bill_projection(
        self,
        user_id: str,
        home_id: str,
        measurement_type: str,
        billing_start_ts: int,
        billing_end_ts: int,
    ) -> Dict[str, Any]:
        path = BILL_PROJECTION_API_PATH_TEMPLATE.format(user_id=user_id, home_id=home_id)
        return self.get(
            path,
            params={
                "measurementType": measurement_type,
                "billingStartTs": billing_start_ts,
                "billingEndTs": billing_end_ts,
            },
        )
# =============================================================================
# HELPERS
# =============================================================================

def epoch_to_date_str(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None or previous == 0:
        return None
    return round(((current - previous) / previous) * 100, 1)


def amount_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None:
        return None
    return round(current - previous, 2)


def safe_round(value: Optional[float], ndigits: int = 1) -> Optional[float]:
    if value is None:
        return None
    return round(value, ndigits)


def category_to_label(category: str) -> str:
    mapping = {
        "alwaysOn": "always-on devices",
        "spaceHeating": "heating",
        "airConditioning": "AC",
        "entertainment": "entertainment",
        "refrigeration": "refrigerator",
        "cooking": "cooking",
        "lighting": "lighting",
        "laundry": "laundry",
        "other": "other appliances",
        "total": "total",
    }
    return mapping.get(category, category)


def ts_to_hour_only(ts: int) -> int:
    return datetime.fromtimestamp(ts, tz=timezone.utc).hour


def load_tbappdata_file(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def get_itemized_categories(itemization: "ItemizationCycle") -> set:
    return {
        appliance.category
        for appliance in itemization.appliances
        if appliance.category and appliance.category != "total"
    }


def choose_tone(
    bill_change_percent: Optional[float],
    dominant_appliance_share_percent: Optional[float],
    appliance_count: int,
) -> str:
    if bill_change_percent is not None and bill_change_percent >= 20:
        return "attention"
    if dominant_appliance_share_percent is not None and dominant_appliance_share_percent >= 40:
        return "attention"
    if bill_change_percent is not None and bill_change_percent <= -10:
        return "excited"
    if appliance_count <= 3:
        return "formal"
    return "casual"


def build_style_config(tone: str) -> Dict[str, str]:
    style_map = {
        "casual": {
            "voice": "friendly, light, conversational",
            "energy_story_length": "45-70 words",
            "breakdown_length": "18-35 words",
        },
        "excited": {
            "voice": "upbeat, positive, energetic but still clear",
            "energy_story_length": "45-70 words",
            "breakdown_length": "18-35 words",
        },
        "formal": {
            "voice": "clear, polished, neutral, concise",
            "energy_story_length": "40-65 words",
            "breakdown_length": "18-30 words",
        },
        "attention": {
            "voice": "clear, direct, noticeable, but not alarming",
            "energy_story_length": "45-75 words",
            "breakdown_length": "18-35 words",
        },
    }
    return style_map[tone]

# =============================================================================
# INSIGHT DERIVATION
# =============================================================================

def derive_weather_signal(weather: WeatherSummary) -> Optional[str]:
    if weather.days_above_90f > 0:
        return f"There were {weather.days_above_90f} hot days above 90°F."
    if weather.days_below_40f > 0:
        return f"There were {weather.days_below_40f} cold days below 40°F."
    if weather.avg_temp_f is not None:
        return f"Average temperatures were around {weather.avg_temp_f}°F."
    return None


def derive_insights(
    selected_bill: BillingCycle,
    previous_bill: Optional[BillingCycle],
    itemization: ItemizationCycle,
    weather: WeatherSummary,
) -> DerivedInsights:
    comparable_appliances = [a for a in itemization.appliances if a.category != "total"]
    comparable_appliances.sort(key=lambda a: a.cost, reverse=True)

    dominant = comparable_appliances[0] if comparable_appliances else None
    total_cost = itemization.total_cost or selected_bill.cost

    dominant_share = None
    remaining_cost = None
    remaining_share = None

    if dominant and total_cost:
        dominant_share = round((dominant.cost / total_cost) * 100, 1)
        remaining_cost = round(total_cost - dominant.cost, 2)
        remaining_share = round((remaining_cost / total_cost) * 100, 1)

    bill_change_percent = pct_change(
        selected_bill.cost,
        previous_bill.cost if previous_bill else None,
    )

    tone = "llm_defined"

    return DerivedInsights(
        bill_change_amount=amount_change(
            selected_bill.cost,
            previous_bill.cost if previous_bill else None,
        ),
        bill_change_percent=bill_change_percent,
        usage_change_kwh=amount_change(
            selected_bill.consumption_kwh,
            previous_bill.consumption_kwh if previous_bill else None,
        ),
        usage_change_percent=pct_change(
            selected_bill.consumption_kwh,
            previous_bill.consumption_kwh if previous_bill else None,
        ),
        dominant_appliance=dominant.label if dominant else None,
        dominant_appliance_cost=round(dominant.cost, 2) if dominant else None,
        dominant_appliance_share_percent=dominant_share,
        remaining_appliance_cost=remaining_cost,
        remaining_appliance_share_percent=remaining_share,
        weather_signal=derive_weather_signal(weather),
        tone=tone,
    )


# =============================================================================
# FACT BUILDERS
# =============================================================================

def build_user_facts(profile: UserProfile) -> Dict[str, Any]:
    return {
        "user_id": profile.user_id,
        "full_name": profile.full_name,
        "zipcode": profile.zipcode,
        "timezone": profile.timezone,
    }


def build_billing_facts(
    selected: BillingCycle,
    previous: Optional[BillingCycle],
    insights: DerivedInsights,
) -> Dict[str, Any]:
    return {
        "billing_start_date": selected.start_date,
        "billing_end_date": selected.end_date,
        "current_bill_amount": round(selected.cost, 2),
        "current_usage_kwh": round(selected.consumption_kwh, 2),
        "previous_bill_amount": round(previous.cost, 2) if previous else None,
        "previous_usage_kwh": round(previous.consumption_kwh, 2) if previous else None,
        "bill_change_amount": insights.bill_change_amount,
        "bill_change_percent": insights.bill_change_percent,
        "usage_change_kwh": insights.usage_change_kwh,
        "usage_change_percent": insights.usage_change_percent,
    }


def build_itemization_facts(itemization: ItemizationCycle) -> Dict[str, Any]:
    appliances = [a for a in itemization.appliances if a.category != "total"]
    appliances.sort(key=lambda a: a.cost, reverse=True)
    top_appliances = appliances[:3]

    return {
        "start_date": itemization.start_date,
        "end_date": itemization.end_date,
        "total_cost": round(itemization.total_cost, 2),
        "total_usage_kwh": round(itemization.total_usage_kwh, 2),
        "top_appliances": [
            {
                "label": a.label,
                "cost": round(a.cost, 2),
                "usage_kwh": round(a.usage_kwh, 2),
                "share_percent": round(a.cost_percentage, 1) if a.cost_percentage is not None else round(a.percentage, 1),
            }
            for a in top_appliances
        ],
        "top_appliance_count": len(top_appliances),
        "remaining_appliance_count": max(len(appliances) - len(top_appliances), 0),
    }


def build_itemization_details(itemization: ItemizationCycle) -> Dict[str, Any]:
    return {
        "start_ts": itemization.start_ts,
        "end_ts": itemization.end_ts,
        "start_date": itemization.start_date,
        "end_date": itemization.end_date,
        "total_cost": round(itemization.total_cost, 2),
        "total_usage_kwh": round(itemization.total_usage_kwh, 2),
        "appliances": [
            {
                "category": a.category,
                "label": a.label,
                "usage_kwh": round(a.usage_kwh, 2),
                "cost": round(a.cost, 2),
                "percentage": round(a.percentage, 1),
                "cost_percentage": round(a.cost_percentage, 1) if a.cost_percentage is not None else None,
                "is_estimated": a.is_estimated,
            }
            for a in itemization.appliances
        ],
    }


def build_weather_facts(weather: WeatherSummary) -> Dict[str, Any]:
    return {
        "billing_cycle_start_ts": weather.start_ts,
        "billing_cycle_end_ts": weather.end_ts,
        "days_in_cycle": weather.total_days,
        "avg_temp_f": weather.avg_temp_f,
        "max_temp_f": weather.max_temp_f,
        "min_temp_f": weather.min_temp_f,
        "days_above_95f": weather.days_above_95f,
        "days_above_90f": weather.days_above_90f,
        "days_below_40f": weather.days_below_40f,
    }


def build_insight_facts(insights: DerivedInsights) -> Dict[str, Any]:
    return asdict(insights)

def build_behavior_facts(behavior: Optional[BehaviorInsights]) -> Dict[str, Any]:
    if not behavior:
        return {
            "is_available": False,
            "message": "Behavior insights are not available."
        }

    return {
        "is_available": True,
        "peak_window": behavior.peak_window,
        "behavior_patterns": behavior.behavior_patterns,
        "peak_cost_awareness": behavior.peak_cost_awareness_text,
        "total_usage": behavior.total_usage,
        "peak_usage": behavior.peak_usage,
        "peak_share_percent": behavior.peak_share_percent,
        "night_usage": behavior.night_usage,
        "night_share_percent": behavior.night_share_percent,
        "appliance_summaries": [
            {
                "label": summary.label,
                "total_usage": summary.total_usage,
                "peak_share_percent": summary.peak_share_percent,
                "night_share_percent": summary.night_share_percent,
                "dominant_hour": summary.dominant_hour,
            }
            for summary in behavior.appliance_summaries
        ],
    }

def build_recommendation_facts() -> Dict[str, Any]:
    return {
        "this_week": {
            "goal": "quick win",
            "savings_per_month": 30,
            "action": "Simple adjustments during peak hours",
            "extra_context": "Takes 2 minutes, zero investment."
        },
        "this_month": {
            "goal": "small investment",
            "savings_per_month": 15,
            "action": "Affordable appliance upgrades",
            "extra_context": "Total investment: ~$35."
        },
        "this_year": {
            "goal": "long-term upgrade",
            "savings_per_month": 50,
            "action": "Smart appliance upgrades and maintenance",
            "extra_context": "ROI in under a year."
        }
    }


def to_compact_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def safe_pct_change(new_value: Optional[float], old_value: Optional[float]) -> Optional[float]:
    if new_value is None or old_value in (None, 0):
        return None
    return round(((new_value - old_value) / old_value) * 100, 1)


def deep_find_first(obj: Any, candidate_keys: List[str]) -> Any:
    if isinstance(obj, dict):
        lowered = {str(k).lower(): v for k, v in obj.items()}
        for key in candidate_keys:
            if key.lower() in lowered and lowered[key.lower()] is not None:
                return lowered[key.lower()]
        for value in obj.values():
            found = deep_find_first(value, candidate_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = deep_find_first(item, candidate_keys)
            if found is not None:
                return found
    return None


def coerce_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(float(value))
    except (TypeError, ValueError):
        return None


def parse_bill_projection_api_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    projected_total_cost = coerce_float(deep_find_first(raw, [
        "projectedTotalCost", "projectedCost", "projectedBillAmount", "projectedBill", "projectedPrice"
    ]))
    projected_total_usage_kwh = coerce_float(deep_find_first(raw, [
        "projectedTotalUsageKwh", "projectedUsageKwh", "projectedUsage", "projectedConsumptionKwh"
    ]))
    observed_cost = coerce_float(deep_find_first(raw, [
        "observedCost", "currentCost", "costToDate", "currentCostToDate", "cost_so_far"
    ]))
    observed_usage_kwh = coerce_float(deep_find_first(raw, [
        "observedUsageKwh", "currentUsageKwh", "usageToDate", "currentUsageToDateKwh", "usage_so_far_kwh"
    ]))
    days_observed = coerce_int(deep_find_first(raw, [
        "daysObserved", "daysElapsed", "elapsedDays"
    ]))
    days_remaining = coerce_int(deep_find_first(raw, [
        "daysRemaining", "remainingDays"
    ]))
    avg_hourly_consumption_kwh = coerce_float(deep_find_first(raw, [
        "avgHourlyConsumptionKwh", "averageHourlyConsumptionKwh", "avgHourlyUsageKwh"
    ]))
    cycle_start_date = deep_find_first(raw, ["cycleStartDate", "billingStartDate", "startDate"])
    cycle_end_date = deep_find_first(raw, ["cycleEndDate", "billingEndDate", "endDate"])

    return {
        "cycle_start_date": cycle_start_date,
        "cycle_end_date": cycle_end_date,
        "days_observed": days_observed,
        "days_remaining": days_remaining,
        "observed_usage_kwh": observed_usage_kwh,
        "observed_cost": observed_cost,
        "avg_hourly_consumption_kwh": avg_hourly_consumption_kwh,
        "projected_total_usage_kwh": projected_total_usage_kwh,
        "projected_total_cost": projected_total_cost,
        "raw": raw,
    }


def merge_bill_projection_facts(
    existing: Optional[BillProjectionFacts],
    tool_data: Dict[str, Any],
    previous_bill: Optional[BillingCycle],
) -> BillProjectionFacts:
    existing = existing or BillProjectionFacts(
        cycle_start_date=None,
        cycle_end_date=None,
        days_observed=None,
        days_remaining=None,
        observed_usage_kwh=None,
        observed_cost=None,
        avg_hourly_consumption_kwh=None,
        projected_remaining_usage_kwh=None,
        projected_total_usage_kwh=None,
        projected_total_cost=None,
        projected_vs_expected_amount=None,
        projected_vs_expected_percent=None,
        projected_vs_last_bill_amount=None,
        projected_vs_last_bill_percent=None,
        projected_dominant_appliance=None,
        projected_dominant_appliance_share_percent=None,
        weather_signal=None,
        raw={},
    )

    projected_total_cost = tool_data.get("projected_total_cost") if tool_data.get("projected_total_cost") is not None else existing.projected_total_cost
    projected_total_usage_kwh = tool_data.get("projected_total_usage_kwh") if tool_data.get("projected_total_usage_kwh") is not None else existing.projected_total_usage_kwh
    observed_usage_kwh = tool_data.get("observed_usage_kwh") if tool_data.get("observed_usage_kwh") is not None else existing.observed_usage_kwh
    observed_cost = tool_data.get("observed_cost") if tool_data.get("observed_cost") is not None else existing.observed_cost
    days_observed = tool_data.get("days_observed") if tool_data.get("days_observed") is not None else existing.days_observed
    days_remaining = tool_data.get("days_remaining") if tool_data.get("days_remaining") is not None else existing.days_remaining
    avg_hourly = tool_data.get("avg_hourly_consumption_kwh") if tool_data.get("avg_hourly_consumption_kwh") is not None else existing.avg_hourly_consumption_kwh

    projected_remaining_usage_kwh = existing.projected_remaining_usage_kwh
    if projected_total_usage_kwh is not None and observed_usage_kwh is not None:
        projected_remaining_usage_kwh = round(projected_total_usage_kwh - observed_usage_kwh, 2)

    projected_vs_expected_amount = None
    projected_vs_expected_percent = None
    if projected_total_cost is not None and observed_cost is not None:
        projected_vs_expected_amount = round(projected_total_cost - observed_cost, 2)
        projected_vs_expected_percent = safe_pct_change(projected_total_cost, observed_cost)

    projected_vs_last_bill_amount = None
    projected_vs_last_bill_percent = None
    if projected_total_cost is not None and previous_bill and previous_bill.cost is not None:
        projected_vs_last_bill_amount = round(projected_total_cost - previous_bill.cost, 2)
        projected_vs_last_bill_percent = safe_pct_change(projected_total_cost, previous_bill.cost)

    raw = dict(existing.raw or {})
    raw["tool_projection_response"] = tool_data.get("raw", {})

    return BillProjectionFacts(
        cycle_start_date=tool_data.get("cycle_start_date") or existing.cycle_start_date,
        cycle_end_date=tool_data.get("cycle_end_date") or existing.cycle_end_date,
        days_observed=days_observed,
        days_remaining=days_remaining,
        observed_usage_kwh=observed_usage_kwh,
        observed_cost=observed_cost,
        avg_hourly_consumption_kwh=avg_hourly,
        projected_remaining_usage_kwh=projected_remaining_usage_kwh,
        projected_total_usage_kwh=projected_total_usage_kwh,
        projected_total_cost=projected_total_cost,
        projected_vs_expected_amount=projected_vs_expected_amount if projected_vs_expected_amount is not None else existing.projected_vs_expected_amount,
        projected_vs_expected_percent=projected_vs_expected_percent if projected_vs_expected_percent is not None else existing.projected_vs_expected_percent,
        projected_vs_last_bill_amount=projected_vs_last_bill_amount if projected_vs_last_bill_amount is not None else existing.projected_vs_last_bill_amount,
        projected_vs_last_bill_percent=projected_vs_last_bill_percent if projected_vs_last_bill_percent is not None else existing.projected_vs_last_bill_percent,
        projected_dominant_appliance=existing.projected_dominant_appliance,
        projected_dominant_appliance_share_percent=existing.projected_dominant_appliance_share_percent,
        weather_signal=existing.weather_signal,
        raw=raw,
    )


def build_bill_projection_facts(facts: Optional[BillProjectionFacts]) -> Dict[str, Any]:
    if not facts:
        return {
            "is_available": False,
            "message": "Bill projection facts are not available."
        }

    return {
        "is_available": True,
        "cycle_start_date": facts.cycle_start_date,
        "cycle_end_date": facts.cycle_end_date,
        "days_observed": facts.days_observed,
        "days_remaining": facts.days_remaining,
        "observed_usage_kwh": facts.observed_usage_kwh,
        "observed_cost": facts.observed_cost,
        "avg_hourly_consumption_kwh": facts.avg_hourly_consumption_kwh,
        "projected_remaining_usage_kwh": facts.projected_remaining_usage_kwh,
        "projected_total_usage_kwh": facts.projected_total_usage_kwh,
        "projected_total_cost": facts.projected_total_cost,
        "projected_vs_expected_amount": facts.projected_vs_expected_amount,
        "projected_vs_expected_percent": facts.projected_vs_expected_percent,
        "projected_vs_last_bill_amount": facts.projected_vs_last_bill_amount,
        "projected_vs_last_bill_percent": facts.projected_vs_last_bill_percent,
        "projected_dominant_appliance": facts.projected_dominant_appliance,
        "projected_dominant_appliance_share_percent": facts.projected_dominant_appliance_share_percent,
        "weather_signal": facts.weather_signal,
        "projection_method": (
            "Projection uses the first half of the billing cycle to estimate the remaining days. "
            "Average hourly consumption observed so far is extended across the rest of the cycle and billing rates are applied to estimate the end-of-cycle cost."
        ),
    }


def derive_bill_projection_facts(
    selected_bill: BillingCycle,
    previous_bill: Optional[BillingCycle],
    itemization: ItemizationCycle,
    weather: WeatherSummary,
) -> BillProjectionFacts:
    total_days = max(1, ((selected_bill.billing_end_ts - selected_bill.billing_start_ts) // 86400) + 1)
    days_observed = min(15, total_days)
    days_remaining = max(total_days - days_observed, 0)

    observed_usage_kwh = round((selected_bill.consumption_kwh / total_days) * days_observed, 2) if selected_bill.consumption_kwh is not None else None
    observed_cost = round((selected_bill.cost / total_days) * days_observed, 2) if selected_bill.cost is not None else None

    observed_hours = days_observed * 24
    avg_hourly_consumption_kwh = round(observed_usage_kwh / observed_hours, 4) if observed_usage_kwh is not None and observed_hours > 0 else None

    projected_remaining_usage_kwh = round(avg_hourly_consumption_kwh * days_remaining * 24, 2) if avg_hourly_consumption_kwh is not None else None
    projected_total_usage_kwh = round((observed_usage_kwh or 0) + (projected_remaining_usage_kwh or 0), 2) if observed_usage_kwh is not None else None

    effective_rate = None
    if selected_bill.consumption_kwh not in (None, 0):
        effective_rate = selected_bill.cost / selected_bill.consumption_kwh

    projected_total_cost = round(projected_total_usage_kwh * effective_rate, 2) if projected_total_usage_kwh is not None and effective_rate is not None else None

    projected_vs_expected_amount = round(projected_total_cost - selected_bill.cost, 2) if projected_total_cost is not None and selected_bill.cost is not None else None
    projected_vs_expected_percent = safe_pct_change(projected_total_cost, selected_bill.cost)
    projected_vs_last_bill_amount = round(projected_total_cost - previous_bill.cost, 2) if projected_total_cost is not None and previous_bill and previous_bill.cost is not None else None
    projected_vs_last_bill_percent = safe_pct_change(projected_total_cost, previous_bill.cost if previous_bill else None)

    appliances = [a for a in itemization.appliances if a.category != "total"]
    appliances.sort(key=lambda a: a.cost, reverse=True)
    dominant = appliances[0] if appliances else None

    return BillProjectionFacts(
        cycle_start_date=selected_bill.start_date,
        cycle_end_date=selected_bill.end_date,
        days_observed=days_observed,
        days_remaining=days_remaining,
        observed_usage_kwh=observed_usage_kwh,
        observed_cost=observed_cost,
        avg_hourly_consumption_kwh=avg_hourly_consumption_kwh,
        projected_remaining_usage_kwh=projected_remaining_usage_kwh,
        projected_total_usage_kwh=projected_total_usage_kwh,
        projected_total_cost=projected_total_cost,
        projected_vs_expected_amount=projected_vs_expected_amount,
        projected_vs_expected_percent=projected_vs_expected_percent,
        projected_vs_last_bill_amount=projected_vs_last_bill_amount,
        projected_vs_last_bill_percent=projected_vs_last_bill_percent,
        projected_dominant_appliance=dominant.label if dominant else None,
        projected_dominant_appliance_share_percent=dominant.cost_percentage if dominant and dominant.cost_percentage is not None else None,
        weather_signal=derive_weather_signal(weather),
        raw={
            "estimated_from_completed_cycle": True,
            "note": "Replace derive_bill_projection_facts() math with true mid-cycle first-15-days data when your bill projection source is wired in."
        },
    )


def derive_behavior_insights_from_tbappdata(
    tb_raw: Dict[str, Any],
    itemization: ItemizationCycle,
) -> BehaviorInsights:
    itemized_categories = get_itemized_categories(itemization)

    appliance_summaries: List[ApplianceBehaviorSummary] = []
    total_hourly_usage = {hour: 0.0 for hour in range(24)}

    records: List[Dict[str, Any]] = []
    for value in tb_raw.values():
        if isinstance(value, list):
            records.extend(value)

    for record in records:
        app_id = record.get("appId")
        category = TB_APP_ID_TO_CATEGORY.get(app_id)
        if not category or category not in itemized_categories:
            continue

        label = category_to_label(category)
        tb_starts = record.get("tbStartList") or []
        tb_values = record.get("tbValues") or []

        hourly_usage = {hour: 0.0 for hour in range(24)}
        total_usage = 0.0

        for ts, value in zip(tb_starts, tb_values):
            if value is None:
                continue
            usage = float(value)
            hour = ts_to_hour_only(int(ts))
            hourly_usage[hour] += usage
            total_hourly_usage[hour] += usage
            total_usage += usage

        if total_usage <= 0:
            continue

        peak_usage = sum(hourly_usage[hour] for hour in PEAK_HOURS)
        night_usage = sum(hourly_usage[hour] for hour in NIGHT_HOURS)
        dominant_hour = max(hourly_usage, key=hourly_usage.get)

        appliance_summaries.append(
            ApplianceBehaviorSummary(
                app_id=app_id,
                label=label,
                total_usage=round(total_usage, 2),
                hourly_usage={hour: round(value, 4) for hour, value in hourly_usage.items() if value > 0},
                peak_usage=round(peak_usage, 2),
                peak_share_percent=round((peak_usage / total_usage) * 100, 1) if total_usage else 0.0,
                night_usage=round(night_usage, 2),
                night_share_percent=round((night_usage / total_usage) * 100, 1) if total_usage else 0.0,
                dominant_hour=dominant_hour,
            )
        )

    total_usage = round(sum(total_hourly_usage.values()), 2)
    peak_usage = round(sum(total_hourly_usage[hour] for hour in PEAK_HOURS), 2)
    night_usage = round(sum(total_hourly_usage[hour] for hour in NIGHT_HOURS), 2)
    peak_share_percent = round((peak_usage / total_usage) * 100, 1) if total_usage else 0.0
    night_share_percent = round((night_usage / total_usage) * 100, 1) if total_usage else 0.0

    patterns: List[str] = []

    def find_summary(label: str) -> Optional[ApplianceBehaviorSummary]:
        for summary in appliance_summaries:
            if summary.label.lower() == label.lower():
                return summary
        return None

    ac_summary = find_summary("AC")
    if ac_summary and ac_summary.peak_share_percent >= 35:
        patterns.append("evening AC user")

    always_on_summary = find_summary("always-on devices")
    if always_on_summary and always_on_summary.night_share_percent >= 20:
        patterns.append("high always-on baseline")

    if night_share_percent >= 22:
        patterns.append("night owl usage")

    if peak_share_percent >= 30 and "evening AC user" not in patterns:
        patterns.append("evening-heavy appliance usage")

    if peak_share_percent >= 30:
        peak_text = "A large portion of your usage happens in the evening hours."
    elif peak_share_percent >= 20:
        peak_text = "A noticeable share of your usage happens in the evening peak window."
    else:
        peak_text = "Your usage appears relatively spread out across the day."

    return BehaviorInsights(
        behavior_patterns=patterns,
        peak_cost_awareness_text=peak_text,
        peak_window="6 PM to 9 PM",
        total_usage=total_usage,
        peak_usage=peak_usage,
        peak_share_percent=peak_share_percent,
        night_usage=night_usage,
        night_share_percent=night_share_percent,
        appliance_summaries=appliance_summaries,
    )

# =============================================================================
# PARSERS
# =============================================================================

def parse_user_profile(raw: Dict[str, Any], user_id: str) -> UserProfile:
    payload = raw["payload"]
    home = payload.get("homeAccounts", {}) or {}

    first_name = payload.get("firstName")
    last_name = payload.get("lastName")
    full_name = " ".join([x for x in [first_name, last_name] if x]).strip() or None

    return UserProfile(
        user_id=user_id,
        full_name=full_name,
        zipcode=home.get("postalCode"),
        latitude=home.get("latitude"),
        longitude=home.get("longitude"),
        timezone=home.get("timeZone"),
        email=payload.get("email"),
        raw=raw,
    )


def parse_endpoint_info(raw: Dict[str, Any]) -> EndpointInfo:
    payload = raw["payload"]
    if not payload:
        raise ValueError("No endpoints found in endpoint response")

    endpoint = payload[0]
    endpoint_id = endpoint.get("endpointId")
    if not endpoint_id:
        raise ValueError("endpointId missing in endpoint response")

    return EndpointInfo(
        endpoint_id=endpoint_id,
        measurement_type=endpoint.get("measurementType"),
        profile=endpoint.get("profile"),
        raw=endpoint,
    )


def parse_billing_cycles(raw: Dict[str, Any]) -> List[BillingCycle]:
    cycles: List[BillingCycle] = []

    for _, cycle in raw.items():
        if not isinstance(cycle, dict):
            continue
        if "billingStartTs" not in cycle or "billingEndTs" not in cycle:
            continue

        invoice = cycle.get("invoiceDataList", [{}])
        invoice0 = invoice[0] if invoice else {}

        billing_start_ts = int(cycle["billingStartTs"])
        billing_end_ts = int(cycle["billingEndTs"])

        cycles.append(
            BillingCycle(
                billing_start_ts=billing_start_ts,
                billing_end_ts=billing_end_ts,
                start_date=epoch_to_date_str(billing_start_ts),
                end_date=epoch_to_date_str(billing_end_ts),
                cost=float(cycle.get("cost", 0.0) or 0.0),
                consumption_kwh=float(cycle.get("value", 0.0) or 0.0),
                bidgely_generated_invoice=bool(cycle.get("bidgelyGeneratedInvoice", False)),
                estimation_type=invoice0.get("estimationType"),
                user_type=cycle.get("userType"),
                raw=cycle,
            )
        )

    cycles.sort(key=lambda c: c.billing_end_ts)
    return cycles


def select_latest_non_bidgely_cycle(cycles: List[BillingCycle]) -> Tuple[BillingCycle, Optional[BillingCycle]]:
    eligible = [c for c in cycles if not c.bidgely_generated_invoice]
    if not eligible:
        raise ValueError("No billing cycle found with bidgelyGeneratedInvoice=false")

    eligible.sort(key=lambda c: c.billing_end_ts, reverse=True)
    selected = eligible[0]
    previous = eligible[1] if len(eligible) > 1 else None
    return selected, previous


def parse_itemization_cycle(raw: Dict[str, Any], start_date: str, end_date: str) -> ItemizationCycle:
    details = raw["payload"]["itemizationDetails"]

    matched = None
    for cycle in details:
        if cycle.get("startDate") == start_date and cycle.get("endDate") == end_date:
            matched = cycle
            break

    if not matched:
        raise ValueError(f"No itemization cycle found for {start_date} to {end_date}")

    electric = matched.get("electric", [])
    appliances: List[ApplianceBreakdown] = []

    total_usage = 0.0
    total_cost = 0.0

    for row in electric:
        category = row.get("category")
        usage = float(row.get("usage", 0.0) or 0.0)
        cost = float(row.get("cost", 0.0) or 0.0)
        percentage = float(row.get("percentage", 0.0) or 0.0)
        cost_percentage = row.get("costPercentage")
        cost_percentage = float(cost_percentage) if cost_percentage is not None else None

        appliance = ApplianceBreakdown(
            category=category,
            label=category_to_label(category),
            usage_kwh=usage,
            cost=cost,
            percentage=percentage,
            cost_percentage=cost_percentage,
            is_estimated=row.get("isEstimated"),
        )
        appliances.append(appliance)

        if category == "total":
            total_usage = usage
            total_cost = cost

    return ItemizationCycle(
        start_ts=int(float(matched["startTs"])),
        end_ts=int(float(matched["endTs"])),
        start_date=matched["startDate"],
        end_date=matched["endDate"],
        appliances=appliances,
        total_usage_kwh=total_usage,
        total_cost=total_cost,
        raw=matched,
    )


def summarize_weather(raw: Dict[str, Any], start_ts: int, end_ts: int) -> WeatherSummary:
    days: List[WeatherDay] = []

    for ts_str, vals in raw.items():
        ts = int(ts_str)
        if start_ts <= ts <= end_ts:
            days.append(
                WeatherDay(
                    ts=ts,
                    min_temp=float(vals["minTemp"]) if vals.get("minTemp") is not None else None,
                    max_temp=float(vals["maxTemp"]) if vals.get("maxTemp") is not None else None,
                    avg_temp=float(vals["avgTemp"]) if vals.get("avgTemp") is not None else None,
                )
            )

    days.sort(key=lambda d: d.ts)

    avg_temps = [d.avg_temp for d in days if d.avg_temp is not None]
    max_temps = [d.max_temp for d in days if d.max_temp is not None]
    min_temps = [d.min_temp for d in days if d.min_temp is not None]

    return WeatherSummary(
        start_ts=start_ts,
        end_ts=end_ts,
        total_days=len(days),
        avg_temp_f=safe_round(sum(avg_temps) / len(avg_temps), 2) if avg_temps else None,
        max_temp_f=safe_round(max(max_temps), 1) if max_temps else None,
        min_temp_f=safe_round(min(min_temps), 1) if min_temps else None,
        days_above_95f=sum(1 for d in days if d.max_temp is not None and d.max_temp > 95),
        days_above_90f=sum(1 for d in days if d.max_temp is not None and d.max_temp > 90),
        days_below_40f=sum(1 for d in days if d.min_temp is not None and d.min_temp < 40),
        warm_days_80_plus=sum(1 for d in days if d.max_temp is not None and d.max_temp >= 80),
        raw_days=days,
    )

def _parse_recommendation_content(data: Dict[str, Any], fallback: RecommendationContent) -> RecommendationContent:
    if not isinstance(data, dict):
        return fallback

    return RecommendationContent(
        intrigue_html=str(data.get("intrigue_html") or fallback.intrigue_html).strip(),
        action_html=str(data.get("action_html") or fallback.action_html).strip(),
        insight_html=str(data.get("insight_html") or fallback.insight_html).strip(),
    )


def parse_recommendation_tips(content: str) -> RecommendationTips:
    fallback = RecommendationTips(
        this_week=RecommendationContent(
            intrigue_html='A tiny change this week could unlock <strong style="color: #667eea;">noticeable savings</strong>.',
            action_html='Adjust your thermostat by 1-2°F during the busiest hours this week.',
            insight_html='Your recent usage pattern suggests small comfort-setting changes can lower avoidable energy use.',
        ),
        this_month=RecommendationContent(
            intrigue_html='One low-effort upgrade this month could trim <strong style="color: #667eea;">ongoing costs</strong>.',
            action_html='Swap the most-used bulbs to LEDs and seal any obvious air leaks.',
            insight_html='A meaningful share of your usage appears to come from everyday baseline consumption across the home.',
        ),
        this_year=RecommendationContent(
            intrigue_html='A bigger efficiency step this year could create <strong style="color: #667eea;">lasting savings</strong>.',
            action_html='Plan a seasonal HVAC tune-up or consider a smart thermostat upgrade.',
            insight_html='Heating and cooling patterns often create the largest long-term savings opportunities over a full year.',
        ),
    )

    if content is None:
        return fallback

    content = content.strip()
    if not content:
        return fallback

    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    elif content.startswith("```"):
        content = content[len("```"):].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        content = match.group(0).strip()

    try:
        data = json.loads(content)
    except Exception:
        return fallback

    tips = data.get("recommendation_tips") if isinstance(data, dict) else None
    if not isinstance(tips, dict):
        return fallback

    return RecommendationTips(
        this_week=_parse_recommendation_content(tips.get("this_week"), fallback.this_week),
        this_month=_parse_recommendation_content(tips.get("this_month"), fallback.this_month),
        this_year=_parse_recommendation_content(tips.get("this_year"), fallback.this_year),
    )

def parse_behavior_summary_content(data: Any, fallback: Optional[BehaviorSummaryContent] = None) -> BehaviorSummaryContent:
    fallback = fallback or BehaviorSummaryContent(
        patterns=[],
        peak_cost_awareness="A large portion of your usage happens in the evening hours.",
    )

    if not isinstance(data, dict):
        return fallback

    raw_patterns = data.get("patterns")
    patterns: List[BehaviorPatternContent] = []
    if isinstance(raw_patterns, list):
        for item in raw_patterns:
            if isinstance(item, dict):
                title = str(item.get("title") or "").strip()
                description = str(item.get("description") or "").strip()
                if title or description:
                    patterns.append(
                        BehaviorPatternContent(
                            title=title or "Behavior pattern",
                            description=description or "This pattern reflects how energy use is distributed across the day.",
                        )
                    )
            elif isinstance(item, str) and item.strip():
                label = item.strip()
                patterns.append(
                    BehaviorPatternContent(
                        title=label.title(),
                        description=f"This pattern indicates {label.lower()} in your energy usage.",
                    )
                )
    elif isinstance(raw_patterns, str) and raw_patterns.strip():
        label = raw_patterns.strip()
        patterns = [
            BehaviorPatternContent(
                title=label.title(),
                description=f"This pattern indicates {label.lower()} in your energy usage.",
            )
        ]

    peak_cost_awareness = str(data.get("peak_cost_awareness") or fallback.peak_cost_awareness).strip()
    return BehaviorSummaryContent(patterns=patterns, peak_cost_awareness=peak_cost_awareness)


def parse_header_content(content: str) -> HeaderContent:
    fallback = HeaderContent(
        subject_line="Your personalized energy update",
        greeting_text="Here’s a quick look at your latest energy trends."
    )

    if content is None:
        return fallback

    content = content.strip()

    if not content:
        return fallback

    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    elif content.startswith("```"):
        content = content[len("```"):].strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    # Extract the first JSON object if the model added extra prose
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        content = match.group(0).strip()

    try:
        data = json.loads(content)
    except Exception:
        return fallback

    subject_line = str(data.get("subject_line", "")).strip()
    greeting_text = str(data.get("greeting_text", "")).strip()

    if not subject_line:
        subject_line = fallback.subject_line
    if not greeting_text:
        greeting_text = fallback.greeting_text

    return HeaderContent(
        subject_line=subject_line,
        greeting_text=greeting_text,
    )


def parse_email_sections(content: str) -> EmailSections:
    header = HeaderContent(
        subject_line="Your personalized energy update",
        greeting_text="Here’s a quick look at your latest energy trends.",
    )
    fallback = EmailSections(
        header=header,
        tone="casual",
        energy_story="<p>We have your latest energy update ready.</p>",
        energy_breakdown="Your top appliances continue to shape most of your usage this cycle.",
        recommendation_tips=RecommendationTips(
            this_week=RecommendationContent(
                intrigue_html='A small shift this week could unlock <strong style="color: #667eea;">quick savings</strong>.',
                action_html='Adjust your thermostat slightly during the highest-usage hours this week.',
                insight_html='Your recent consumption pattern suggests a short-term opportunity to trim avoidable usage.',
            ),
            this_month=RecommendationContent(
                intrigue_html='A simple home tweak this month could reduce <strong style="color: #667eea;">ongoing waste</strong>.',
                action_html='Focus on LEDs, weatherstripping, or reducing standby power this month.',
                insight_html='Part of your bill appears to come from steady everyday usage that small upgrades can improve.',
            ),
            this_year=RecommendationContent(
                intrigue_html='A bigger annual upgrade could create <strong style="color: #667eea;">long-term value</strong>.',
                action_html='Plan an HVAC tune-up or a smart thermostat upgrade this year.',
                insight_html='Longer-term efficiency improvements are often most effective where heating and cooling drive a large share of use.',
            ),
        ),
        behavior_summary=BehaviorSummaryContent(
            patterns=[],
            peak_cost_awareness='A large portion of your usage happens in the evening hours.',
        ),
    )

    if content is None:
        return fallback

    content = content.strip()
    if not content:
        return fallback

    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    elif content.startswith("```"):
        content = content[len("```"):].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        content = match.group(0).strip()

    try:
        data = json.loads(content)
    except Exception:
        return fallback

    subject_line = str(data.get("subject_line") or fallback.header.subject_line).strip()
    greeting_text = str(data.get("greeting_text") or fallback.header.greeting_text).strip()
    tone = str(data.get("tone") or fallback.tone).strip().lower()
    if tone not in {"casual", "excited", "formal", "attention"}:
        tone = fallback.tone

    energy_story = str(data.get("energy_story") or fallback.energy_story).strip()
    energy_breakdown = str(data.get("energy_breakdown") or fallback.energy_breakdown).strip()

    tips = data.get("recommendation_tips") or {}
    recommendation_tips = RecommendationTips(
        this_week=_parse_recommendation_content(tips.get("this_week"), fallback.recommendation_tips.this_week),
        this_month=_parse_recommendation_content(tips.get("this_month"), fallback.recommendation_tips.this_month),
        this_year=_parse_recommendation_content(tips.get("this_year"), fallback.recommendation_tips.this_year),
    )
    behavior_summary = parse_behavior_summary_content(data.get("behavior_summary"), fallback.behavior_summary)

    return EmailSections(
        header=HeaderContent(subject_line=subject_line, greeting_text=greeting_text),
        tone=tone,
        energy_story=energy_story,
        energy_breakdown=energy_breakdown,
        recommendation_tips=recommendation_tips,
        behavior_summary=behavior_summary,
    )


def extract_token_usage(response: Any) -> Dict[str, Any]:
    usage = {}

    try:
        response_metadata = getattr(response, "response_metadata", None) or {}
        usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    except Exception:
        usage = {}

    if not usage:
        try:
            usage = getattr(response, "usage_metadata", None) or {}
        except Exception:
            usage = {}

    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    normalized = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "raw": usage,
    }
    return normalized


def log_token_usage(usage: Dict[str, Any]) -> None:
    logger.info(MODEL_NAME)
    logger.info(
        "LLM token usage -> prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )


# =============================================================================
# PROMPTS
# =============================================================================

EMAIL_CONTENT_TEMPLATE = """
You are generating a personalized residential energy email.

Return valid JSON only with exactly this schema:
{{
  "subject_line": "...",
  "greeting_text": "...",
  "tone": "casual | excited | formal | attention",
  "energy_story": "<p>...</p>",
  "energy_breakdown": "...",
  "behavior_summary": {{
    "patterns": [
      {{
        "title": "...",
        "description": "..."
      }}
    ],
    "peak_cost_awareness": "..."
  }},
  "recommendation_tips": {{
    "this_week": {{
      "intrigue_html": "...",
      "action_html": "...",
      "insight_html": "..."
    }},
    "this_month": {{
      "intrigue_html": "...",
      "action_html": "...",
      "insight_html": "..."
    }},
    "this_year": {{
      "intrigue_html": "...",
      "action_html": "...",
      "insight_html": "..."
    }}
  }}
}}

CORE RULES:
- Use ONLY the provided facts. Do NOT invent values, causes, or behaviors.
- Keep language simple, clear, and user-friendly.
- Avoid repetition and generic phrasing.
- Use simple appliance names only (e.g., AC, heater, fridge).
- Do NOT use technical jargon.

HEADER RULES:
subject_line:
- Under 60 characters
- Warm, engaging, and action-oriented

greeting_text:
- Exactly 1 short sentence
- Friendly tone
- No HTML

TONE SELECTION:
Choose exactly one: casual, excited, formal, attention
- Minimum Usage spike → formal
- Abnormally High Usage spike → attention
- Savings or positive trend → excited
- Neutral summary → casual

ENERGY STORY (HTML ONLY):
- Use ONLY <p> and <strong style="color: ...;">
- Maximum 3 short paragraphs
- Include weather related fact
- Be concise and fact-driven
- Keep the text inline with the tone
Color usage:
- Good news → #38A169
- Higher usage/cost → #E53E3E
- Key values → #667eea
- If notification_type is bill_projection, clearly mention values are projected (not final)

ENERGY BREAKDOWN:
- Plain text only, inline with the tone
- 1–2 short sentences
- Focus only on top contributing appliances

BEHAVIOR SUMMARY:
- Use behavior facts ONLY if provided

patterns:
- List 1–3 short time-of-day usage patterns
- Example style: "Higher AC usage in late evening"

peak_cost_awareness:
- 1 short sentence about usage during peak hours, good if usage is low, inline with the tone
- Must be derived from facts (time-of-day or usage pattern)
- MUST NOT reuse or copy any example phrasing from input
- MUST be rephrased in fresh, natural language

If behavior facts are NOT available:
- patterns: []
- peak_cost_awareness: "Your energy usage appears evenly distributed throughout the day."

STRICT RULES:
- Do NOT infer patterns (e.g., weekends) unless explicitly present
- Do NOT use date-specific statements
- Only describe time-of-day behavior

RECOMMENDATIONS (STRICT):
For each: this_week, this_month, this_year
Each must include:
- intrigue_html
- action_html
- insight_html

HTML RULES (MANDATORY):
- Return ONLY short HTML fragments
- Do NOT use <p>, <div>, <span>
- Only allowed styling: <strong style="color: #667eea;">...</strong>

CONTENT RULES FOR RECOMMENDATIONS:
intrigue_html:
- Curiosity-driven sentence
- MUST include appliance name

action_html:
- Clear, specific action
- MUST include appliance name
- Avoid generic advice

insight_html:
- Explain WHY this matters for THIS user
- Must tie directly to provided facts

HARD CONSTRAINTS:
- Do NOT reuse generic or example recommendations
- Do NOT introduce appliances not present in facts
- Do NOT hallucinate savings or behavior
- Do NOT exceed length limits
- Do NOT include extra text outside JSON
- Do NOT simply reuse the example recommendations

FINAL CHECK:
- Output is valid JSON
- All fields are filled
- HTML rules are strictly followed
- Tone matches the facts
- Recommendations are specific and personalized

Tone guidance:
tone_options={tone_options}

Style guidance:
{style_config}

Notification type:
{notification_type}

Facts:
user={user_facts}
billing={billing_facts}
itemization={itemization_facts}
weather={weather_facts}
insights={insight_facts}
bill_projection={bill_projection_facts}
behavior={behavior_facts}
recommendations={recommendation_facts}
""".strip()

# =============================================================================
# LLM
# =============================================================================

def build_llm() -> ChatOpenAI:
    # return ChatOllama(
    #     model=OLLAMA_MODEL,
    #     base_url=OLLAMA_BASE_URL,
    #     temperature=0.2,
    # )
    return ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.7,
    timeout=30,
    )


# =============================================================================
# AGENT
# =============================================================================

class EnergyEmailAgent:
    def __init__(self, client: BidgelyClient, llm: Optional[ChatOpenAI] = None):
        self.client = client
        self.llm = llm or build_llm()
        self.email_content_chain = ChatPromptTemplate.from_template(EMAIL_CONTENT_TEMPLATE) | self.llm

        self.bill_projection_tool = self._build_bill_projection_tool()
        try:
            self.bill_projection_tool_llm = ChatOpenAI(model=MODEL_NAME,temperature=0,).bind_tools([self.bill_projection_tool])
        except Exception:
            logger.exception("LLM bind_tools failed; bill projection tool flow will fall back to prompt-only generation.")
            self.bill_projection_tool_llm = None

    def _build_bill_projection_tool(self):
        client = self.client

        @tool("get_bill_projection")
        def get_bill_projection(
            user_id: str,
            home_id: str,
            measurement_type: str,
            billing_start_ts: int,
            billing_end_ts: int,
        ) -> str:
            """Fetch the latest projected bill for the in-progress billing cycle."""
            raw = client.fetch_bill_projection(
                user_id=user_id,
                home_id=home_id,
                measurement_type=measurement_type,
                billing_start_ts=billing_start_ts,
                billing_end_ts=billing_end_ts,
            )
            parsed = parse_bill_projection_api_response(raw)
            return json.dumps(parsed, indent=2)

        return get_bill_projection

    def _populate_bill_projection_facts(self, state: AgentState) -> AgentState:
        try:
            tool_result = self.bill_projection_tool.invoke({
                "user_id": state.user_profile.user_id,
                "home_id": state.home_id,
                "measurement_type": state.measurement_type,
                "billing_start_ts": state.selected_bill_cycle.billing_start_ts,
                "billing_end_ts": state.selected_bill_cycle.billing_end_ts,
            })
            tool_data = json.loads(tool_result)
            state.bill_projection_facts_obj = merge_bill_projection_facts(
                state.bill_projection_facts_obj,
                tool_data,
                state.previous_bill_cycle,
            )
        except Exception:
            logger.exception("Failed to fetch bill projection via Python tool wrapper; continuing with existing facts.")
        return state


    def load_base_data(self, user_id: str, home_id: str, measurement_type: str) -> AgentState:
        state = AgentState()
        state.home_id = home_id
        state.measurement_type = measurement_type

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.client.fetch_user_details, user_id): "user",
                executor.submit(self.client.fetch_user_endpoints, user_id): "endpoints",
                executor.submit(
                    self.client.fetch_billing_details,
                    user_id,
                    home_id,
                    BILLING_T0,
                    BILLING_T1,
                    measurement_type,
                ): "billing",
            }

            for future in as_completed(futures):
                key = futures[future]
                result = future.result()
                if key == "user":
                    state.user_raw = result
                elif key == "endpoints":
                    state.endpoints_raw = result
                elif key == "billing":
                    state.billing_raw = result

        state.user_profile = parse_user_profile(state.user_raw, user_id)
        state.endpoint = parse_endpoint_info(state.endpoints_raw)

        billing_cycles = parse_billing_cycles(state.billing_raw)
        state.selected_bill_cycle, state.previous_bill_cycle = select_latest_non_bidgely_cycle(billing_cycles)

        return state

    def load_dependent_data(self, state: AgentState, user_id: str, measurement_type: str) -> AgentState:
        if not state.endpoint or not state.selected_bill_cycle or not state.user_profile:
            raise ValueError("Base data must be loaded before dependent data.")

        state.itemization_raw = self.client.fetch_itemization(
            user_id=user_id,
            endpoint_id=state.endpoint.endpoint_id,
            from_date=ITEMIZATION_FROM_DATE,
            to_date=ITEMIZATION_TO_DATE,
            measurement_type=measurement_type,
        )

        state.itemization_cycle = parse_itemization_cycle(
            state.itemization_raw,
            start_date=state.selected_bill_cycle.start_date,
            end_date=state.selected_bill_cycle.end_date,
        )

        if not state.user_profile.zipcode:
            raise ValueError("zipcode missing from user profile; cannot fetch weather")

        state.weather_raw = self.client.fetch_weather(
            country_code="US",
            zipcode=state.user_profile.zipcode,
            t0=state.selected_bill_cycle.billing_start_ts,
            t1=state.selected_bill_cycle.billing_end_ts,
        )

        state.weather_summary = summarize_weather(
            state.weather_raw,
            start_ts=state.selected_bill_cycle.billing_start_ts,
            end_ts=state.selected_bill_cycle.billing_end_ts,
        )
        logger.info(
            "Weather summarized only for selected bill cycle start_ts=%s end_ts=%s total_days=%s",
            state.selected_bill_cycle.billing_start_ts,
            state.selected_bill_cycle.billing_end_ts,
            state.weather_summary.total_days,
        )

        return state

    def derive(self, state: AgentState) -> AgentState:
        if not all([state.selected_bill_cycle, state.itemization_cycle, state.weather_summary]):
            raise ValueError("Cannot derive insights before all dependent data is loaded.")

        state.insights = derive_insights(
            selected_bill=state.selected_bill_cycle,
            previous_bill=state.previous_bill_cycle,
            itemization=state.itemization_cycle,
            weather=state.weather_summary,
        )
        state.style_config = build_style_config("casual")

        if state.notification_type == NOTIFICATION_TYPE_BILL_PROJECTION:
            state.bill_projection_facts_obj = derive_bill_projection_facts(
                selected_bill=state.selected_bill_cycle,
                previous_bill=state.previous_bill_cycle,
                itemization=state.itemization_cycle,
                weather=state.weather_summary,
            )

        try:
            if TBAPPDATA_FILE_PATH and os.path.exists(TBAPPDATA_FILE_PATH):
                tb_raw = load_tbappdata_file(TBAPPDATA_FILE_PATH)
                state.behavior_insights = derive_behavior_insights_from_tbappdata(
                    tb_raw=tb_raw,
                    itemization=state.itemization_cycle,
                )
            else:
                logger.info("TB app data file not found at path=%s; skipping behavior analysis", TBAPPDATA_FILE_PATH)
        except Exception:
            logger.exception("Failed to derive behavior insights from TB app data; continuing without behavior facts.")
            state.behavior_insights = None

        return state

    def generate_sections(self, state: AgentState) -> AgentState:
        if not all([
            state.selected_bill_cycle,
            state.itemization_cycle,
            state.weather_summary,
            state.insights,
            state.style_config,
        ]):
            raise ValueError("Missing state for section generation.")

        if state.notification_type == NOTIFICATION_TYPE_BILL_PROJECTION:
            state = self._populate_bill_projection_facts(state)

        billing_facts = to_compact_json(
            build_billing_facts(state.selected_bill_cycle, state.previous_bill_cycle, state.insights)
        )
        itemization_facts = to_compact_json(build_itemization_facts(state.itemization_cycle))
        weather_facts = to_compact_json(build_weather_facts(state.weather_summary))
        insight_facts = to_compact_json(build_insight_facts(state.insights))
        behavior_facts = to_compact_json(build_behavior_facts(state.behavior_insights))
        style_config = to_compact_json(state.style_config)
        bill_projection_facts = to_compact_json(build_bill_projection_facts(state.bill_projection_facts_obj))
        recommendation_facts = to_compact_json(build_recommendation_facts())
        user_facts = to_compact_json(build_user_facts(state.user_profile))

        tone_options = to_compact_json({
            "casual": build_style_config("casual"),
            "excited": build_style_config("excited"),
            "formal": build_style_config("formal"),
            "attention": build_style_config("attention"),
        })

        response = self.email_content_chain.invoke({
            "style_config": style_config,
            "tone_options": tone_options,
            "notification_type": state.notification_type,
            "llm_usage": state.llm_usage,
            "user_facts": user_facts,
            "billing_facts": billing_facts,
            "itemization_facts": itemization_facts,
            "weather_facts": weather_facts,
            "insight_facts": insight_facts,
            "behavior_facts": behavior_facts,
            "bill_projection_facts": bill_projection_facts,
            "recommendation_facts": recommendation_facts,
        })
        email_sections_raw = response.content.strip()
        state.llm_usage = extract_token_usage(response)
        log_token_usage(state.llm_usage)

        print("EMAIL SECTIONS RAW >>>", repr(email_sections_raw))
        state.sections = parse_email_sections(email_sections_raw)
        if state.sections and state.insights:
            state.insights.tone = state.sections.tone
            state.style_config = build_style_config(state.sections.tone)
        return state

    def run(
        self,
        user_id: str,
        home_id: str,
        measurement_type: str = "ELECTRIC",
        notification_type: str = NOTIFICATION_TYPE_MONTHLY_SUMMARY,
    ) -> AgentState:
        state = self.load_base_data(user_id, home_id, measurement_type)
        state.notification_type = notification_type
        state = self.load_dependent_data(state, user_id, measurement_type)
        state = self.derive(state)
        state = self.generate_sections(state)
        return state


# =============================================================================
# OUTPUT BUILDERS
# =============================================================================

def build_email_payload(state: AgentState) -> Dict[str, Any]:
    if not state.sections or not state.insights:
        raise ValueError("Sections or insights missing")

    payload = {
        "home_id": state.home_id,
        "user": build_user_facts(state.user_profile),
        "billing": build_billing_facts(state.selected_bill_cycle, state.previous_bill_cycle, state.insights),
        "itemization": build_itemization_facts(state.itemization_cycle),
        "weather": build_weather_facts(state.weather_summary),
        "insights": build_insight_facts(state.insights),
        "notification_type": state.notification_type,
        "llm_usage": state.llm_usage,
        "sections": {
            "subject_line": state.sections.header.subject_line,
            "greeting_text": state.sections.header.greeting_text,
            "tone": state.sections.tone,
            "bill_cycle_details": {
                "bill_start_ts": state.selected_bill_cycle.billing_start_ts,
                "bill_end_ts": state.selected_bill_cycle.billing_end_ts,
                "bill_start_date": state.selected_bill_cycle.start_date,
                "bill_end_date": state.selected_bill_cycle.end_date,
                "bill_amount": round(state.selected_bill_cycle.cost, 2),
                "bill_consumption_kwh": round(state.selected_bill_cycle.consumption_kwh, 2),
            },
            "itemization_details": build_itemization_details(state.itemization_cycle),
            "energy_story": state.sections.energy_story,
            "energy_breakdown": state.sections.energy_breakdown,
            "behavior_summary": {
                "patterns": [asdict(pattern) for pattern in state.sections.behavior_summary.patterns],
                "peak_cost_awareness": state.sections.behavior_summary.peak_cost_awareness,
            },
            "recommendation_tips": {
                "this_week": asdict(state.sections.recommendation_tips.this_week),
                "this_month": asdict(state.sections.recommendation_tips.this_month),
                "this_year": asdict(state.sections.recommendation_tips.this_year),
            },
        },
    }

    if state.notification_type == NOTIFICATION_TYPE_BILL_PROJECTION:
        payload["bill_projection"] = build_bill_projection_facts(state.bill_projection_facts_obj)

    return payload


# =============================================================================
# MAIN
# =============================================================================

# if __name__ == "__main__":
#     if BIDGELY_BEARER_TOKEN == "YOUR_BEARER_TOKEN":
#         raise RuntimeError("Set BIDGELY_BEARER_TOKEN before running")

#     client = BidgelyClient(BASE_URL, BIDGELY_BEARER_TOKEN)
#     agent = EnergyEmailAgent(client)

#     state = agent.run(
#         user_id=USER_ID,
#         home_id=HOME_ID,
#         measurement_type=MEASUREMENT_TYPE,
#     )

#     payload = build_email_payload(state)

#     print("\n=== USER ===")
#     print(json.dumps(payload["user"], indent=2))

#     print("\n=== BILLING ===")
#     print(json.dumps(payload["billing"], indent=2))

#     print("\n=== ITEMIZATION ===")
#     print(json.dumps(payload["itemization"], indent=2))

#     print("\n=== WEATHER ===")
#     print(json.dumps(payload["weather"], indent=2))

#     print("\n=== INSIGHTS ===")
#     print(json.dumps(payload["insights"], indent=2))

#     print("\n=== STYLE ===")
#     print(json.dumps(payload["style"], indent=2))

#     print("\n=== ENERGY STORY ===")
#     print(payload["sections"]["energy_story"])

#     print("\n=== ENERGY BREAKDOWN ===")
#     print(payload["sections"]["energy_breakdown"])

#     print("\n=== ENERGY TIPS ===")
#     print(payload["sections"]["recommendation_tips"])

def validate_message(body: dict) -> None:
    required_fields = ["user_id", "home_id"]
    for field in required_fields:
        if not body.get(field):
            raise ValueError(f"Missing required field: {field}")


def build_generic_event_data(payload: dict) -> Dict[str, Any]:
    user = payload.get("user") or {}
    sections = payload.get("sections") or {}
    recommendations = sections.get("recommendation_tips") or {}

    event_type = "MonthlySummary"
    if payload.get("notification_type") == NOTIFICATION_TYPE_BILL_PROJECTION:
        event_type = "BillProjection"
    elif payload.get("notification_type") == NOTIFICATION_TYPE_REGULAR:
        event_type = "Regular"

    event = {
        "uuid": user.get("user_id"),
        "hid": int(payload.get("home_id") or 1),
        "eventType": event_type,
        "userDeliveryModes": ["Email"],
        "llmEmailContent": {
            "subjectLine": sections.get("subject_line"),
            "greetingText": sections.get("greeting_text"),
            "tone": sections.get("tone"),
            "billCycleDetails": sections.get("bill_cycle_details") or {},
            "itemizationDetails": sections.get("itemization_details") or {},
            "energyStory": sections.get("energy_story"),
            "energyBreakdown": sections.get("energy_breakdown"),
            "behaviorSummary": sections.get("behavior_summary") or {},
            "recommendationTips": {
                "thisWeek": recommendations.get("this_week") or {},
                "thisMonth": recommendations.get("this_month") or {},
                "thisYear": recommendations.get("this_year") or {},
            },
            "tokenUsage": payload.get("llm_usage") or {},
        },
    }

    if payload.get("notification_type") == NOTIFICATION_TYPE_BILL_PROJECTION:
        event["billProjection"] = payload.get("bill_projection") or {}

    return event


def build_output_queue_message(payload: dict) -> str:
    generic_event_data = build_generic_event_data(payload)

    root = ET.Element("genericEventData")

    def add_text(parent: ET.Element, tag: str, value: Any) -> None:
        if value is None:
            return
        child = ET.SubElement(parent, tag)
        child.text = str(value)

    add_text(root, "uuid", generic_event_data.get("uuid"))
    add_text(root, "hid", generic_event_data.get("hid"))
    add_text(root, "eventType", generic_event_data.get("eventType"))

    bill_projection = generic_event_data.get("billProjection") or {}
    if generic_event_data.get("eventType") == "BillProjection" and bill_projection:
        bill_projection_el = ET.SubElement(root, "billProjection")
        add_text(bill_projection_el, "cycleStartDate", bill_projection.get("cycle_start_date"))
        add_text(bill_projection_el, "cycleEndDate", bill_projection.get("cycle_end_date"))
        add_text(bill_projection_el, "daysObserved", bill_projection.get("days_observed"))
        add_text(bill_projection_el, "daysRemaining", bill_projection.get("days_remaining"))
        add_text(bill_projection_el, "observedUsageKwh", bill_projection.get("observed_usage_kwh"))
        add_text(bill_projection_el, "observedCost", bill_projection.get("observed_cost"))
        add_text(bill_projection_el, "projectedRemainingUsageKwh", bill_projection.get("projected_remaining_usage_kwh"))
        add_text(bill_projection_el, "projectedTotalUsageKwh", bill_projection.get("projected_total_usage_kwh"))
        add_text(bill_projection_el, "projectedTotalCost", bill_projection.get("projected_total_cost"))
        add_text(bill_projection_el, "projectedVsExpectedAmount", bill_projection.get("projected_vs_expected_amount"))
        add_text(bill_projection_el, "projectedVsExpectedPercent", bill_projection.get("projected_vs_expected_percent"))
        add_text(bill_projection_el, "projectedVsLastBillAmount", bill_projection.get("projected_vs_last_bill_amount"))
        add_text(bill_projection_el, "projectedVsLastBillPercent", bill_projection.get("projected_vs_last_bill_percent"))
        add_text(bill_projection_el, "projectedDominantAppliance", bill_projection.get("projected_dominant_appliance"))
        add_text(bill_projection_el, "projectedDominantApplianceSharePercent", bill_projection.get("projected_dominant_appliance_share_percent"))
        add_text(bill_projection_el, "weatherSignal", bill_projection.get("weather_signal"))

    for mode in generic_event_data.get("userDeliveryModes") or []:
        add_text(root, "userDeliveryModes", mode)

    llm_email_content = generic_event_data.get("llmEmailContent") or {}
    llm_el = ET.SubElement(root, "llmEmailContent")
    add_text(llm_el, "subjectLine", llm_email_content.get("subjectLine"))
    add_text(llm_el, "greetingText", llm_email_content.get("greetingText"))
    add_text(llm_el, "tone", llm_email_content.get("tone"))

    bill_cycle_details = llm_email_content.get("billCycleDetails") or {}
    bill_cycle_el = ET.SubElement(llm_el, "billCycleDetails")
    add_text(bill_cycle_el, "billStartTs", bill_cycle_details.get("bill_start_ts"))
    add_text(bill_cycle_el, "billEndTs", bill_cycle_details.get("bill_end_ts"))
    add_text(bill_cycle_el, "billStartDate", bill_cycle_details.get("bill_start_date"))
    add_text(bill_cycle_el, "billEndDate", bill_cycle_details.get("bill_end_date"))
    add_text(bill_cycle_el, "billAmount", bill_cycle_details.get("bill_amount"))
    add_text(bill_cycle_el, "billConsumptionKwh", bill_cycle_details.get("bill_consumption_kwh"))

    itemization_details = llm_email_content.get("itemizationDetails") or {}
    itemization_el = ET.SubElement(llm_el, "itemizationDetails")
    add_text(itemization_el, "startTs", itemization_details.get("start_ts"))
    add_text(itemization_el, "endTs", itemization_details.get("end_ts"))
    add_text(itemization_el, "startDate", itemization_details.get("start_date"))
    add_text(itemization_el, "endDate", itemization_details.get("end_date"))
    add_text(itemization_el, "totalCost", itemization_details.get("total_cost"))
    add_text(itemization_el, "totalUsageKwh", itemization_details.get("total_usage_kwh"))

    for appliance in itemization_details.get("appliances") or []:
        appliance_el = ET.SubElement(itemization_el, "appliance")
        add_text(appliance_el, "category", appliance.get("category"))
        add_text(appliance_el, "label", appliance.get("label"))
        add_text(appliance_el, "usageKwh", appliance.get("usage_kwh"))
        add_text(appliance_el, "cost", appliance.get("cost"))
        add_text(appliance_el, "percentage", appliance.get("percentage"))
        add_text(appliance_el, "costPercentage", appliance.get("cost_percentage"))
        add_text(appliance_el, "isEstimated", appliance.get("is_estimated"))

    add_text(llm_el, "energyStory", llm_email_content.get("energyStory"))
    add_text(llm_el, "energyBreakdown", llm_email_content.get("energyBreakdown"))

    behavior_summary = llm_email_content.get("behaviorSummary") or {}
    behavior_summary_el = ET.SubElement(llm_el, "behaviorSummary")
    for pattern in behavior_summary.get("patterns") or []:
        pattern_el = ET.SubElement(behavior_summary_el, "pattern")
        if isinstance(pattern, dict):
            add_text(pattern_el, "title", pattern.get("title"))
            add_text(pattern_el, "description", pattern.get("description"))
        else:
            add_text(pattern_el, "title", str(pattern))
            add_text(pattern_el, "description", f"This pattern indicates {str(pattern).lower()} in your energy usage.")
    add_text(behavior_summary_el, "peakCostAwareness", behavior_summary.get("peak_cost_awareness"))

    recommendation_tips = llm_email_content.get("recommendationTips") or {}
    recommendation_el = ET.SubElement(llm_el, "recommendationTips")

    def add_recommendation(parent: ET.Element, tag: str, recommendation: Dict[str, Any]) -> None:
        rec_el = ET.SubElement(parent, tag)
        add_text(rec_el, "intrigueHtml", recommendation.get("intrigue_html"))
        add_text(rec_el, "actionHtml", recommendation.get("action_html"))
        add_text(rec_el, "insightHtml", recommendation.get("insight_html"))

    add_recommendation(recommendation_el, "thisWeek", recommendation_tips.get("thisWeek") or {})
    add_recommendation(recommendation_el, "thisMonth", recommendation_tips.get("thisMonth") or {})
    add_recommendation(recommendation_el, "thisYear", recommendation_tips.get("thisYear") or {})

    token_usage = llm_email_content.get("tokenUsage") or {}
    token_usage_el = ET.SubElement(llm_el, "tokenUsage")
    add_text(token_usage_el, "promptTokens", token_usage.get("prompt_tokens"))
    add_text(token_usage_el, "completionTokens", token_usage.get("completion_tokens"))
    add_text(token_usage_el, "totalTokens", token_usage.get("total_tokens"))

    xml_body = ET.tostring(root, encoding="unicode", method="xml")
    logger.info("Output queue XML body: %s", xml_body)
    return xml_body


def send_to_output_queue(sqs_client: Any, payload: dict) -> None:
    if not OUTPUT_SQS_QUEUE_URL:
        logger.info("OUTPUT_SQS_QUEUE_URL not set; skipping downstream queue publish")
        return

    message_body = build_output_queue_message(payload)
    sqs_client.send_message(
        QueueUrl=OUTPUT_SQS_QUEUE_URL,
        MessageBody=message_body,
    )
    logger.info("Published generated payload XML to output queue user_id=%s queue=%s", payload.get("user", {}).get("user_id"), OUTPUT_SQS_QUEUE_URL)


def handle_generated_payload(payload: dict, sqs_client: Any) -> None:
    logger.info("Generated payload for user_id=%s", payload.get("user", {}).get("user_id"))
    logger.info("LLM output payload: %s", json.dumps(payload, separators=(",", ":")))
    send_to_output_queue(sqs_client, payload)


def process_message(agent: EnergyEmailAgent, sqs_client: Any, message_body: dict) -> None:
    validate_message(message_body)

    user_id = message_body["user_id"]
    home_id = message_body["home_id"]
    measurement_type = message_body.get("measurement_type", "ELECTRIC")
    notification_type = message_body.get("notification_type", NOTIFICATION_TYPE_MONTHLY_SUMMARY)

    logger.info(
        "Processing message user_id=%s home_id=%s notification_type=%s",
        user_id,
        home_id,
        notification_type,
    )

    state = agent.run(
        user_id=user_id,
        home_id=home_id,
        measurement_type=measurement_type,
        notification_type=notification_type,
    )

    payload = build_email_payload(state)
    handle_generated_payload(payload, sqs_client)


def poll_forever() -> None:
    if not SQS_QUEUE_URL:
        raise ValueError("SQS_QUEUE_URL env var is required")
    if not BASE_URL:
        raise ValueError("BASE_URL env var is required")
    if not BIDGELY_BEARER_TOKEN:
        raise ValueError("BIDGELY_BEARER_TOKEN env var is required")

    sqs = boto3.client("sqs", region_name=AWS_REGION)

    client = BidgelyClient(BASE_URL, BIDGELY_BEARER_TOKEN)
    agent = EnergyEmailAgent(client)

    logger.info("Starting SQS worker")
    logger.info("Queue URL: %s", SQS_QUEUE_URL)
    logger.info("Output Queue URL: %s", OUTPUT_SQS_QUEUE_URL)

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=MAX_MESSAGES,
                WaitTimeSeconds=POLL_WAIT_SECONDS,
                VisibilityTimeout=VISIBILITY_TIMEOUT,
                MessageAttributeNames=["All"],
            )

            messages = response.get("Messages", [])
            if not messages:
                logger.info("No message available..")
                continue

            for message in messages:
                receipt_handle = message["ReceiptHandle"]
                raw_body = message.get("Body", "{}")

                try:
                    body = json.loads(raw_body)
                    process_message(agent, sqs, body)

                    sqs.delete_message(
                        QueueUrl=SQS_QUEUE_URL,
                        ReceiptHandle=receipt_handle,
                    )
                    logger.info("Message processed successfully and deleted")

                except json.JSONDecodeError:
                    logger.exception("Invalid JSON in message body: %s", raw_body)
                    try:
                        sqs.delete_message(
                            QueueUrl=SQS_QUEUE_URL,
                            ReceiptHandle=receipt_handle,
                        )
                        logger.info("Deleted poison message from queue")
                    except Exception:
                        logger.exception("Failed to delete poison message")

                except Exception:
                    logger.exception("Failed to process message")
                    # do not delete; SQS will retry after visibility timeout

        except ClientError:
            logger.exception("SQS client error")
            time.sleep(5)

        except Exception:
            logger.exception("Unexpected worker error")
            time.sleep(5)


if __name__ == "__main__":
    poll_forever()