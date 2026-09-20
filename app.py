

from __future__ import annotations
import dataclasses, json, re, io, time, textwrap, smtplib, datetime
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
COURSE_PYTHON = "Python (V-SCOPE)"
COURSE_OOP    = "OOP (Difficulty)"

OOP_GROUP_ORDER  = ["Lab 2 & Others", "Lab 3+4", "Lab 5+6", "Lab 7+8"]
OOP_GROUP_COLORS = {
    "Lab 2 & Others": "#534AB7",
    "Lab 3+4":        "#1D9E75",
    "Lab 5+6":        "#D85A30",
    "Lab 7+8":        "#378ADD",
}
OOP_GROUP_NUMS = {
    "Lab 2 & Others": {0, 2},
    "Lab 3+4":        {3, 4},
    "Lab 5+6":        {5, 6},
    "Lab 7+8":        {7, 8},
}

KNOWN_LABS = ["1D Array","2D Array","Functions","Strings","Pointers",
              "Structures","Classes","Sorting","Searching"]

BANDS = [
    {"label":"Excellent","min":90,  "max":100.01,"risk":1,"color":"#1D9E75","bg":"#E8F7F2","text_c":"#0F6E56"},
    {"label":"Good",     "min":80,  "max":90,    "risk":2,"color":"#378ADD","bg":"#E6F1FB","text_c":"#185FA5"},
    {"label":"Average",  "min":70,  "max":80,    "risk":3,"color":"#BA7517","bg":"#FAEEDA","text_c":"#854F0B"},
    {"label":"Below Avg","min":0,   "max":70,    "risk":4,"color":"#D85A30","bg":"#FAECE7","text_c":"#993C1D"},
]
BAND_MAP = {b["label"]: b for b in BANDS}
PALETTE  = {"bg":"#FAFAF8","text":"#1A1A1A","muted":"#888780"}

DEFAULT_MODEL  = "Qwen/Qwen2.5-Coder-7B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# ─────────────────────────────────────────────────────────────────────────────
# BAND UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def assign_band(pct: float, quick_rate: float = 0.0) -> str:
    for b in BANDS:
        if b["min"] <= pct < b["max"]:
            label = b["label"]
            if label == "Excellent" and quick_rate > 30:
                return "Good"
            return label
    return "Below Avg"

def band_color(pct: float) -> str:
    return BAND_MAP.get(assign_band(pct), BAND_MAP["Below Avg"])["color"]

# ─────────────────────────────────────────────────────────────────────────────
# COURSE AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_course(test_series: pd.Series) -> str:
    if test_series.astype(str).str.lower().str.contains("python").any():
        return COURSE_PYTHON
    return COURSE_OOP

# ─────────────────────────────────────────────────────────────────────────────
# GROUPING — PYTHON (lab subdivisions merged)
# ─────────────────────────────────────────────────────────────────────────────
def assign_lab_group_python(test_name: str) -> str:
    s = str(test_name).strip()
    pat = re.search(r"pat[\s_]+assessment[\s_]*(\d+)", s, re.IGNORECASE)
    if pat:
        return f"PAT Assessment {pat.group(1)}"
    pat = re.search(r"assessment[\s_]*(\d+)", s, re.IGNORECASE)
    if pat:
        return f"Assessment {pat.group(1)}"
    lab = re.search(r"lab[\s_]*(\d+)", s, re.IGNORECASE)
    if lab:
        return f"Lab {lab.group(1)}"
    cleaned = re.sub(r"^.*Python[\s_]*", "", s, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else s

# ─────────────────────────────────────────────────────────────────────────────
# GROUPING — OOP
# ─────────────────────────────────────────────────────────────────────────────
def assign_lab_group_oop(lab_no: int) -> str:
    for grp, nums in OOP_GROUP_NUMS.items():
        if lab_no in nums:
            return grp
    return "Lab 2 & Others"

def extract_lab_no(s: str) -> int:
    m = re.search(r"lab[\s_]*(\d+)", str(s), re.IGNORECASE)
    return int(m.group(1)) if m else 0

def extract_lab_name(s: str) -> str:
    for p in KNOWN_LABS:
        if p.lower() in str(s).lower():
            return p
    parts = str(s).split("_")
    for i, p in enumerate(parts):
        if p.strip() in ["Easy","Medium","Hard"] and i > 0:
            return parts[i-1].strip()
    return "Other"

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def extract_diff(s: str) -> str:
    for d in ["Easy","Medium","Hard"]:
        if d.lower() in str(s).lower(): return d
    return "Unknown"

def parse_dur(val) -> float:
    m = re.match(r"(\d+):(\d+):(\d+)", str(val).strip())
    if m:
        h,mn,s = int(m.group(1)),int(m.group(2)),int(m.group(3))
        return round(h*60+mn+s/60,1)
    try: return float(val)
    except: return np.nan


def compute_weighted_average_pct(marks, totals, round_digits: int = 1) -> float:
    marks_series = pd.to_numeric(marks, errors="coerce")
    totals_series = pd.to_numeric(totals, errors="coerce")
    valid = marks_series.notna() & totals_series.notna() & (totals_series > 0)
    if not valid.any():
        return 0.0
    total_marks = float(marks_series[valid].sum())
    total_possible = float(totals_series[valid].sum())
    if total_possible <= 0:
        return 0.0
    return round((total_marks / total_possible) * 100.0, round_digits)


def compute_mean_pct(pct_series, round_digits: int = 1) -> float:
    pct_values = pd.to_numeric(pct_series, errors="coerce").dropna().astype(float)
    if len(pct_values) == 0:
        return 0.0
    return round(float(np.nanmean(pct_values)), round_digits)


def find_col(cols, *kw, exclude=None):
    exclude = exclude or []
    for c in cols:
        lc = c.lower()
        if all(k in lc for k in kw) and not any(e in lc for e in exclude):
            return c

def natural_sort_key(x):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(x))]


def count_filled_outputs(mem: Optional[LearnerMemory]) -> int:
    if not mem:
        return 0
    return sum(1 for x in [mem.goals, mem.diagnosis, mem.plan, mem.feedback] if x)


def compute_participation_adjusted_score(pct_series: pd.Series, reference_mean: float, reference_n: int,
                                        participation_count: Optional[int] = None, prior_weight: float = 3.0) -> float:
    valid_scores = pct_series.dropna().astype(float)
    if len(valid_scores) == 0:
        return 0.0
    return round(float(np.nanmean(valid_scores)), 1)


def determine_band_by_completion_and_avg(avg_marks: float) -> str:
    """Determine final band label using average marks only."""
    if avg_marks >= 85.0:
        return "Excellent"
    if avg_marks >= 70.0:
        return "Good"
    if avg_marks >= 50.0:
        return "Average"
    return "Poor"

# ─────────────────────────────────────────────────────────────────────────────
# MEMORY DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class LearnerMemory:
    reg_no:       str
    name:         str
    class_id:     str
    course_type:  str
    episodic:     List[Dict] = dataclasses.field(default_factory=list)
    overall_pct:  float = 0.0
    band:         str   = "Below Avg"
    risk:         int   = 4
    quick_rate:   float = 0.0
    lab_scores:   Dict[str,float] = dataclasses.field(default_factory=dict)
    group_scores: Dict[str,float] = dataclasses.field(default_factory=dict)
    diff_scores:  Dict[str,float] = dataclasses.field(default_factory=dict)
    weak_labs:    List[str] = dataclasses.field(default_factory=list)
    strong_labs:  List[str] = dataclasses.field(default_factory=list)
    avg_duration: float = 0.0
    perfect_rate: float = 0.0
    completed_submissions: int = 0
    total_required_submissions: int = 0
    completion_rate: float = 0.0
    lab_completed_count: int = 0
    assessment_completed_count: int = 0
    pat_completed_count: int = 0
    other_completed_count: int = 0
    lab_required_count: int = 0
    assessment_required_count: int = 0
    pat_required_count: int = 0
    other_required_count: int = 0
    goals:        Optional[Dict] = None
    diagnosis:    Optional[Dict] = None
    plan:         Optional[Dict] = None
    feedback:     Optional[str]  = None

    def to_context_str(self) -> str:
        gs = ", ".join(f"{g}: {v}%" for g,v in self.group_scores.items())
        ls = ", ".join(f"{l}: {v}%" for l,v in self.lab_scores.items())
        ds = ", ".join(f"{d}: {v}%" for d,v in self.diff_scores.items())
        return (
            f"Course Track: {self.course_type}\n"
            f"Student: {self.name} ({self.reg_no}) | Class: {self.class_id}\n"
            f"Overall: {self.overall_pct}% | Band: {self.band} | Risk: {self.risk}/4\n"
            f"Group Scores: {gs or 'none'}\n"
            f"All Lab/Test Scores: {ls or 'none'}\n"
            f"Difficulty Breakdown: {ds or 'none'}\n"
            f"Weak Areas: {', '.join(self.weak_labs) or 'none'}\n"
            f"Strong Areas: {', '.join(self.strong_labs) or 'none'}\n"
            f"Avg Duration: {self.avg_duration} min | Perfect Rate: {self.perfect_rate}%\n"
            f"Quick-Submit Rate (<5 min): {self.quick_rate}%\n"
        )

@dataclasses.dataclass
class ClassMemory:
    total_students: int = 0
    course_types:   List[str] = dataclasses.field(default_factory=list)
    band_counts:    Dict[str,int]   = dataclasses.field(default_factory=dict)
    group_avgs:     Dict[str,float] = dataclasses.field(default_factory=dict)
    at_risk_regs:   List[str] = dataclasses.field(default_factory=list)
    quick_sub_regs: List[str] = dataclasses.field(default_factory=list)
    class_trend:    str = "stable"

@dataclasses.dataclass
class AgentMessage:
    sender:    str
    recipient: str
    msg_type:  str
    payload:   Dict[str,Any]
    timestamp: float = dataclasses.field(default_factory=time.time)

class AgentBus:
    def __init__(self):
        self._queue: List[AgentMessage] = []
        self._log:   List[AgentMessage] = []
    def post(self, msg: AgentMessage):
        self._queue.append(msg); self._log.append(msg)
    def pull(self, recipient:str, msg_type:str) -> List[AgentMessage]:
        msgs = [m for m in self._queue if m.recipient==recipient and m.msg_type==msg_type]
        self._queue = [m for m in self._queue if m not in msgs]
        return msgs
    def audit_log(self) -> List[Dict]:
        return [{"sender":m.sender,"recipient":m.recipient,"type":m.msg_type,
                 "ts":datetime.datetime.fromtimestamp(m.timestamp).strftime("%H:%M:%S")}
                for m in self._log]

# ─────────────────────────────────────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────────────────────────────────────
class BaseAgent:
    name = "BaseAgent"
    def __init__(self, client:InferenceClient, bus:AgentBus):
        self.client=client; self.bus=bus

    def _llm(self, system, user, max_tokens=400, temperature=0.7, retries=3):
        for attempt in range(retries):
            try:
                resp = self.client.chat_completion(
                    messages=[{"role":"system","content":system},{"role":"user","content":user}],
                    max_tokens=max_tokens, temperature=max(temperature,0.01))
                raw = resp.choices[0].message.content.strip()
                raw = re.sub(r"^```json\s*", "", raw)
                raw = re.sub(r"^```\s*", "", raw)
                raw = re.sub(r"```$", "", raw.strip())
                return raw
            except Exception:
                time.sleep(2**(attempt+1))
        return ""

    def _llm_json(self, system, user, max_tokens=600):
        raw = self._llm(system,user,max_tokens=max_tokens,temperature=0.2)
        try:
            m = re.search(r"\{[\s\S]*\}",raw)
            return json.loads(m.group(0)) if m else {}
        except: return {}


def _fallback_goal(memory: LearnerMemory) -> Dict[str, Any]:
    weak = memory.weak_labs[:2] or ["core lab concepts"]
    return {
        "primary_goal": f"Improve {weak[0]} performance",
        "sub_goals": [f"Practice {w} for 20 minutes daily" for w in weak[:3]],
        "success_criteria": "Consistent daily practice and improved lab scores",
        "stretch_goal": "Reach the next band within 1 week",
    }


def _fallback_diagnosis(memory: LearnerMemory) -> Dict[str, Any]:
    weak = memory.weak_labs[:2] or ["overall lab understanding"]
    return {
        "gap_severity_map": {
            weak[0]: {"severity": "moderate", "root_cause": "Needs more guided practice and revision"}
        },
        "misconceptions": ["Review fundamentals before attempting advanced tasks"],
        "practice_pattern": "Use short focused revision sessions and lab walkthroughs",
        "confidence_score": 0.6,
        "prerequisite_gaps": weak[:2],
    }


def _fallback_plan(memory: LearnerMemory) -> Dict[str, Any]:
    weak = memory.weak_labs[:2] or ["core concepts"]
    base_topic = weak[0]
    second_topic = weak[1] if len(weak) > 1 else weak[0]
    return {
        "week_1": [
            {"day_range": "Day 1", "topic": base_topic, "activity": "Review notes and solve 3 easy problems", "platform": "Portal", "time_mins": 45, "priority": "high"},
            {"day_range": "Day 2", "topic": second_topic, "activity": "Attempt 2 medium lab questions", "platform": "Portal", "time_mins": 45, "priority": "medium"},
            {"day_range": "Day 3", "topic": base_topic, "activity": "Rework mistakes and solve 2 mixed practice questions", "platform": "Portal", "time_mins": 45, "priority": "high"},
            {"day_range": "Day 4", "topic": second_topic, "activity": "Complete a short timed revision quiz", "platform": "Portal", "time_mins": 30, "priority": "medium"},
            {"day_range": "Day 5", "topic": base_topic, "activity": "Revise theory and attempt one harder problem", "platform": "Portal", "time_mins": 45, "priority": "high"},
            {"day_range": "Day 6", "topic": second_topic, "activity": "Practice 3 short questions and correct errors", "platform": "Portal", "time_mins": 45, "priority": "medium"},
            {"day_range": "Day 7", "topic": base_topic, "activity": "Take a mini mock and review weak spots", "platform": "Portal", "time_mins": 45, "priority": "high"},
        ],
        "week_2": [],
        "milestones": ["Complete 3 revision sessions", "Re-test on weak areas", "Maintain daily practice streak"],
        "daily_minimum_mins": 30,
    }


def _normalize_plan(plan: Optional[Dict[str, Any]], memory: LearnerMemory) -> Dict[str, Any]:
    fallback = _fallback_plan(memory)
    if not isinstance(plan, dict):
        plan = {}

    def _clean_step(step: Optional[Dict[str, Any]], fallback_step: Dict[str, Any]) -> Dict[str, Any]:
        step = step or {}
        return {
            "day_range": step.get("day_range") or fallback_step.get("day_range", ""),
            "topic": step.get("topic") or fallback_step.get("topic", ""),
            "activity": step.get("activity") or fallback_step.get("activity", ""),
            "platform": step.get("platform") or fallback_step.get("platform", "Portal"),
            "time_mins": step.get("time_mins") or fallback_step.get("time_mins", 30),
            "priority": step.get("priority") or fallback_step.get("priority", "medium"),
        }

    week_1 = [s for s in (plan.get("week_1") or []) if isinstance(s, dict)]
    week_2 = [s for s in (plan.get("week_2") or []) if isinstance(s, dict)]

    if not week_1:
        week_1 = list(fallback["week_1"])

    if len(week_1) < len(fallback["week_1"]):
        week_1 = week_1 + fallback["week_1"][len(week_1):len(fallback["week_1"])]

    return {
        "week_1": [_clean_step(step, fallback["week_1"][idx]) for idx, step in enumerate(week_1[:len(fallback["week_1"])] )],
        "week_2": [],
        "milestones": plan.get("milestones") or fallback["milestones"],
        "daily_minimum_mins": plan.get("daily_minimum_mins") or fallback["daily_minimum_mins"],
    }


def _fallback_feedback(memory: LearnerMemory) -> str:
    return (
        f"You scored {memory.overall_pct:.1f}% and are currently in the {memory.band} band. "
        f"Keep practicing the main weak areas and use short daily revision sessions to build consistency. "
        f"Aim to finish one focused lab task each day and review mistakes carefully."
    )

class GoalAgent(BaseAgent):
    name = "GoalAgent"
    SYSTEM = textwrap.dedent("""
    You are a Goal-Setting Agent. Given a student band and performance, define 3 concrete goals.
    STRICT OUTPUT valid JSON only no markdown:
    {"primary_goal":"...","sub_goals":["...","...","..."],"success_criteria":"...","stretch_goal":"..."}
    """).strip()
    def run(self, memory:LearnerMemory) -> None:
        user = f"{memory.to_context_str()}\nSet 3 goals for {memory.band} band. JSON only."
        memory.goals = self._llm_json(self.SYSTEM, user, max_tokens=500) or _fallback_goal(memory)
        self.bus.post(AgentMessage("GoalAgent","DiagnosisAgent","goals",memory.goals or {}))

class DiagnosisAgent(BaseAgent):
    name = "DiagnosisAgent"
    SYSTEM = textwrap.dedent("""
    You are a Diagnosis Agent. Identify root-cause knowledge gaps.
    STRICT OUTPUT valid JSON only no markdown:
    {"gap_severity_map":{"<topic>":{"severity":"critical|high|moderate","root_cause":"..."}},"misconceptions":[],"practice_pattern":"...","confidence_score":0.0,"prerequisite_gaps":[]}
    """).strip()
    def run(self, memory:LearnerMemory) -> None:
        user = f"{memory.to_context_str()}\nGoals:\n{json.dumps(memory.goals or {})}\nDiagnose. JSON only."
        memory.diagnosis = self._llm_json(self.SYSTEM, user, max_tokens=700) or _fallback_diagnosis(memory)
        self.bus.post(AgentMessage("DiagnosisAgent","PathPlannerAgent","diagnosis",memory.diagnosis or {}))

class PathPlannerAgent(BaseAgent):
    name = "PathPlannerAgent"
    SYSTEM = textwrap.dedent("""
    You are a Path-Planner Agent. Create a 1-week learning roadmap.
    STRICT OUTPUT valid JSON only no markdown.
    Required structure:
    {"week_1":[{"day_range":"Day 1","topic":"...","activity":"...","platform":"...","time_mins":45,"priority":"high|medium|critical"}, {"day_range":"Day 2",...}],"milestones":[],"daily_minimum_mins":30}
    Rules:
    - Cover all 7 days in week_1.
    - Include exactly 7 steps in week_1.
    - Use clear day ranges such as Day 1, Day 2, Day 3, Day 4, Day 5, Day 6, Day 7.
    """).strip()
    def run(self, memory:LearnerMemory) -> None:
        user = (f"{memory.to_context_str()}\nGoals:\n{json.dumps(memory.goals or {})}\n"
                f"Diagnosis:\n{json.dumps(memory.diagnosis or {})}\nBuild a 1-week roadmap. JSON only.")
        raw_plan = self._llm_json(self.SYSTEM, user, max_tokens=900)
        memory.plan = _normalize_plan(raw_plan or _fallback_plan(memory), memory)
        self.bus.post(AgentMessage("PathPlannerAgent","FeedbackAgent","plan",memory.plan or {}))

class FeedbackAgent(BaseAgent):
    name = "FeedbackAgent"
    SYSTEM = textwrap.dedent("""
    You are a Mentor Professor. Write ONE personalised 220-280 word plain-text feedback.
    Rules: no JSON, no bullets, no asterisks. First name only. First sentence states exact score and band.
    Weave in top diagnosis finding and one roadmap action. Warm, direct, encouraging.
    """).strip()
    BAND_HINT = {
        "Excellent":"Push beyond coursework: competitive coding, hackathons, open-source.",
        "Good":"Close the gap to Excellent by tackling weak areas head-on.",
        "Average":"Targeted remediation on 2 specific weak areas this week.",
        "Below Avg":"Step-by-step: theory to examples to Easy problems to portal practice.",
    }
    def run(self, memory:LearnerMemory) -> None:
        user = (f"{memory.to_context_str()}\nBand hint: {self.BAND_HINT.get(memory.band,'')}\n"
                f"Goals:\n{json.dumps(memory.goals or {})}\nDiagnosis:\n{json.dumps(memory.diagnosis or {})}\n"
                f"Plan:\n{json.dumps(memory.plan or {})}\nWrite mentor feedback. Plain text, 220-280 words.")
        memory.feedback = self._llm(self.SYSTEM, user, max_tokens=450, temperature=0.75) or _fallback_feedback(memory)
        self.bus.post(AgentMessage("FeedbackAgent","AnalyticsAgent","feedback",{"feedback":memory.feedback or ""}))

class AnalyticsAgent(BaseAgent):
    name = "AnalyticsAgent"
    SYSTEM = textwrap.dedent("""
    You are an Analytics Agent for the whole class (not individual students).
    STRICT OUTPUT valid JSON only no markdown:
    {"class_health":"Excellent|Good|Concerning|Critical","dominant_band":"...","at_risk_count":0,"key_finding":"...","weak_lab_groups":[],"instructor_actions":[],"intervention_priority":[{"reg_no":"...","reason":"..."}]}
    Special rule: if class_health is Excellent, make instructor_actions motivational and forward-looking, encouraging advanced learning and exploration of competitive coding platforms such as CodeChef, HackerRank, LeetCode, or similar.
    """).strip()
    def run_class(self, memories:List[LearnerMemory], class_mem:ClassMemory) -> Dict:
        user = (f"Class: {class_mem.total_students} students | Courses: {', '.join(set(class_mem.course_types))}\n"
                f"Bands: {json.dumps(class_mem.band_counts)}\nGroup Averages: {json.dumps(class_mem.group_avgs)}\n"
                f"At-risk: {class_mem.at_risk_regs[:10]}\nQuick-submit: {class_mem.quick_sub_regs[:10]}\n"
                f"Provide class analytics. JSON only.")
        return self._llm_json(self.SYSTEM, user, max_tokens=700)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process(file_bytes_list, file_names):
    all_dfs = []
    for i,(fb,fn) in enumerate(zip(file_bytes_list,file_names)):
        df = pd.read_excel(io.BytesIO(fb))
        df.columns = df.columns.str.strip()
        cols = list(df.columns)
        rename = {}
        for src,dst in [
            (find_col(cols,"reg"),                                                         "Reg_No"),
            (find_col(cols,"name"),                                                        "Name"),
            (find_col(cols,"test",exclude=["mark","total","%","status","start","submit"]), "Test"),
            (find_col(cols,"mark",exclude=["total"]),                                      "Marks"),
            (find_col(cols,"total","mark"),                                                "Total_Marks"),
            (find_col(cols,"duration"),                                                    "Duration"),
        ]:
            if src: rename[src] = dst
        df = df.rename(columns=rename)
        df["Duration_min"] = df["Duration"].apply(parse_dur)
        course = detect_course(df["Test"])
        df["Course_Type"] = course
        if course == COURSE_PYTHON:
            df["Lab_Group"] = df["Test"].apply(assign_lab_group_python)
            df["Lab"]       = df["Lab_Group"]
        else:
            df["Lab_No"]    = df["Test"].apply(extract_lab_no)
            df["Lab_Group"] = df["Lab_No"].apply(assign_lab_group_oop)
            df["Lab"]       = df["Test"].apply(extract_lab_name)
        df["Difficulty"]   = df["Test"].apply(extract_diff)
        df["Pct_num"]      = (df["Marks"] / df["Total_Marks"] * 100).round(1)
        df["Class_ID"]     = f"Class_{i+1} ({fn[:20]})"
        df["Flag_Quick"]   = df["Duration_min"] < 5.0
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

@st.cache_data(show_spinner=False)
def build_memories_cached(file_bytes_list, file_names):
    raw = load_and_process(file_bytes_list, file_names)
    groups_by_course: Dict[str,List[str]] = {}
    for course in raw["Course_Type"].unique():
        subset = raw[raw["Course_Type"]==course]
        if course == COURSE_PYTHON:
            unique_g = subset["Lab_Group"].unique().tolist()
            groups_by_course[course] = sorted(unique_g, key=natural_sort_key)
        else:
            groups_by_course[course] = OOP_GROUP_ORDER
    memories: List[LearnerMemory] = []
    # Compute per-course reference means and required submission maxima by category
    course_reference_mean: Dict[str, float] = {}
    course_requirements: Dict[str, Dict[str,int]] = {}
    for course in raw["Course_Type"].unique():
        subset = raw[raw["Course_Type"] == course]
        course_reference_mean[course] = float(np.nanmean(subset["Pct_num"])) if len(subset) else 60.0
        unique_course_tests = pd.Series(subset["Test"].dropna().astype(str).unique())
        lab_max = int(unique_course_tests.str.contains("Lab", case=False, na=False).sum())
        pat_max = int(unique_course_tests.str.contains("pat", case=False, na=False).sum())
        assess_max = int(unique_course_tests[(unique_course_tests.str.contains("assessment", case=False, na=False)) & ~unique_course_tests.str.contains("pat", case=False, na=False)].count())
        other_max = int((~unique_course_tests.str.contains("lab|assessment|pat", case=False, na=False)).sum())
        total_required = lab_max + assess_max + pat_max + other_max
        course_requirements[course] = {
            "lab_max": lab_max,
            "assess_max": assess_max,
            "pat_max": pat_max,
            "other_max": other_max,
            "total_required": max(1, total_required),
        }

    # Build learner memories using the new completion + average-based banding rules
    for (reg,name,cls),sdf in raw.groupby(["Reg_No","Name","Class_ID"]):
        course        = sdf["Course_Type"].iloc[0]
        active_groups = groups_by_course.get(course,[])
        reference_mean = course_reference_mean.get(course, 60.0)
        req = course_requirements.get(course, {"total_required":1})
        unique_student_tests = pd.Series(sdf["Test"].dropna().astype(str).unique())
        lab_count = int(unique_student_tests.str.contains("Lab", case=False, na=False).sum())
        pat_count = int(unique_student_tests.str.contains("pat", case=False, na=False).sum())
        assessment_count = int(unique_student_tests[(unique_student_tests.str.contains("assessment", case=False, na=False)) & ~unique_student_tests.str.contains("pat", case=False, na=False)].count())
        other_count = int((~unique_student_tests.str.contains("lab|assessment|pat", case=False, na=False)).sum())
        completed = lab_count + assessment_count + pat_count + other_count
        total_required = req.get("total_required", 1)
        completion_rate = round(float(completed) / float(total_required) * 100.0, 1) if total_required > 0 else 0.0

        quick_rt = round(sdf["Flag_Quick"].mean()*100, 1)
        lab_scores = {}
        for lab in sdf["Lab"].dropna().unique():
            subset = sdf[sdf["Lab"] == lab]
            lab_scores[lab] = compute_mean_pct(subset["Pct_num"])
        group_scores = {}
        for g in active_groups:
            subset = sdf[sdf["Lab_Group"] == g]
            if len(subset) > 0:
                group_scores[g] = compute_mean_pct(subset["Pct_num"])

        # Average marks are computed from the average of all lab/assessment/pat/other group scores
        avg_marks = compute_mean_pct(pd.Series(list(group_scores.values())))

        # Determine band using average marks only
        band_label = determine_band_by_completion_and_avg(avg_marks)
        # Map explicit 'Poor' label to existing 'Below Avg' band used across the app
        band = "Below Avg" if band_label == "Poor" else band_label
        bi = BAND_MAP.get(band, BAND_MAP["Below Avg"])

        if course == COURSE_PYTHON:
            lab_subset = sdf[sdf["Lab_Group"].str.contains("Lab", case=False, na=False)]
            assess_subset = sdf[sdf["Lab_Group"].str.contains("Assessment", case=False, na=False)]
            diff_scores: Dict[str,float] = {}
            if len(lab_subset) > 0:
                diff_scores["Lab Component"] = compute_mean_pct(lab_subset["Pct_num"])
            if len(assess_subset) > 0:
                diff_scores["Assessments"] = compute_mean_pct(assess_subset["Pct_num"])
        else:
            diff_scores = {}
            for d in ["Easy","Medium","Hard"]:
                subset = sdf[sdf["Difficulty"] == d]
                if len(subset) > 0:
                    diff_scores[d] = compute_mean_pct(subset["Pct_num"])
        episodic  = sdf[["Lab","Lab_Group","Difficulty","Pct_num","Duration_min","Flag_Quick"]].to_dict("records")
        email_val = ""
        if "Email" in sdf.columns:
            eq = sdf["Email"].dropna()
            if len(eq): email_val = str(eq.iloc[0]).strip()
        mem = LearnerMemory(
            reg_no=reg, name=name, class_id=cls, course_type=course,
            episodic=episodic, overall_pct=avg_marks, band=band, risk=bi["risk"],
            quick_rate=quick_rt, lab_scores=lab_scores, group_scores=group_scores, diff_scores=diff_scores,
            weak_labs=[l for l,s in group_scores.items() if s<80],
            strong_labs=[l for l,s in group_scores.items() if s>=90],
            avg_duration=round(sdf["Duration_min"].mean(),1),
            perfect_rate=round((sdf["Marks"]==sdf["Total_Marks"]).mean()*100,1),
            completed_submissions=completed,
            total_required_submissions=total_required,
            completion_rate=completion_rate,
            lab_completed_count=lab_count,
            assessment_completed_count=assessment_count,
            pat_completed_count=pat_count,
            other_completed_count=other_count,
            lab_required_count=req.get("lab_max", 0),
            assessment_required_count=req.get("assess_max", 0),
            pat_required_count=req.get("pat_max", 0),
            other_required_count=req.get("other_max", 0),
        )
        mem._email = email_val  # type: ignore
        memories.append(mem)
    return memories, raw, groups_by_course

def build_class_memory(memories:List[LearnerMemory], groups_by_course:Dict) -> ClassMemory:
    cm = ClassMemory()
    cm.total_students = len(memories)
    cm.course_types   = list({m.course_type for m in memories})
    for b in BANDS:
        cm.band_counts[b["label"]] = sum(1 for m in memories if m.band==b["label"])
    group_totals: Dict[str,List[float]] = {}
    for m in memories:
        for g,v in m.group_scores.items():
            group_totals.setdefault(g,[]).append(v)
    for g,vals in group_totals.items():
        cm.group_avgs[g] = round(np.mean(vals),1)
    cm.at_risk_regs   = [m.reg_no for m in memories if m.band=="Below Avg"]
    cm.quick_sub_regs = [m.reg_no for m in memories if m.quick_rate>30]
    return cm

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPER: render metric cards (shared by Python and OOP sections)
# ─────────────────────────────────────────────────────────────────────────────
def render_group_cards(groups, c_mems, border_color, section_label):
    if not groups: return
    st.markdown(f"##### {section_label}")
    num_cols = min(len(groups), 4)
    cols     = st.columns(num_cols)
    for idx, grp in enumerate(groups):
        vals = [m.group_scores.get(grp) for m in c_mems if m.group_scores.get(grp) is not None]
        if not vals: continue
        avg = round(np.mean(vals), 1)
        with cols[idx % num_cols]:
           st.markdown(
    f'<div style="background:white337ab7;border:2px solid #000000;border-radius:12px;'
    f'padding:14px;text-align:center;margin-bottom:12px;">'
    f'<div style="font-size:12px;color:#337ab7;font-weight:700;min-height:32px;overflow:hidden;">{grp}</div>'
    f'<div style="font-size:24px;font-weight:800;color:#337ab7;margin:4px 0;">{avg}%</div>'
    f'<div style="font-size:10px;color:#337ab7;font-weight:600;">{assign_band(avg)}</div>'
    f'<div style="font-size:9px;color:#337ab7;margin-top:2px;">{len(vals)} students</div></div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# PDF GENERATION  — Name + Reg No prominent
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"DejaVu Sans"})

def generate_pdf(memories:List[LearnerMemory]) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for mem in memories:
            bi  = BAND_MAP.get(mem.band, BAND_MAP["Below Avg"])
            fig = plt.figure(figsize=(14,10))
            fig.patch.set_facecolor(PALETTE["bg"])

            # HEADER — prominent Name (centre) + Reg No (left large)
            hax = fig.add_axes([0,0.91,1,0.09])
            hax.set_facecolor(bi["color"]); hax.axis("off")
            # Left: Reg No bold large
            hax.text(0.015, 0.80, mem.reg_no,
                     fontsize=15, fontweight="bold", color="black", va="center", transform=hax.transAxes)
            hax.text(0.015, 0.22, mem.course_type,
                     fontsize=8,  color="white", alpha=0.75, va="center", transform=hax.transAxes)
            # Centre: Full Name (largest)
            hax.text(0.50, 0.60, mem.name,
                     fontsize=20, fontweight="bold", color="black", va="center", ha="center", transform=hax.transAxes)
            hax.text(0.50, 0.18, mem.class_id,
                     fontsize=9,  color="white", alpha=0.75, va="center", ha="center", transform=hax.transAxes)
            # Right: Band + Score
            stars = "★"*mem.risk + "☆"*(4-mem.risk)
            hax.text(0.985, 0.80, f"{mem.overall_pct:.1f}%",
                     fontsize=16, fontweight="bold", color="white", va="center", ha="right", transform=hax.transAxes)
            hax.text(0.985, 0.25, f"{mem.band}  |  Risk {mem.risk}/4  {stars}",
                     fontsize=9,  color="white", alpha=0.85, va="center", ha="right", transform=hax.transAxes)

            # Metric tiles
            tiles = [("Overall",f"{mem.overall_pct:.1f}%"),("Perfect",f"{mem.perfect_rate:.0f}%"),
                     ("Avg Time",f"{mem.avg_duration:.0f}m"),("Quick%",f"{mem.quick_rate:.0f}%")]
            for ti,(lbl,val) in enumerate(tiles):
                ax = fig.add_axes([0.03+ti*0.24, 0.82, 0.21, 0.08])
                iq = lbl=="Quick%" and mem.quick_rate>30
                fc = "#FFF8E1" if iq else bi["bg"]
                ec = "#FFC107" if iq else bi["color"]
                tc = "#8B6914" if iq else bi["text_c"]
                ax.set_facecolor(fc); ax.axis("off")
                ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0.04",linewidth=0.8,edgecolor=ec,facecolor=fc))
                ax.text(0.5,0.65,val, ha="center",fontsize=13,fontweight="bold",color=tc,transform=ax.transAxes)
                ax.text(0.5,0.18,lbl, ha="center",fontsize=8,color=PALETTE["muted"],transform=ax.transAxes)

            # Group scores strip
            ax_grp = fig.add_axes([0.03,0.76,0.94,0.055])
            ax_grp.set_facecolor(PALETTE["bg"]); ax_grp.axis("off")
            ax_grp.set_xlim(0,1); ax_grp.set_ylim(0,1)
            ax_grp.text(0,0.92,"Group scores:",fontsize=7.5,color=PALETTE["muted"],transform=ax_grp.transAxes,va="top")
            groups = list(mem.group_scores.keys())
            n = max(len(groups),1)
            for ji,grp in enumerate(groups[:8]):
                v   = mem.group_scores.get(grp)
                xc  = 0.10 + ji*(0.88/n)
                fc2 = "#1D9E75" if (v and v>=90) else "#BA7517" if (v and v>=70) else "#D85A30"
                lbl2 = f"{grp}\n{v:.0f}%" if v is not None else f"{grp}\n--"
                ax_grp.text(xc,0.45,lbl2,ha="center",va="center",fontsize=8,fontweight="bold",
                            color=fc2 if v else "#888",transform=ax_grp.transAxes,linespacing=1.4)

            # Lab bar chart
            ax_lab = fig.add_axes([0.03,0.46,0.44,0.28])
            ax_lab.set_facecolor(PALETTE["bg"])
            ls = mem.lab_scores
            if ls:
                colors = ["#1D9E75" if s>=90 else "#BA7517" if s>=70 else "#D85A30" for s in ls.values()]
                ax_lab.barh(list(ls.keys())[:15], list(ls.values())[:15], color=colors[:15], height=0.6, zorder=3)
                ax_lab.set_xlim(0,112)
                ax_lab.set_title("Per-lab / Per-test Scores",fontsize=9,fontweight="bold",color=PALETTE["muted"],loc="left",pad=5)

            # Diff / component chart
            ax_diff = fig.add_axes([0.54,0.46,0.43,0.28])
            ax_diff.set_facecolor(PALETTE["bg"])
            ds = mem.diff_scores
            if ds:
                dcols = {"Easy":"#1D9E75","Medium":"#BA7517","Hard":"#D85A30",
                         "Lab Component":"#1D9E75","Assessments":"#378ADD"}
                ax_diff.bar(list(ds.keys()), list(ds.values()),
                            color=[dcols.get(d,"#888") for d in ds], width=0.45, zorder=3)
                ax_diff.set_ylim(0,112)
                title = "Difficulty Breakdown" if mem.course_type==COURSE_OOP else "Component Summary"
                ax_diff.set_title(title,fontsize=9,fontweight="bold",color=PALETTE["muted"],loc="left",pad=5)
                for i,(d,v) in enumerate(ds.items()):
                    ax_diff.text(i,v+1,f"{v:.0f}%",ha="center",fontsize=8.5,color=PALETTE["muted"])

            # Agent outputs box
            ax_ag = fig.add_axes([0.03,0.02,0.94,0.42])
            ax_ag.set_facecolor(bi["bg"]); ax_ag.axis("off")
            ax_ag.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0.02",linewidth=1.2,edgecolor=bi["color"],facecolor=bi["bg"]))
            goal_txt  = (mem.goals or {}).get("primary_goal","--")[:90]
            diag_map  = (mem.diagnosis or {}).get("gap_severity_map",{})
            diag_txt  = "--"
            if diag_map:
                top = next(iter(diag_map.items()),None)
                if top: diag_txt = f"{top[0]}: {top[1].get('severity','?')}"
            plan_wk1  = (mem.plan or {}).get("week_1",[{}])
            plan_txt  = (plan_wk1[0].get("activity","--") if plan_wk1 else "--")[:80]
            y = 0.95
            for label,txt,color in [
                ("Goal:",     goal_txt, bi["color"]),
                ("Diagnosis:",diag_txt, "#E24B4A"),
                ("Plan Wk1:", plan_txt, "#1D9E75"),
            ]:
                ax_ag.text(0.01,y,label,fontsize=7.5,fontweight="bold",color=color,va="top",transform=ax_ag.transAxes)
                ax_ag.text(0.14,y,txt,fontsize=7.5,color=PALETTE["text"],va="top",transform=ax_ag.transAxes,clip_on=True)
                y -= 0.09
            ax_ag.text(0.01,y-0.01,"Feedback:",fontsize=8,fontweight="bold",color=bi["text_c"],va="top",transform=ax_ag.transAxes)
            wrapped = textwrap.fill(mem.feedback or "Not generated.",width=115)
            ax_ag.text(0.01,y-0.09,wrapped,fontsize=8.2,color=PALETTE["text"],va="top",transform=ax_ag.transAxes,linespacing=1.55)
            fig.text(0.99,0.005,"Multi-Agent Educational Ecosystem",ha="right",fontsize=6.5,color=PALETTE["muted"])
            pdf.savefig(fig,bbox_inches="tight",facecolor=PALETTE["bg"])
            plt.close(fig)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
# EMAIL COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
EMAIL_CFG = {
    "Excellent":{"subject":"Outstanding Performance!","hbg":"#1D9E75","htxt":"Outstanding Performance",
                 "cta":"Push toward competitive coding challenges and hackathons this week."},
    "Good":     {"subject":"Good Progress - Path to Excellent","hbg":"#378ADD","htxt":"Good Performance",
                 "cta":"Re-do your weak lab exercises from scratch to reach the top band."},
    "Average":  {"subject":"Targeted Practice Will Make the Difference","hbg":"#BA7517","htxt":"Average Band - Action Needed",
                 "cta":"Pick your 2 weakest areas and solve 3 Easy problems each this week."},
    "Below Avg":{"subject":"Performance Review and Study Plan","hbg":"#D85A30","htxt":"Below Average - Focused Support Needed",
                 "cta":"Start from theory on your weakest topic today: read, examples, Easy, portal."},
}

def _plan_to_html(plan: Optional[Dict[str, Any]]) -> str:
    if not isinstance(plan, dict):
        return ""
    steps = (plan.get("week_1") or []) if isinstance(plan.get("week_1"), list) else []
    if not steps:
        return "<div>No plan generated yet.</div>"

    items = []
    for step in steps:
        day_range = step.get("day_range", "")
        topic = step.get("topic", "")
        activity = step.get("activity", "")
        platform = step.get("platform", "")
        time_mins = step.get("time_mins", 0)
        items.append(
            f"<div style='margin-bottom:6px;'><span style='color:#534AB7;font-size:10px;'>{day_range}</span> <strong>{topic}</strong><br>"
            f"<span style='color:#888;font-size:11px;'>{activity[:70]} - {platform} ({time_mins}min)</span></div>"
        )
    return f"<div style='margin-top:6px;'><strong>Week 1</strong></div>{''.join(items)}"


def compose_email(mem:LearnerMemory, signature_name: str = "Course Instructor") -> tuple:
    cfg   = EMAIL_CFG.get(mem.band, EMAIL_CFG["Below Avg"])
    hbg   = cfg["hbg"]
    first = mem.name.split()[0]
    sig_name = (signature_name or "").strip()
    signature_html = ""
    if sig_name:
        signature_html = f"<p style=\"color:#e0e0e0;font-size:13px;margin-top:20px;\">Best regards,<br><strong>{sig_name}</strong></p>"
    def _gc(v): return "#1D9E75" if v>=90 else "#BA7517" if v>=70 else "#D85A30"
    grp_rows = "".join(
        f"<tr><td style='padding:8px 12px;color:#ccc;border-bottom:1px solid #2a2a3e;'>{g}</td>"
        f"<td style='padding:8px 12px;text-align:center;font-weight:700;color:{_gc(v)};border-bottom:1px solid #2a2a3e;'>{v:.1f}%</td></tr>"
        for g,v in mem.group_scores.items()
    )
    plan_items = ""
    if mem.plan:
        for step in (mem.plan.get("week_1",[]))[:7]:
            plan_items += (f"<li style='color:#b0b0b0;font-size:13px;margin-bottom:5px;'>"
                           f"<strong>{step.get('day_range','')}</strong> · <strong>{step.get('topic','')}</strong> - {step.get('activity','')} ({step.get('platform','')})</li>")
    goal_txt = (mem.goals or {}).get("primary_goal","")
    diag_map = (mem.diagnosis or {}).get("gap_severity_map",{})
    diag_txt = ""
    if diag_map:
        top = next(iter(diag_map.items()),None)
        if top: diag_txt = f"Key gap: <strong style='color:#fff;'>{top[0]}</strong> ({top[1].get('severity','?')} severity)"
    fb_block = ""
    if mem.feedback:
        fb_block = (f"<div style='background:#0f1929;border:1px solid #1D9E7544;border-radius:10px;padding:16px;margin:16px 0;'>"
                    f"<p style='color:#1D9E75;font-size:13px;font-weight:600;margin:0 0 8px;'>AI Mentor Feedback</p>"
                    f"<p style='color:#b0b0b0;font-size:13px;line-height:1.75;margin:0;'>{mem.feedback}</p></div>")
    subject = cfg["subject"]
    html = (
        "<html><body style=\"margin:0;padding:0;background:#0d1117;font-family:'Segoe UI',Arial,sans-serif;\">"
        "<div style=\"max-width:660px;margin:20px auto;background:#161b22;border-radius:16px;border:1px solid #30363d;overflow:hidden;\">"
        f"<div style=\"background:{hbg};padding:28px 32px;text-align:center;\">"
        f"<h1 style=\"margin:0;color:#fff;font-size:20px;font-weight:800;\">{cfg['htxt']}</h1>"
        f"<p style=\"margin:6px 0 0;color:rgba(255,255,255,.85);font-size:13px;\">{mem.course_type} - Multi-Agent Performance Analytics</p>"
        "</div>"
        "<div style=\"padding:28px 32px;\">"
        f"<p style=\"color:#e0e0e0;font-size:15px;margin:0 0 4px;\">Dear <strong>{first}</strong>,</p>"
        "<div style=\"background:#1a1a2e;border-radius:10px;padding:14px 18px;margin:12px 0;\">"
        f"<span style=\"color:#888;font-size:12px;\">Reg No: <strong style=\"color:#fff;\">{mem.reg_no}</strong></span>&nbsp;&nbsp;"
        f"<span style=\"color:#888;font-size:12px;\">Overall: <strong style=\"color:{hbg};font-size:18px;\">{mem.overall_pct:.1f}%</strong></span>&nbsp;&nbsp;"
        f"<span style=\"color:#888;font-size:12px;\">Band: <strong style=\"color:{hbg};\">{mem.band}</strong></span>"
        "</div>"
        + (f"<div style=\"background:#1a1a2e;border-left:4px solid {hbg};padding:12px 16px;border-radius:0 8px 8px 0;color:#ccc;font-size:13px;margin:12px 0;\">Goal: <strong style=\"color:#fff;\">{goal_txt}</strong></div>" if goal_txt else "")
        + (f"<p style=\"color:#b0b0b0;font-size:13px;margin:8px 0;\">{diag_txt}</p>" if diag_txt else "")
        + "<h3 style=\"color:#42A5F5;font-size:14px;margin:20px 0 8px;border-bottom:1px solid #2a2a3e;padding-bottom:6px;\">Group Scores</h3>"
        "<table style=\"width:100%;border-collapse:collapse;background:#1e1e2f;border-radius:8px;overflow:hidden;\">"
        "<thead><tr style=\"background:#2a2a3e;\"><th style=\"padding:10px 14px;text-align:left;color:#888;font-size:11px;\">Group</th>"
        "<th style=\"padding:10px 14px;text-align:center;color:#888;font-size:11px;\">Score</th></tr></thead>"
        f"<tbody>{grp_rows}</tbody></table>"
        + (f"<h3 style=\"color:#1D9E75;font-size:14px;margin:20px 0 8px;\">Week 1 Study Plan</h3><ul style=\"padding-left:20px;margin:0;\">{plan_items}</ul>" if plan_items else "")
        + fb_block
        + f"<div style=\"background:#1a1a2e;border-left:4px solid {hbg};padding:14px 18px;border-radius:0 10px 10px 0;margin:16px 0;\">"
        f"<p style=\"color:#b0b0b0;font-size:13px;margin:0;line-height:1.7;\"><strong style=\"color:#fff;\">This week's action:</strong> {cfg['cta']}</p></div>"
        + signature_html
        + "</div>"
        "<div style=\"background:#0d1117;padding:10px 32px;text-align:center;border-top:1px solid #30363d;\">"
        "<p style=\"color:#555;font-size:10px;margin:0;\">Multi-Agent Educational Ecosystem - Auto-generated report</p>"
        "</div></div></body></html>"
    )
    return subject, html

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Agent Student Analyzer",page_icon="🤖",
                   layout="wide",initial_sidebar_state="expanded")
CSS = """<style>
.card{background:linear-gradient(135deg,#1e1e2f,#2a2a3e);border:1px solid #3a3a5c;
      border-radius:14px;padding:16px 20px;margin-bottom:12px}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:10px;font-weight:700;margin:2px}
.agent-step{border-radius:10px;padding:12px 16px;margin-bottom:8px;border-left:4px solid}
.agent-output{background:#0f172a;border:1px solid #334155;border-radius:10px;
              padding:14px 18px;margin:6px 0;font-size:13px;color:#cbd5e1;line-height:1.75}
.bus-msg{background:#1e293b;border-radius:6px;padding:6px 12px;margin:3px 0;
         font-size:11px;color:#94a3b8;font-family:monospace}
.course-header{border-radius:10px;padding:10px 16px;margin:8px 0 16px;font-size:13px;font-weight:700}

/* Make text boxes white with black text */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
}

div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stNumberInput"] input::placeholder {
    color: #6b7280 !important;
}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("## Multi-Agent Student Performance Analyzer")
st.caption("Auto-detects Python V-SCOPE or OOP Difficulty per file - GoalAgent - DiagnosisAgent - PathPlannerAgent - FeedbackAgent - AnalyticsAgent")

# SIDEBAR
with st.sidebar:
    st.markdown("### HuggingFace")
    hf_token     = st.text_input("HF Token", type="password", placeholder="hf_...")
    hf_model     = st.text_input("Model", value=DEFAULT_MODEL)
    st.markdown("---")
    st.markdown("### Gmail SMTP")
    sender_email = st.text_input("Sender Gmail", placeholder="you@gmail.com")
    sender_pass  = st.text_input("App Password", type="password", placeholder="xxxx xxxx xxxx xxxx")
    signature_name = st.text_input("Email Signature Name", value="Course Instructor",
                                   placeholder="Your Name or leave blank to remove")
    st.caption("Use a Gmail App Password, not your login password. This name appears in the email closing.")
    st.markdown("---")
    st.markdown("""
**Auto-detection:**
- "python" in Test -> Python V-SCOPE (Labs merged + Assessments)
- otherwise -> OOP Difficulty (Lab groups + Easy/Medium/Hard)
""")

# FILE UPLOAD
st.markdown("### Upload Excel Files")
uploaded = st.file_uploader("One or more .xlsx files (Python or OOP - auto-detected)",
                             type=["xlsx"], accept_multiple_files=True)
if not uploaded:
    st.info("Upload at least one Excel file to begin."); st.stop()

with st.spinner("Auto-detecting course type and building learner profiles..."):
    file_bytes = [f.read() for f in uploaded]
    file_names = [f.name for f in uploaded]
    memories, raw, groups_by_course = build_memories_cached(tuple(file_bytes), tuple(file_names))
    class_mem = build_class_memory(memories, groups_by_course)

detected_courses = list(groups_by_course.keys())
for course in detected_courses:
    icon  = "S" if course==COURSE_PYTHON else "C"
    color = "#1D9E75" if course==COURSE_PYTHON else "#378ADD"
    count = sum(1 for m in memories if m.course_type==course)
    st.sidebar.markdown(
        f'<div class="course-header" style="background:{color}22;border:1px solid {color}55;color:{color};">'
        f'{course} - {count} students</div>', unsafe_allow_html=True)

st.success(f"  {len(memories)} student profiles loaded - Courses: {' + '.join(detected_courses)}")

for k in ["agent_results","class_analytics","bus_log","email_previews","pdf_bytes","excel_bytes"]:
    if k not in st.session_state:
        st.session_state[k] = {} if k in ["agent_results","class_analytics","email_previews"] else [] if k=="bus_log" else None

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "Overview","Agent Pipeline","Student Analysis","Class Analytics","Email Notifications","PDF / Excel"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 - OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Performance Overview")
    for course in detected_courses:
        c_mems   = [m for m in memories if m.course_type==course]
        c_groups = groups_by_course.get(course,[])
        color    = "#337ab7" if course==COURSE_PYTHON else "#378ADD"
        st.markdown(f'<div class="course-header" style="background:{color}22;border:1px solid {color}55;color:{color};">{course} - {len(c_mems)} students</div>', unsafe_allow_html=True)
        valid_groups = [g for g in c_groups if any(m.group_scores.get(g) is not None for m in c_mems)]

        if course == COURSE_PYTHON:
            pat_ag = [g for g in valid_groups if "pat" in g.lower()]
            ag = [g for g in valid_groups if "assessment" in g.lower() and "pat" not in g.lower()]
            lg = [g for g in valid_groups if "lab" in g.lower()]
            other_g = [g for g in valid_groups if "assessment" not in g.lower() and "lab" not in g.lower() and "pat" not in g.lower()]
            # Render in order: Assessments, Labs, PAT Assessments, Other Groups
            if ag:
                render_group_cards(ag, c_mems, "#337ab7", "Assessments")
            render_group_cards(lg, c_mems, "#337ab7", "Labs (subdivisions merged)")
            if pat_ag:
                render_group_cards(pat_ag, c_mems, "#337ab7", "PAT Assessments")
            render_group_cards(other_g, c_mems, "#337ab7", "Other Groups")
            st.markdown("##### Progression Curve")
            fig_p = go.Figure()
            # Add traces in the same logical order
            if ag:
                a_avgs = [round(np.mean([m.group_scores[g] for m in c_mems if g in m.group_scores]),1) for g in ag]
                fig_p.add_trace(go.Scatter(x=ag,y=a_avgs,mode="lines+markers+text",name="Assessments",
                    text=[f"{v}%" for v in a_avgs],textposition="top center",line=dict(color="#378ADD",width=3),marker=dict(size=10)))
            if lg:
                l_avgs = [round(np.mean([m.group_scores[g] for m in c_mems if g in m.group_scores]),1) for g in lg]
                fig_p.add_trace(go.Scatter(x=lg,y=l_avgs,mode="lines+markers+text",name="Labs",
                    text=[f"{v}%" for v in l_avgs],textposition="bottom center",line=dict(color="#1D9E75",width=3),marker=dict(size=10,symbol="diamond")))
            if pat_ag:
                pat_avgs = [round(np.mean([m.group_scores[g] for m in c_mems if g in m.group_scores]),1) for g in pat_ag]
                fig_p.add_trace(go.Scatter(x=pat_ag,y=pat_avgs,mode="lines+markers+text",name="PAT Assessments",
                    text=[f"{v}%" for v in pat_avgs],textposition="top center",line=dict(color="#FFC107",width=3),marker=dict(size=10,symbol="square")))
            fig_p.update_layout(height=320,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                 font_color="white",yaxis=dict(range=[0,110],gridcolor="#333"),
                                 xaxis=dict(showgrid=False,tickangle=-45),
                                 legend=dict(orientation="h",y=1.1,x=1,xanchor="right"),margin=dict(t=30,b=20,l=10,r=10))
            st.plotly_chart(fig_p,use_container_width=True)
        else:
            # OOP: Lab groups + Difficulty breakdown (same separation as Python)
            render_group_cards(valid_groups, c_mems, "#534AB7", "Lab Groups")
            st.markdown("##### Difficulty Breakdown (Easy / Medium / Hard)")
            diff_colors = {"Easy":"#1D9E75","Medium":"#BA7517","Hard":"#D85A30"}
            d_cols = st.columns(3)
            for di,dlbl in enumerate(["Easy","Medium","Hard"]):
                vals = [m.diff_scores.get(dlbl) for m in c_mems if m.diff_scores.get(dlbl) is not None]
                if not vals: d_cols[di].info(f"{dlbl}: no data"); continue
                avg = round(np.mean(vals),1)
                dc  = diff_colors[dlbl]
                with d_cols[di]:
                    st.markdown(
                        f'<div style="background:#1a1a2e;border:2px solid {dc};border-radius:12px;padding:14px;text-align:center;margin-bottom:12px;">'
                        f'<div style="font-size:12px;color:#aaa;font-weight:700;">{dlbl}</div>'
                        f'<div style="font-size:24px;font-weight:800;color:{band_color(avg)};margin:4px 0;">{avg}%</div>'
                        f'<div style="font-size:10px;color:#666;">{assign_band(avg)}</div>'
                        f'<div style="font-size:9px;color:#444;">{len(vals)} students</div></div>', unsafe_allow_html=True)
            st.markdown("##### Lab Group Progression")
            gm = [{"Group":g,"Avg":round(np.mean([m.group_scores[g] for m in c_mems if g in m.group_scores]),1)}
                  for g in OOP_GROUP_ORDER if any(g in m.group_scores for m in c_mems)]
            if gm:
                gdf = pd.DataFrame(gm)
                fig_o = go.Figure()
                fig_o.add_trace(go.Scatter(x=gdf["Group"],y=gdf["Avg"],mode="lines+markers+text",
                    text=[f"{v:.1f}%" for v in gdf["Avg"]],textposition="top center",
                    line=dict(color="#534AB7",width=3),
                    marker=dict(size=14,color=[OOP_GROUP_COLORS[g] for g in gdf["Group"]],line=dict(width=2,color="white"))))
                fig_o.update_layout(height=280,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                    font_color="white",yaxis=dict(range=[0,105],gridcolor="#333"),
                                    xaxis=dict(showgrid=False),margin=dict(t=20,b=30))
                st.plotly_chart(fig_o,use_container_width=True)

        # Heatmap
        st.markdown(f"##### Heatmap - {course}")
        heat_rows = [{"Reg_No":m.reg_no,**{g:m.group_scores.get(g,0) for g in valid_groups}} for m in c_mems]
        if heat_rows and valid_groups:
            hdf = pd.DataFrame(heat_rows).set_index("Reg_No").fillna(0)
            fig_h = go.Figure(go.Heatmap(z=hdf.values,x=list(hdf.columns),y=hdf.index.tolist(),
                colorscale="RdYlGn",zmin=0,zmax=100,
                text=[[f"{v:.0f}%" if v>0 else "--" for v in row] for row in hdf.values],
                texttemplate="%{text}",textfont=dict(size=9)))
            fig_h.update_layout(height=max(300,len(hdf)*22),width=max(600,len(valid_groups)*90),
                                 paper_bgcolor="rgba(0,0,0,0)",font_color="white",margin=dict(t=10,b=40,l=130))
            st.plotly_chart(fig_h,use_container_width=len(valid_groups)<=8)

        # Band distribution
        st.markdown(f"##### Band Distribution - {course}")
        bc_vals = pd.Series([m.band for m in c_mems]).value_counts()
        fig_bd  = go.Figure()
        for b in BANDS:
            cnt = bc_vals.get(b["label"],0)
            fig_bd.add_trace(go.Bar(x=[b["label"]],y=[cnt],marker_color=b["color"],text=[cnt],textposition="outside",name=b["label"]))
        fig_bd.update_layout(showlegend=False,height=260,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                              font_color="white",yaxis=dict(gridcolor="#333"),margin=dict(t=10,b=10))
        st.plotly_chart(fig_bd,use_container_width=True)
        st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 - AGENT PIPELINE (Single student + At-risk bulk)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Agent Pipeline Execution")
    st.caption("GoalAgent -> DiagnosisAgent -> PathPlannerAgent -> FeedbackAgent run in sequence.")
    agent_defs = [("GoalAgent","Learning objectives","#1D9E75"),
                  ("DiagnosisAgent","Root-cause gap analysis","#378ADD"),
                  ("PathPlannerAgent","1-week study roadmap","#534AB7"),
                  ("FeedbackAgent","Synthesises all outputs","#BA7517"),
                  ("AnalyticsAgent","Class-level insights","#D85A30")]
    cols_ag = st.columns(5)
    for ci,(name,desc,col_) in enumerate(agent_defs):
        cols_ag[ci].markdown(
            f'<div style="background:#1a1a2e;border:2px solid {col_};border-radius:10px;padding:12px;text-align:center;">'
            f'<div style="font-size:11px;color:{col_};font-weight:700;margin-top:4px;">{name}</div>'
            f'<div style="font-size:9px;color:#666;margin-top:3px;">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown("---")

    c1,c2 = st.columns(2)
    run_mode    = c1.radio("Run mode",["Single student","All at-risk students","All students"],horizontal=True)
    filter_band = c2.selectbox("Filter by band",["All"]+[b["label"] for b in BANDS])

    # ─────────────────────────────────────────────
    # SINGLE STUDENT
    # ─────────────────────────────────────────────
    if run_mode == "Single student":
        filt = memories if filter_band=="All" else [m for m in memories if m.band==filter_band]
        opts = [f"{m.name} ({m.reg_no}) [{m.course_type}] - {m.band} {m.overall_pct:.1f}%" for m in filt]
        if not opts:
            st.info("No students in selected band.")
        else:
            selected_opt = st.selectbox("Select student",opts)
            sel_mem = filt[opts.index(selected_opt)]
            badge_color = "#1D9E75" if sel_mem.course_type==COURSE_PYTHON else "#378ADD"
            st.markdown(f'<span class="badge" style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}55;">{sel_mem.course_type}</span>', unsafe_allow_html=True)

            if st.button(f"Run Agent Pipeline for {sel_mem.name.split()[0]}",type="primary",key="run_single"):
                if not hf_token: st.error("Enter HuggingFace token in sidebar.")
                else:
                    client  = InferenceClient(model=hf_model,token=hf_token,timeout=180)
                    bus     = AgentBus()
                    agents  = [GoalAgent(client,bus),DiagnosisAgent(client,bus),PathPlannerAgent(client,bus),FeedbackAgent(client,bus)]
                    a_colors= ["#1D9E75","#378ADD","#534AB7","#BA7517"]
                    prog    = st.progress(0)
                    for ai,(agent,ac) in enumerate(zip(agents,a_colors)):
                        with st.spinner(f"[{ai+1}/4] {agent.name} running..."):
                            agent.run(sel_mem)
                        st.markdown(f'<div class="agent-step" style="border-color:{ac};background:#1a1a2e;"><span style="color:{ac};font-weight:700;">{agent.name}</span> <span style="color:#888;font-size:11px;">complete</span></div>', unsafe_allow_html=True)
                        prog.progress((ai+1)/4); time.sleep(0.4)
                    st.session_state["agent_results"][sel_mem.reg_no] = sel_mem
                    st.session_state["bus_log"] = bus.audit_log()
                    st.success(f"All 4 agents completed for {sel_mem.name}")

            result_mem = st.session_state["agent_results"].get(sel_mem.reg_no, sel_mem)

            with st.expander("LearnerMemory - current state",expanded=False):
                mc1,mc2,mc3 = st.columns(3)
                mc1.markdown(f'<div style="background:#0D2B1F;border:1px solid #1D9E75;border-radius:8px;padding:12px;"><div style="font-size:11px;font-weight:700;color:#1D9E75;">TIER 1 - Episodic</div><div style="font-size:10px;color:#ccc;margin-top:4px;">{len(result_mem.episodic)} lab attempt records</div></div>', unsafe_allow_html=True)
                mc2.markdown(f'<div style="background:#0A1929;border:1px solid #378ADD;border-radius:8px;padding:12px;"><div style="font-size:11px;font-weight:700;color:#378ADD;">TIER 2 - Semantic</div><div style="font-size:10px;color:#ccc;margin-top:4px;">Overall: {result_mem.overall_pct:.1f}% | Band: {result_mem.band}</div></div>', unsafe_allow_html=True)
                wk = sum(1 for x in [result_mem.goals,result_mem.diagnosis,result_mem.plan,result_mem.feedback] if x)
                mc3.markdown(f'<div style="background:#1A0D2E;border:1px solid #534AB7;border-radius:8px;padding:12px;"><div style="font-size:11px;font-weight:700;color:#534AB7;">TIER 3 - Working</div><div style="font-size:10px;color:#ccc;margin-top:4px;">{wk}/4 outputs filled</div></div>', unsafe_allow_html=True)

            if sel_mem.reg_no in st.session_state["agent_results"]:
                mem_r = st.session_state["agent_results"][sel_mem.reg_no]
                st.markdown("---")
                st.markdown("#### AgentBus Audit Log")
                for entry in st.session_state.get("bus_log",[]):
                    st.markdown(f'<div class="bus-msg">{entry["ts"]} | <strong>{entry["sender"]}</strong> -> {entry["recipient"]} [{entry["type"]}]</div>', unsafe_allow_html=True)
                st.markdown("#### Working Memory - Agent Outputs")
                wc1,wc2 = st.columns(2)
                with wc1:
                    if mem_r.goals:
                        st.markdown('<div style="font-size:12px;font-weight:700;color:#1D9E75;">GoalAgent</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="agent-output"><strong>{mem_r.goals.get("primary_goal","--")}</strong><br><br>'+"<br>".join(f"- {sg}" for sg in mem_r.goals.get("sub_goals",[]))+"</div>", unsafe_allow_html=True)
                    if mem_r.plan:
                        st.markdown('<div style="font-size:12px;font-weight:700;color:#534AB7;margin-top:8px;">PathPlannerAgent</div>', unsafe_allow_html=True)
                        p_html = _plan_to_html(mem_r.plan)
                        st.markdown(f'<div class="agent-output">{p_html}<br><strong>Daily min:</strong> {mem_r.plan.get("daily_minimum_mins","--")} min</div>', unsafe_allow_html=True)
                with wc2:
                    if mem_r.diagnosis:
                        st.markdown('<div style="font-size:12px;font-weight:700;color:#378ADD;">DiagnosisAgent</div>', unsafe_allow_html=True)
                        gaps = mem_r.diagnosis.get("gap_severity_map",{})
                        d_html = "".join(f"<div style='margin-bottom:5px;'><span style='color:#E24B4A;font-size:10px;font-weight:700;'>{v.get('severity','?').upper()}</span> <strong>{k}</strong><br><span style='color:#888;font-size:11px;'>{v.get('root_cause','')[:70]}</span></div>" for k,v in list(gaps.items())[:3])
                        misc = " | ".join(mem_r.diagnosis.get("misconceptions",[])[:2])
                        st.markdown(f'<div class="agent-output">{d_html}{"<br><em style=color:#888;font-size:11px;>Misconceptions: "+misc+"</em>" if misc else ""}</div>', unsafe_allow_html=True)
                    if mem_r.feedback:
                        st.markdown('<div style="font-size:12px;font-weight:700;color:#BA7517;margin-top:8px;">FeedbackAgent</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="agent-output" style="border:1px solid #BA751755;">{mem_r.feedback}</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # AT-RISK BULK MODE
    # ─────────────────────────────────────────────
    elif run_mode == "All at-risk students":
        at_risk = [m for m in memories if m.band in ["Average","Below Avg"]]
        if filter_band != "All":
            at_risk = [m for m in at_risk if m.band == filter_band]
        below   = [m for m in at_risk if m.band=="Below Avg"]
        avg_b   = [m for m in at_risk if m.band=="Average"]
        st.info(f"{len(at_risk)} at-risk students - Below Avg: {len(below)} - Average: {len(avg_b)}")
        if at_risk:
            st.dataframe(pd.DataFrame([{
                "Name":m.name,
                "Reg No":m.reg_no,
                "Band":m.band,
                "Score":f"{m.overall_pct:.1f}%",
                "Working Outputs":f"{count_filled_outputs(st.session_state['agent_results'].get(m.reg_no, m))}/4",
                "Agents":"done" if m.reg_no in st.session_state["agent_results"] else "pending"
            } for m in at_risk]),use_container_width=True,hide_index=True)
        if st.button("Run Pipeline for ALL At-Risk Students",type="primary",key="run_bulk"):
            if not hf_token: st.error("Enter HuggingFace token in sidebar.")
            else:
                client  = InferenceClient(model=hf_model,token=hf_token,timeout=180)
                bus     = AgentBus()
                agents  = [GoalAgent(client,bus),DiagnosisAgent(client,bus),PathPlannerAgent(client,bus),FeedbackAgent(client,bus)]
                prog    = st.progress(0); status = st.empty()
                for i,m_b in enumerate(at_risk):
                    status.markdown(f"**[{i+1}/{len(at_risk)}]** Running for **{m_b.name}** ({m_b.band} - {m_b.course_type})...")
                    for agent in agents: agent.run(m_b); time.sleep(0.3)
                    st.session_state["agent_results"][m_b.reg_no] = m_b
                    prog.progress((i+1)/max(len(at_risk),1))
                st.session_state["bus_log"] = bus.audit_log()
                st.success(f"Pipeline complete for {len(at_risk)} students.")

    # ─────────────────────────────────────────────
    # ALL STUDENTS MODE  ← NEW
    # ─────────────────────────────────────────────
    else:
        filt = memories if filter_band == "All" else [m for m in memories if m.band == filter_band]

        # Band breakdown summary
        band_counts = {}
        for m in filt:
            band_counts[m.band] = band_counts.get(m.band, 0) + 1
        band_summary = " | ".join(f"{b}: {c}" for b, c in band_counts.items())
        st.info(f"{len(filt)} students selected — {band_summary}")

        if filt:
            st.dataframe(
                pd.DataFrame([{
                    "Name":   m.name,
                    "Reg No": m.reg_no,
                    "Course": m.course_type,
                    "Band":   m.band,
                    "Score":  f"{m.overall_pct:.1f}%",
                    "Working Outputs": f"{count_filled_outputs(st.session_state['agent_results'].get(m.reg_no, m))}/4",
                    "Agents": "done" if m.reg_no in st.session_state["agent_results"] else "pending"
                } for m in filt]),
                use_container_width=True,
                hide_index=True
            )

        if st.button("Run Pipeline for ALL Students", type="primary", key="run_all"):
            if not hf_token:
                st.error("Enter HuggingFace token in sidebar.")
            else:
                client  = InferenceClient(model=hf_model, token=hf_token, timeout=180)
                bus     = AgentBus()
                agents  = [GoalAgent(client, bus), DiagnosisAgent(client, bus),
                           PathPlannerAgent(client, bus), FeedbackAgent(client, bus)]
                prog    = st.progress(0)
                status  = st.empty()

                for i, m_s in enumerate(filt):
                    status.markdown(
                        f"**[{i+1}/{len(filt)}]** Running for **{m_s.name}** "
                        f"({m_s.band} — {m_s.course_type})..."
                    )
                    for agent in agents:
                        agent.run(m_s)
                        time.sleep(0.3)
                    st.session_state["agent_results"][m_s.reg_no] = m_s
                    prog.progress((i + 1) / max(len(filt), 1))

                st.session_state["bus_log"] = bus.audit_log()
                st.success(f"Pipeline complete for all {len(filt)} students.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 - STUDENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Student Analysis with Agent Memory")
    c1, c2, c3, c4 = st.columns(4)
    fb_  = c1.selectbox("Band", ["All"] + [b["label"] for b in BANDS], key="sb_band")
    fc_  = c2.selectbox("Course Type", ["All"] + detected_courses, key="sb_course")
    fsq_ = c3.text_input("Search name / reg", key="sb_search")
    all_groups = sorted(list(set(g for m in memories for g in m.group_scores.keys())))
    fg_  = c4.selectbox("Group filter", ["All"] + all_groups, key="sb_grp")

    disp = memories.copy()
    if fb_ != "All":
        disp = [m for m in disp if m.band == fb_]
    if fc_ != "All":
        disp = [m for m in disp if m.course_type == fc_]
    if fg_ != "All":
        disp = [m for m in disp if fg_ in m.group_scores]
    if fsq_:
        fsq_l = fsq_.lower()
        disp = [m for m in disp if fsq_l in m.name.lower() or fsq_l in m.reg_no.lower()]
    disp.sort(key=lambda m: m.overall_pct)
    st.caption(f"Showing {len(disp)} students")

    for m in disp:
        bclr = BAND_MAP.get(m.band, BAND_MAP["Below Avg"])["color"]
        has_results = m.reg_no in st.session_state["agent_results"]
        badge_c = "#1D9E75" if m.course_type == COURSE_PYTHON else "#378ADD"
        agent_badge = (
            '<span class="badge" style="background:rgba(83,74,183,.2);color:#7F77DD;border:1px solid #534AB744;">agents run</span>'
            if has_results
            else '<span class="badge" style="background:rgba(136,135,128,.1);color:#888;border:1px solid #33333355;">not yet run</span>'
        )
        quick_badge = (
            f'<span class="badge" style="background:rgba(255,193,7,.15);color:#FFC107;border:1px solid rgba(255,193,7,.3);">{m.quick_rate:.0f}% quick-submit</span>'
            if m.quick_rate > 30 else ""
        )
        course_badge = f'<span class="badge" style="background:{badge_c}22;color:{badge_c};border:1px solid {badge_c}55;">{m.course_type}</span>'

        st.markdown(f"""
          <div class="card" style="border-color:{bclr}44;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <div>
                <span style="color:#fff;font-size:14px;font-weight:700;">{m.name}</span>
                <span style="color:#555;font-size:11px;margin-left:8px;">{m.reg_no}</span>
                <span style="color:#555;font-size:10px;margin-left:8px;">{m.class_id}</span>
                <div style="margin-top:5px;">{course_badge}{agent_badge}{quick_badge}</div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:22px;font-weight:800;color:{bclr};">{m.overall_pct:.1f}%</div>
                <div style="font-size:11px;color:#555;">{m.band} - Risk {m.risk}/4</div>
                <div style="font-size:10px;color:#FFFFFF;margin-top:2px;">Completion {m.completion_rate:.1f}%</div>
              </div>
            </div>
          </div>""", unsafe_allow_html=True)

        with st.expander(f"Analysis - {m.name}", expanded=False):
            if m.completion_rate < 50.0:
                st.markdown(
                    f'<div style="background:#FAECE7;border:1px solid #D85A30;border-radius:10px;padding:10px;margin-bottom:10px;">'
                    f'<div style="font-size:12px;font-weight:700;color:#D85A30;">Low completion alert</div>'
                    f'<div style="font-size:11px;color:#555;margin-top:4px;">Completed submissions: {m.completed_submissions}/{m.total_required_submissions}</div>'
                    f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;">'
                    f'<span style="background:#fff;border:1px solid #D85A30;border-radius:999px;padding:3px 8px;font-size:10px;color:#993C1D;">Labs: {m.lab_completed_count}/{m.lab_required_count}</span>'
                    f'<span style="background:#fff;border:1px solid #D85A30;border-radius:999px;padding:3px 8px;font-size:10px;color:#993C1D;">Assessments: {m.assessment_completed_count}/{m.assessment_required_count}</span>'
                    f'<span style="background:#fff;border:1px solid #D85A30;border-radius:999px;padding:3px 8px;font-size:10px;color:#993C1D;">PAT: {m.pat_completed_count}/{m.pat_required_count}</span>'
                    f'<span style="background:#fff;border:1px solid #D85A30;border-radius:999px;padding:3px 8px;font-size:10px;color:#993C1D;">Other: {m.other_completed_count}/{m.other_required_count}</span>'
                    f'</div></div>', unsafe_allow_html=True)

            student_groups = sorted(list(m.group_scores.keys()))
            if student_groups:
                n_cols = min(len(student_groups), 4)
                gcols = st.columns(n_cols)
                for idx, grp in enumerate(student_groups):
                    v = m.group_scores.get(grp)
                    bclr2 = band_color(v) if v is not None else "#555"
                    gclr = (
                        "#378ADD" if "assess" in grp.lower() else
                        "#1D9E75" if "lab" in grp.lower() else
                        OOP_GROUP_COLORS.get(grp, "#534AB7")
                    )
                    with gcols[idx % n_cols]:
                        st.markdown(
                            f'<div style="background:#1a1a2e;border:1px solid {gclr};border-radius:10px;padding:10px;text-align:center;margin-bottom:8px;">'
                            f'<div style="font-size:9px;color:{gclr};font-weight:700;min-height:22px;">{grp}</div>'
                            f'<div style="font-size:18px;font-weight:800;color:{bclr2};">{f"{v:.1f}%" if v is not None else "--"}</div>'
                            f'<div style="font-size:9px;color:#555;">{assign_band(v) if v is not None else "--"}</div></div>', unsafe_allow_html=True)

            for lab, score in sorted(m.lab_scores.items()):
                bc3 = "#1D9E75" if score >= 90 else "#BA7517" if score >= 70 else "#D85A30"
                bw3 = min(int(score), 100)
                st.markdown(
                    f'<div style="background:#1e1e2f;border-radius:8px;padding:7px 12px;margin-bottom:4px;">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                    f'<span style="color:#ccc;font-size:12px;">{lab}</span>'
                    f'<span style="color:{bc3};font-weight:700;font-size:13px;">{score:.1f}%</span></div>'
                    f'<div style="background:#333;border-radius:999px;height:6px;">'
                    f'<div style="width:{bw3}%;background:{bc3};height:6px;border-radius:999px;"></div></div></div>', unsafe_allow_html=True)

            st.markdown("---")
            has_agent_run = m.reg_no in st.session_state.get("agent_results", {})
            mem_r = st.session_state["agent_results"].get(m.reg_no, m)
            has_agent_outputs = any([
                isinstance(getattr(mem_r, "goals", None), dict) and bool(getattr(mem_r, "goals", None)),
                isinstance(getattr(mem_r, "diagnosis", None), dict) and bool(getattr(mem_r, "diagnosis", None)),
                isinstance(getattr(mem_r, "plan", None), dict) and bool(getattr(mem_r, "plan", None)),
                bool(getattr(mem_r, "feedback", None)),
            ])
            if has_agent_run or has_agent_outputs:
                st.markdown("**Agent Memory Outputs:**")
                wc1, wc2 = st.columns(2)
                with wc1:
                    if mem_r.goals:
                        st.markdown('<div style="font-size:12px;font-weight:700;color:#1D9E75;">GoalAgent</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="agent-output"><strong>{mem_r.goals.get("primary_goal", "--")}</strong><br><br>' + "<br>".join(f"- {sg}" for sg in mem_r.goals.get("sub_goals", [])) + "</div>", unsafe_allow_html=True)
                    if mem_r.plan:
                        st.markdown('<div style="font-size:12px;font-weight:700;color:#534AB7;margin-top:8px;">PathPlannerAgent</div>', unsafe_allow_html=True)
                        p_html = _plan_to_html(mem_r.plan)
                        st.markdown(f'<div class="agent-output">{p_html}<br><strong>Daily min:</strong> {mem_r.plan.get("daily_minimum_mins", "--")} min</div>', unsafe_allow_html=True)
                with wc2:
                    if mem_r.diagnosis:
                        st.markdown('<div style="font-size:12px;font-weight:700;color:#378ADD;">DiagnosisAgent</div>', unsafe_allow_html=True)
                        gaps = mem_r.diagnosis.get("gap_severity_map", {})
                        d_html = "".join(f"<div style='margin-bottom:5px;'><span style='color:#E24B4A;font-size:10px;font-weight:700;'>{v.get('severity', '?').upper()}</span> <strong>{k}</strong><br><span style='color:#888;font-size:11px;'>{v.get('root_cause', '')[:70]}</span></div>" for k, v in list(gaps.items())[:3])
                        misc = " | ".join(mem_r.diagnosis.get("misconceptions", [])[:2])
                        st.markdown(f'<div class="agent-output">{d_html}{"<br><em style=color:#888;font-size:11px;>Misconceptions: " + misc + "</em>" if misc else ""}</div>', unsafe_allow_html=True)
                    if mem_r.feedback:
                        st.markdown('<div style="font-size:12px;font-weight:700;color:#BA7517;margin-top:8px;">FeedbackAgent</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="agent-output" style="border:1px solid #BA751755;">{mem_r.feedback}</div>', unsafe_allow_html=True)
            else:
                st.info("Agent pipeline not yet run. Go to Agent Pipeline tab.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 - CLASS ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Class Analytics - AnalyticsAgent")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Students",len(memories))
    c2.metric("At-Risk",       len(class_mem.at_risk_regs))
    c3.metric("Quick-Submit",  len(class_mem.quick_sub_regs))
    c4.metric("Courses",       len(detected_courses))
    for course in detected_courses:
        cnt = sum(1 for m in memories if m.course_type==course)
        st.markdown(f"**{course}:** {cnt} students")
    st.markdown("---")
    if st.button("Run AnalyticsAgent (class-level)",type="primary"):
        if not hf_token: st.error("Enter HuggingFace token in sidebar.")
        else:
            with st.spinner("AnalyticsAgent analysing whole class..."):
                client  = InferenceClient(model=hf_model,token=hf_token,timeout=180)
                anal_ag = AnalyticsAgent(client,AgentBus())
                st.session_state["class_analytics"] = anal_ag.run_class(memories,class_mem)
            st.success("Class analytics complete.")
    ca = st.session_state.get("class_analytics",{})
    actions = []
    health = "--"
    if isinstance(ca, dict):
        raw_actions = ca.get("instructor_actions") or ca.get("actions") or ca.get("instructor_action", [])
        if isinstance(raw_actions, str):
            actions = [a.strip() for a in raw_actions.split("\n") if a.strip()]
        else:
            actions = list(raw_actions) if isinstance(raw_actions, list) else []

        health = str(ca.get("class_health", "--") or "--").strip().title()

        if health == "Excellent":
            actions = [
                "Motivate advanced learning: encourage students to explore competitive coding platforms such as CodeChef, HackerRank, and LeetCode for deeper challenge-based practice.",
                "Invite the class to tackle harder problems, hackathons, and open-source challenges to keep growing beyond the current syllabus."
            ]
    hc = {"Excellent":"#1D9E75","Good":"#378ADD","Concerning":"#BA7517","Critical":"#D85A30"}.get(health,"#888")
    if ca:
        st.markdown(f'<div style="background:#1a1a2e;border:2px solid {hc};border-radius:12px;padding:20px;margin:12px 0;"><div style="font-size:20px;font-weight:800;color:{hc};">Class Health: {health}</div><div style="color:#ccc;font-size:14px;margin-top:8px;">{ca.get("key_finding","--")}</div></div>', unsafe_allow_html=True)
        ic1,ic2 = st.columns(2)
        with ic1:
            st.markdown("**Instructor Actions:**")
            if actions:
                for act in actions:
                    st.markdown(f'<div style="background:#1e1e2f;border-left:3px solid #534AB7;padding:8px 12px;border-radius:0 8px 8px 0;margin-bottom:6px;color:#ccc;font-size:13px;">-> {act}</div>', unsafe_allow_html=True)
            else:
                st.info("No instructor actions were returned by AnalyticsAgent for this run.")
        with ic2:
            st.markdown("**Weak Lab Groups:**")
            for wg in ca.get("weak_lab_groups",[]):
                st.markdown(f'<div style="background:#FAECE7;border-left:3px solid #D85A30;padding:8px 12px;border-radius:0 8px 8px 0;margin-bottom:6px;color:#993C1D;font-size:13px;">{wg}</div>', unsafe_allow_html=True)
        ip = ca.get("intervention_priority",[])
        if ip:
            st.markdown("**Intervention Priority:**")
            st.dataframe(pd.DataFrame(ip),use_container_width=True,hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 - EMAIL NOTIFICATIONS (full preview + SMTP send)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Email Notifications")
    if sender_email and sender_pass:
        st.success(f"Gmail configured: {sender_email}")
    else:
        st.warning("Enter Gmail credentials in the sidebar to enable sending. You can still generate previews without credentials.")

    st.markdown("---")
    col_e1,col_e2 = st.columns(2)
    band_filter_e   = col_e1.multiselect("Send to bands",  [b["label"] for b in BANDS],default=["Average","Below Avg"])
    course_filter_e = col_e2.multiselect("Course types",   detected_courses,           default=detected_courses)
    include_quick_e = st.checkbox("Also include students with >30% quick-submit (any band)",value=True)

    targets_e = []
    for m in memories:
        if m.course_type not in course_filter_e: continue
        if m.band in band_filter_e: targets_e.append(m)
        elif include_quick_e and m.quick_rate>30 and m not in targets_e: targets_e.append(m)

    if not targets_e:
        st.info("No students match selected criteria.")
    else:
        st.caption(f"{len(targets_e)} students selected")
        st.dataframe(pd.DataFrame([{"Reg No":m.reg_no,"Name":m.name,"Course":m.course_type,
            "Band":m.band,"Score":f"{m.overall_pct:.1f}%","Quick%":f"{m.quick_rate:.0f}%",
            "Agents":"done" if m.reg_no in st.session_state["agent_results"] else "pending",
            "Email":getattr(m,"_email","--")} for m in targets_e]),use_container_width=True,hide_index=True)
        st.markdown("---")

        if st.button("Generate Email Previews (runs agents if needed)",type="secondary"):
            st.session_state["email_previews"] = {}
            if hf_token:
                client   = InferenceClient(model=hf_model,token=hf_token,timeout=180)
                bus_e    = AgentBus()
                agents_e = [GoalAgent(client,bus_e),DiagnosisAgent(client,bus_e),PathPlannerAgent(client,bus_e),FeedbackAgent(client,bus_e)]
                needs    = [m for m in targets_e if m.reg_no not in st.session_state["agent_results"]]
                if needs:
                    prog_e = st.progress(0)
                    for i,m_e in enumerate(needs):
                        with st.spinner(f"Agents for {m_e.name.split()[0]} ({m_e.course_type})..."):
                            for agent in agents_e: agent.run(m_e); time.sleep(0.3)
                            st.session_state["agent_results"][m_e.reg_no] = m_e
                        prog_e.progress((i+1)/max(len(needs),1))
            else:
                st.info("No HF token - emails generated without AI content.")
            for m in targets_e:
                mem_r = st.session_state["agent_results"].get(m.reg_no,m)
                subj,html = compose_email(mem_r, signature_name=signature_name)
                st.session_state["email_previews"][m.reg_no] = {
                    "name":m.name,"email":getattr(m,"_email",""),
                    "band":m.band,"course":m.course_type,"subject":subj,"html":html}
            st.success(f"Previews ready for {len(targets_e)} students!")

        if st.session_state.get("email_previews"):
            for reg,pd_ in st.session_state["email_previews"].items():
                with st.expander(f"{pd_['name']} ({pd_['band']} - {pd_['course']}) - {pd_['email'] or 'no email on file'}"):
                    st.markdown(f"**To:** `{pd_['email'] or 'MISSING'}`  |  **Subject:** {pd_['subject']}")
                    st.components.v1.html(pd_["html"],height=720,scrolling=True)

            st.markdown("---")
            n_v = sum(1 for v in st.session_state["email_previews"].values() if v["email"] and "@" in v["email"])
            n_m = len(st.session_state["email_previews"]) - n_v
            col_s1,col_s2 = st.columns([2,1])
            with col_s1:
                st.markdown(f'<div style="background:#1a1a2e;border:2px solid #FFC107;border-radius:12px;padding:16px;margin-bottom:12px;"><div style="color:#FFC107;font-size:14px;font-weight:700;margin-bottom:6px;">Confirm Before Sending</div><div style="color:#ccc;font-size:13px;">{n_v} emails with valid addresses - {n_m} skipped (no email on file). Each email includes AI-generated goals, diagnosis, study plan, and mentor feedback.</div></div>', unsafe_allow_html=True)
            with col_s2:
                st.metric("Ready to send",n_v)
                st.metric("Will be skipped",n_m)

            confirm_e = st.checkbox(f"I confirm sending {n_v} personalised email(s)")
            if st.button(f"Send {n_v} Email(s) via Gmail SMTP",type="primary",disabled=not confirm_e):
                if not sender_email or not sender_pass:
                    st.error("Gmail credentials missing. Enter them in the sidebar first.")
                else:
                    sent_e=failed_e=skipped_e=0
                    progress_send = st.progress(0)
                    total_send    = max(len(st.session_state["email_previews"]),1)
                    try:
                        smtp = smtplib.SMTP("smtp.gmail.com",587)
                        smtp.starttls(); smtp.login(sender_email,sender_pass)
                        for idx,(reg,pd_) in enumerate(st.session_state["email_previews"].items()):
                            recip = (pd_["email"].strip() if pd_["email"] else "")
                            if not recip or "@" not in recip:
                                skipped_e += 1
                            else:
                                try:
                                    msg = MIMEMultipart("alternative")
                                    msg["From"]=sender_email; msg["To"]=recip; msg["Subject"]=pd_["subject"]
                                    msg.attach(MIMEText(pd_["html"],"html"))
                                    smtp.send_message(msg); sent_e += 1
                                except Exception: failed_e += 1
                            progress_send.progress((idx+1)/total_send)
                        smtp.quit()
                        c1s,c2s,c3s = st.columns(3)
                        c1s.metric("Sent",sent_e); c2s.metric("Failed",failed_e); c3s.metric("Skipped",skipped_e)
                        if sent_e>0: st.success(f"Successfully sent {sent_e} email(s)!")
                        if failed_e>0: st.error(f"{failed_e} email(s) failed. Check SMTP credentials.")
                    except Exception as ex:
                        st.error(f"SMTP connection error: {ex}\nMake sure you are using a Gmail App Password.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 - PDF / EXCEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("PDF / Excel Export")
    c1p,c2p,c3p = st.columns(3)
    pdf_bands   = c1p.multiselect("Bands",   [b["label"] for b in BANDS],default=[b["label"] for b in BANDS])
    pdf_courses = c2p.multiselect("Courses", detected_courses,            default=detected_courses)
    pdf_classes = c3p.multiselect("Classes", sorted({m.class_id for m in memories}),
                                  default=sorted({m.class_id for m in memories}))

    pdf_mems = [m for m in memories if m.band in pdf_bands and m.course_type in pdf_courses and m.class_id in pdf_classes]
    pdf_mems.sort(key=lambda m:(m.course_type,m.class_id,m.band,-m.overall_pct))
    for i,m in enumerate(pdf_mems):
        if m.reg_no in st.session_state["agent_results"]:
            pdf_mems[i] = st.session_state["agent_results"][m.reg_no]

    agents_run = sum(1 for m in pdf_mems if m.reg_no in st.session_state["agent_results"])
    st.info(f"{len(pdf_mems)} students selected - {agents_run} have agent outputs - {len(pdf_mems)-agents_run} charts only.")

    if pdf_mems:
        st.dataframe(pd.DataFrame([{"Reg No":m.reg_no,"Name":m.name,"Course":m.course_type,
            "Class":m.class_id,"Band":m.band,"Score":f"{m.overall_pct:.1f}%",
            "Agents":"done" if m.reg_no in st.session_state["agent_results"] else "pending"
        } for m in pdf_mems]),use_container_width=True,hide_index=True)

        st.markdown("#### PDF Report (one page per student)")
        st.caption("Each page: prominent Name + Reg No in header, metric tiles, group scores, bar charts, agent outputs.")
        if st.button("Generate PDF",type="primary"):
            with st.spinner(f"Generating PDF for {len(pdf_mems)} students..."):
                st.session_state["pdf_bytes"] = generate_pdf(pdf_mems)
            st.success("PDF ready.")
        if st.session_state.get("pdf_bytes"):
            st.download_button("Download PDF Report",data=st.session_state["pdf_bytes"],
                               file_name="student_reports.pdf",mime="application/pdf",type="primary")

        st.markdown("---")
        st.markdown("#### Excel Workbook")
        if st.button("Generate Excel",type="secondary"):
            with st.spinner("Building workbook..."):
                rows_summary = []
                for m in pdf_mems:
                    mr    = st.session_state["agent_results"].get(m.reg_no,m)
                    g_str = (mr.goals or {}).get("primary_goal","")
                    d_str = ""
                    if mr.diagnosis:
                        top = next(iter((mr.diagnosis or {}).get("gap_severity_map",{}).items()),None)
                        if top: d_str = f"{top[0]} ({top[1].get('severity','')})"
                    p_str = ""
                    if mr.plan:
                        wk1 = mr.plan.get("week_1",[{}]); p_str = wk1[0].get("activity","") if wk1 else ""
                    row = {"Reg_No":mr.reg_no,"Name":mr.name,"Class":mr.class_id,"Course":mr.course_type,
                           "Overall_%":mr.overall_pct,"Band":mr.band,"Risk":mr.risk,"Quick_%":mr.quick_rate,
                           "Weak_Areas":", ".join(mr.weak_labs) or "None","Strong_Areas":", ".join(mr.strong_labs) or "None",
                           "Goal":g_str,"Top_Gap":d_str,"Week1_Action":p_str,"Feedback":mr.feedback or ""}
                    for g,v in mr.group_scores.items(): row[f"Group_{g}_%"] = v
                    rows_summary.append(row)
                buf_xl = io.BytesIO()
                with pd.ExcelWriter(buf_xl,engine="openpyxl") as w:
                    pd.DataFrame(rows_summary).to_excel(w,sheet_name="All_Students",index=False)
                    for course in detected_courses:
                        course_rows = [r for r in rows_summary if r.get("Course")==course]
                        if course_rows:
                            sn = "Python_VSCOPE" if course==COURSE_PYTHON else "OOP_Difficulty"
                            pd.DataFrame(course_rows).sort_values("Overall_%",ascending=False).to_excel(w,sheet_name=sn,index=False)
                    if st.session_state.get("class_analytics"):
                        pd.DataFrame([{"Key":k,"Value":str(v)} for k,v in st.session_state["class_analytics"].items()]).to_excel(w,sheet_name="Class_Analytics",index=False)
                buf_xl.seek(0)
                st.session_state["excel_bytes"] = buf_xl.read()
            st.success("Excel ready.")
        if st.session_state.get("excel_bytes"):
            st.download_button("Download Excel Workbook",data=st.session_state["excel_bytes"],
                               file_name="student_analysis.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("No students match the selected filters.")

