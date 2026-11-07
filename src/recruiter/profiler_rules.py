"""C2a Rule profiler: skills, years of experience and seniority without an LLM."""

import re

from recruiter import config
from recruiter.data.clean import SECTIONS
from recruiter.schemas import RuleProfile, Seniority
from recruiter.skills import extract_skills

MONTH_NUM = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_END = r"(?P<now>current|present|now|date)"


def _date(p: str) -> str:
    """Date like 05/2014, May 2014 or 2014 with named groups prefixed by p."""
    return rf"(?:(?P<{p}m>\d{{1,2}})/|(?P<{p}mn>{_MONTH})\s+)?(?P<{p}y>(?:19|20)\d{{2}})"


RANGE_RE = re.compile(
    _date("s") + r"\s*(?:to|-|–|—|until)\s*(?:" + _END + "|" + _date("e") + ")",
    re.IGNORECASE,
)

SENIOR_WORDS = {
    "lead": r"\b(lead|head|director|vp|vice president|chief|principal|partner|owner)\b",
    "senior": r"\b(senior|sr\.?|manager|supervisor|iii)\b",
    "junior": r"\b(intern|internship|junior|jr\.?|trainee|assistant|entry[- ]level)\b",
}


def _month(num: str | None, name: str | None) -> int:
    if num:
        return min(max(int(num), 1), 12)
    if name:
        return MONTH_NUM.get(name[:3].lower(), 1)
    return 1


def parse_ranges(text: str) -> list[tuple[float, float]]:
    """Employment-like date ranges as (start, end) in fractional years."""
    ranges = []
    for m in RANGE_RE.finditer(text):
        start = int(m["sy"]) + (_month(m["sm"], m["smn"]) - 1) / 12
        if m["now"]:
            end = float(config.DATA_REFERENCE_YEAR)
        else:
            end = int(m["ey"]) + (_month(m["em"], m["emn"]) - 1) / 12
        if 0 <= end - start <= 45:
            ranges.append((start, end))
    return ranges


def years_of_experience(text: str) -> float:
    """Total years covered by date ranges, overlaps merged (parallel jobs count once)."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(parse_ranges(text)):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return round(total, 1)


def headline(text: str) -> str:
    """CVs in the dataset start with the candidate's current title."""
    prefix = r"(?:(?:Professional|Executive|Career|Work|Relevant|Core)\s+)?"
    first = re.split(rf"\b{prefix}(?:{SECTIONS}|Profile|Objective|Overview)\b", text)[0]
    return " ".join(first.split()[:12]).strip(" -,:")


def seniority(title: str, years: float) -> Seniority:
    """Title keywords first, years-of-experience bands as fallback."""
    t = title.lower()
    for level, pattern in SENIOR_WORDS.items():
        if re.search(pattern, t):
            return level  # type: ignore[return-value]
    if years < 2:
        return "junior"
    if years < 5:
        return "mid"
    return "senior"


def profile(cv_id: str, text: str) -> RuleProfile:
    """Build the cheap profile used by the ranker for every CV."""
    years = years_of_experience(text)
    title = headline(text)
    return RuleProfile(
        id=cv_id,
        headline=title,
        skills=extract_skills(text),
        years_experience=years,
        seniority=seniority(title, years),
    )
