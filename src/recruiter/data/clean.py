"""C0 Data cleaning: rules derived from docs/02_data_understanding.md."""

import hashlib
import re

import pandas as pd

from recruiter import config

SECTIONS = (
    "Summary|Highlights|Experience|Education|Skills|Accomplishments|Certifications|"
    "Qualifications|Interests|Languages|Projects|Training|Affiliations|Activities"
)
MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]\d{4}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
# "HighlightsMicrosoft" -> "Highlights Microsoft"
GLUED_SECTION_RE = re.compile(rf"\b({SECTIONS})(?=[A-Z][a-z])")
# "ExperienceData" handled above; "inMay" / "skillsEducation" -> space before month/section
GLUED_BEFORE_RE = re.compile(rf"([a-z])(?=(?:{MONTHS}|{SECTIONS})(?:[a-z]*)\b)")
SPACES_RE = re.compile(r"[ \t ]+")


def mask_pii(text: str) -> str:
    """Replace e-mails, phones and URLs so PII never reaches the index or the LLM."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)
    return PHONE_RE.sub("[PHONE]", text)


def fix_glued_words(text: str) -> str:
    """Split words glued by PDF extraction at section headers and month names."""
    text = GLUED_SECTION_RE.sub(r"\1 ", text)
    return GLUED_BEFORE_RE.sub(r"\1 ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and keep at most one blank line."""
    text = SPACES_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_text(text: str) -> str:
    """Full cleaning chain applied to every CV and JD."""
    return normalize_whitespace(fix_glued_words(mask_pii(text)))


def text_hash(text: str) -> str:
    """Hash of lower-cased alphanumerics, used for duplicate detection."""
    return hashlib.md5(re.sub(r"\W+", "", text.lower()).encode()).hexdigest()


def clean_resumes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the candidate DB; return cleaned frame and a report of dropped rows."""
    out = pd.DataFrame(
        {"id": df["ID"].astype(str), "category": df["Category"], "text": df["Resume_str"].map(clean_text)}
    )
    out["n_words"] = out.text.str.split().str.len()
    too_short = out.n_words < config.MIN_CV_WORDS
    dup = out.text.map(text_hash).duplicated()
    report = {"input_rows": len(df), "dropped_too_short": int(too_short.sum()),
              "dropped_duplicates": int((dup & ~too_short).sum())}
    out = out[~too_short & ~dup].reset_index(drop=True)
    report["output_rows"] = len(out)
    return out, report


def clean_fit_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Clean benchmark texts and add ids for resumes/JDs and a graded relevance label."""
    out = pd.DataFrame({
        "resume_text": df.resume_text.map(clean_text),
        "jd_text": df.job_description_text.map(clean_text),
        "label": df.label,
    })
    out["resume_id"] = "r" + out.resume_text.map(text_hash).str[:10]
    out["jd_id"] = "j" + out.jd_text.map(text_hash).str[:10]
    out["relevance"] = out.label.map({"No Fit": 0, "Potential Fit": 1, "Good Fit": 2})
    return out
