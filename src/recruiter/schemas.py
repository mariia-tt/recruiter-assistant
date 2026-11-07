"""Pydantic contracts exchanged between pipeline components (AGENTS.md rule 5)."""

from typing import Literal

from pydantic import BaseModel, Field

Seniority = Literal["junior", "mid", "senior", "lead"]


class JobSpec(BaseModel):
    """Structured vacancy produced by the Requirements Collector (C1)."""

    title: str | None = None
    seniority: Seniority | None = None
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    min_years: float | None = None
    domain: str | None = None
    location: str | None = None

    def missing_mandatory(self) -> list[str]:
        """Slots the collector must ask about before search can start."""
        missing = []
        if not self.title:
            missing.append("title")
        if not self.must_have_skills:
            missing.append("must_have_skills")
        if self.min_years is None:
            missing.append("min_years")
        if self.seniority is None:
            missing.append("seniority")
        return missing


class RuleProfile(BaseModel):
    """Cheap facts extracted from every CV by the rule profiler (C2a)."""

    id: str
    headline: str
    skills: list[str]
    years_experience: float
    seniority: Seniority


class SearchHit(BaseModel):
    """One retrieved CV with the scores of each retriever (C3)."""

    id: str
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class CriterionScore(BaseModel):
    """One scorecard line: normalised score in [0, 1], its weight and the evidence."""

    name: str
    score: float
    weight: float
    evidence: list[str] = Field(default_factory=list)


class CandidateScore(BaseModel):
    """Ranked candidate produced by the Scorecard Ranker (C4)."""

    id: str
    total: float
    knocked_out: bool
    criteria: list[CriterionScore]
    matched_must: list[str] = Field(default_factory=list)
    missing_must: list[str] = Field(default_factory=list)


class LLMCandidateRating(BaseModel):
    """LLM re-ranker (C4b) verdict for one candidate, interview-scorecard style."""

    candidate: int = Field(description="candidate number as given in the prompt")
    skills_fit: int = Field(ge=1, le=5)
    experience_fit: int = Field(ge=1, le=5)
    role_fit: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    evidence_quote: str = Field(description="short verbatim quote from the CV supporting the rating")
    concern: str | None = Field(default=None, description="main gap or risk, if any")


class LLMRerankResult(BaseModel):
    ratings: list[LLMCandidateRating]
