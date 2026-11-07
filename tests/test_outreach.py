from recruiter import outreach
from recruiter.schemas import CandidateProfile, Claim, GroundingReport, JobSpec, OutreachMessage

CV = "SENIOR ACCOUNTANT Summary Prepared monthly financial statements. Skills QuickBooks, Excel"
PROFILE = CandidateProfile(headline="SENIOR ACCOUNTANT", years_experience=8, seniority="senior",
                           skills=["quickbooks", "excel"],
                           achievements=["Prepared monthly financial statements"])
SPEC = JobSpec(title="Accountant", seniority="senior")
GOOD = "Hi, your work preparing monthly financial statements fits our Senior Accountant role. Open to a chat?"


def test_constraints():
    assert outreach.constraint_violations(GOOD) == []
    bad = "Dear [Name], you did great at Company Name. " + "word " * 130
    v = " ".join(outreach.constraint_violations(bad))
    assert "too long" in v and "Company Name" in v and "placeholder" in v and "call to action" in v


def test_checker_rejects_non_verbatim_evidence(monkeypatch):
    def fake(messages, schema, *, component, **kw):
        return GroundingReport(claims=[
            Claim(claim="prepares statements", supported=True,
                  evidence="Prepared monthly financial statements"),
            Claim(claim="led 10 people", supported=True, evidence="led a team of 10"),
        ])
    monkeypatch.setattr(outreach.llm, "structured", fake)
    r = outreach.check_grounding("msg", CV)
    assert [c.supported for c in r.claims] == [True, False]


def test_generate_rewrites_once_with_feedback(monkeypatch):
    calls = []

    def fake(messages, schema, *, component, **kw):
        calls.append(component)
        if schema is OutreachMessage:
            first = sum(c == "outreach_writer" for c in calls) == 1
            text = "Hi, you led 10 people. Interested?" if first else GOOD
            return OutreachMessage(text=text)
        supported = "led 10" not in messages[1]["content"]
        return GroundingReport(claims=[Claim(claim="x", supported=supported,
                                             evidence="Prepared monthly financial statements")])

    monkeypatch.setattr(outreach.llm, "structured", fake)
    res = outreach.generate_outreach(SPEC, PROFILE, CV)
    assert res.passed and res.attempts == 2
    assert calls == ["outreach_writer", "grounding_checker"] * 2


def test_facts_come_from_profile():
    facts = outreach.candidate_facts(PROFILE, None)
    assert "Achievement: Prepared monthly financial statements" in facts
