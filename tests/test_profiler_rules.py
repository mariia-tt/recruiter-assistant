from recruiter.profiler_rules import headline, parse_ranges, profile, seniority, years_of_experience
from recruiter.skills import extract_skills, has_skill


def test_parse_ranges_formats():
    text = "01/2010 to 01/2014 Analyst. Jan 2015 - Current Manager. 2005 to 2007 Clerk"
    assert len(parse_ranges(text)) == 3


def test_years_merges_overlaps():
    # 2010-2014 and 2012-2016 overlap -> 6 years, not 8
    assert years_of_experience("01/2010 to 01/2014 ... 01/2012 to 01/2016") == 6.0


def test_years_current_uses_reference_year():
    assert years_of_experience("Jan 2018 to Current") == 2.0


def test_headline_and_seniority():
    text = "SENIOR DATA ANALYST Professional Summary Experienced analyst"
    assert headline(text) == "SENIOR DATA ANALYST"
    assert seniority("SENIOR DATA ANALYST", 1) == "senior"
    assert seniority("DATA ANALYST", 1) == "junior"
    assert seniority("DATA ANALYST", 7) == "senior"
    assert seniority("OFFICE MANAGER", 3) == "senior"
    assert headline("SALES ASSOCIATE Experience 08/2014 to Current") == "SALES ASSOCIATE"


def test_skills_synonyms_and_boundaries():
    assert has_skill("Python", "built pipelines with pandas") == "pandas"
    assert has_skill("C++", "C++ and C#") == "c++"
    assert has_skill("go", "we go to market") is None
    assert "java" not in extract_skills("JavaScript developer")


def test_profile_end_to_end():
    p = profile("1", "IT DIRECTOR Summary Led a team. 03/2012 to Current. Skills: SQL, Agile, AWS")
    assert p.seniority == "lead" and p.years_experience == 7.8
    assert {"sql", "agile", "aws", "team leadership"} <= set(p.skills)


def test_unicode_dashes_and_education_excluded():
    assert years_of_experience("Analyst , 05/2009 \uff0d 05/2011 Company Name") == 2.0
    assert years_of_experience("Consultant (March 2008 \u00ad March 2010)") == 2.0
    text = "Experience 01/2015 to 01/2017 Analyst Education BSc 2010 - 2014 University Skills Excel"
    assert years_of_experience(text) == 2.0
