"""Skills dictionary with synonyms and text matching, used by C2a and C4.

Canonical name -> aliases (lower-case). Skills not listed are matched literally.
"""

import re
from functools import lru_cache

SKILL_ALIASES: dict[str, list[str]] = {
    # programming & data
    "python": ["python", "pandas", "numpy"],
    "java": ["java", "j2ee", "spring boot"],
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    ".net": [".net", "asp.net", "dotnet"],
    "php": ["php"],
    "ruby": ["ruby", "rails"],
    "go": ["golang"],
    "r": ["r programming", "rstudio"],
    "sql": ["sql", "t-sql", "pl/sql", "mysql", "postgresql", "postgres", "sql server", "oracle db"],
    "nosql": ["nosql", "mongodb", "cassandra", "dynamodb"],
    "html/css": ["html", "css", "html5", "css3"],
    "react": ["react", "react.js", "reactjs"],
    "angular": ["angular", "angularjs"],
    "node.js": ["node.js", "nodejs", "node js"],
    "machine learning": ["machine learning", "scikit-learn", "sklearn", "deep learning", "ml models"],
    "nlp": ["nlp", "natural language processing"],
    "data analysis": ["data analysis", "data analytics", "data analyst", "analyzing data"],
    "statistics": ["statistics", "statistical analysis", "regression analysis"],
    "spark": ["spark", "pyspark", "databricks"],
    "hadoop": ["hadoop", "hive", "hdfs", "mapreduce"],
    "etl": ["etl", "data warehouse", "data warehousing", "informatica", "ssis"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "spreadsheets", "vlookup", "pivot tables"],
    "sas": ["sas"],
    "matlab": ["matlab"],
    # infrastructure
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "azure": ["azure"],
    "gcp": ["gcp", "google cloud"],
    "docker": ["docker", "containers"],
    "kubernetes": ["kubernetes", "k8s"],
    "linux": ["linux", "unix", "red hat", "ubuntu"],
    "networking": ["networking", "tcp/ip", "lan", "wan", "cisco", "routers", "switches", "vpn"],
    "active directory": ["active directory"],
    "windows server": ["windows server"],
    "git": ["git", "github", "gitlab", "version control"],
    "ci/cd": ["ci/cd", "jenkins", "continuous integration"],
    "cybersecurity": ["cybersecurity", "information security", "network security", "firewall"],
    "technical support": ["technical support", "help desk", "helpdesk", "troubleshooting", "desktop support"],
    "sap": ["sap"],
    "salesforce": ["salesforce", "sfdc"],
    "sharepoint": ["sharepoint"],
    "jira": ["jira"],
    # methods & management
    "agile": ["agile", "scrum", "kanban", "sprint"],
    "project management": ["project management", "project manager", "pmp", "managed projects"],
    "product management": ["product management", "product manager", "product roadmap"],
    "stakeholder management": ["stakeholder management", "stakeholders"],
    "team leadership": ["team leadership", "team leader", "supervised", "led a team", "managed a team",
                        "managed team", "leadership"],
    "budgeting": ["budgeting", "budget management", "budgets", "forecasting"],
    "process improvement": ["process improvement", "lean", "six sigma", "continuous improvement"],
    "business analysis": ["business analysis", "business analyst", "requirements gathering"],
    "risk management": ["risk management", "risk assessment"],
    "vendor management": ["vendor management", "vendors", "procurement"],
    "training": ["training", "trained", "coaching", "mentoring"],
    "communication": ["communication skills", "communication", "presentation skills"],
    "negotiation": ["negotiation", "negotiating"],
    "customer service": ["customer service", "client service", "customer support", "customer satisfaction"],
    # finance & accounting
    "accounting": ["accounting", "general ledger", "accounts payable", "accounts receivable", "gaap",
                   "reconciliation", "reconciliations"],
    "financial analysis": ["financial analysis", "financial modeling", "financial reporting",
                           "financial statements"],
    "auditing": ["audit", "auditing", "internal audit", "sox"],
    "tax": ["tax", "taxation", "tax returns"],
    "payroll": ["payroll", "adp"],
    "quickbooks": ["quickbooks"],
    "banking": ["banking", "loans", "lending", "credit analysis", "underwriting"],
    "cpa": ["cpa", "certified public accountant"],
    # sales & marketing
    "sales": ["sales", "selling", "business development", "account management", "quota"],
    "lead generation": ["lead generation", "prospecting", "cold calling"],
    "crm": ["crm", "customer relationship management"],
    "marketing": ["marketing", "marketing campaigns", "brand management"],
    "digital marketing": ["digital marketing", "seo", "sem", "google analytics", "social media marketing",
                          "email marketing", "google ads"],
    "social media": ["social media", "facebook", "instagram", "twitter", "linkedin"],
    "content writing": ["content writing", "copywriting", "content creation", "blog"],
    "public relations": ["public relations", "media relations", "press releases"],
    "event planning": ["event planning", "event management", "events"],
    # HR
    "recruiting": ["recruiting", "recruitment", "talent acquisition", "sourcing", "interviewing"],
    "onboarding": ["onboarding", "new hire orientation"],
    "employee relations": ["employee relations", "conflict resolution"],
    "benefits administration": ["benefits administration", "benefits", "compensation"],
    "hris": ["hris", "workday", "peoplesoft", "kronos"],
    # design & media
    "photoshop": ["photoshop"],
    "illustrator": ["illustrator"],
    "indesign": ["indesign"],
    "adobe creative suite": ["adobe creative suite", "adobe creative cloud", "adobe suite"],
    "graphic design": ["graphic design", "graphic designer", "visual design"],
    "ui/ux": ["ui/ux", "ux", "user experience", "user interface", "wireframes", "prototyping"],
    "autocad": ["autocad", "cad"],
    "video editing": ["video editing", "final cut", "premiere"],
    "photography": ["photography", "photographer"],
    # engineering & construction & operations
    "mechanical engineering": ["mechanical engineering", "mechanical design", "solidworks"],
    "electrical engineering": ["electrical engineering", "electrical systems", "wiring", "schematics"],
    "construction management": ["construction management", "construction", "site management"],
    "quality assurance": ["quality assurance", "quality control", "qa", "qc", "inspection"],
    "safety compliance": ["osha", "safety compliance", "safety procedures", "workplace safety"],
    "supply chain": ["supply chain", "logistics", "inventory management", "inventory", "purchasing"],
    "manufacturing": ["manufacturing", "production"],
    "maintenance": ["maintenance", "repair", "repairs"],
    "aviation": ["aviation", "aircraft", "faa", "pilot", "flight"],
    "automotive": ["automotive", "vehicle", "vehicles", "mechanic"],
    "agriculture": ["agriculture", "agricultural", "farm", "crop", "crops", "agronomy"],
    # healthcare, education, legal, hospitality
    "patient care": ["patient care", "patients", "clinical", "nursing", "registered nurse"],
    "medical records": ["medical records", "ehr", "emr", "hipaa", "medical billing", "icd-10"],
    "teaching": ["teaching", "teacher", "classroom", "lesson plans", "curriculum", "instruction"],
    "legal research": ["legal research", "litigation", "legal writing", "contracts", "paralegal"],
    "food preparation": ["food preparation", "cooking", "menu", "culinary", "kitchen"],
    "fitness training": ["personal training", "personal trainer", "fitness", "exercise"],
    "retail": ["retail", "merchandising", "store"],
    # languages
    "spanish": ["spanish"],
    "french": ["french"],
    "german": ["german"],
}

ALIAS_TO_CANONICAL = {a: c for c, aliases in SKILL_ALIASES.items() for a in [c, *aliases]}
# Canonical names that are ordinary words/letters: match only via their aliases.
AMBIGUOUS = {"go", "r"}


@lru_cache(maxsize=4096)
def _pattern(alias: str) -> re.Pattern:
    """Word-boundary pattern that also works for names like c++, c#, .net."""
    return re.compile(rf"(?<![\w+#]){re.escape(alias)}(?![\w+#])", re.IGNORECASE)


def canonicalize(skill: str) -> str:
    """Map a free-text skill (e.g. from JobSpec) to its canonical name if known."""
    s = skill.strip().lower()
    return ALIAS_TO_CANONICAL.get(s, s)


def aliases_for(skill: str) -> list[str]:
    """All surface forms to look for; unknown skills match only themselves."""
    c = canonicalize(skill)
    aliases = SKILL_ALIASES.get(c, [])
    return aliases if c in AMBIGUOUS else [c, *aliases]


def has_skill(skill: str, text: str) -> str | None:
    """Return the alias found in text (evidence) or None."""
    for alias in aliases_for(skill):
        if _pattern(alias).search(text):
            return alias
    return None


def extract_skills(text: str) -> list[str]:
    """All dictionary skills present in a CV (canonical names, sorted)."""
    return sorted(c for c in SKILL_ALIASES if has_skill(c, text))
