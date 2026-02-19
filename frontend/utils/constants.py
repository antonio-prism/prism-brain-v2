"""
PRISM Brain - Configuration Constants
=====================================
Edit this file to change application settings without modifying code.
"""

# Application Settings
APP_NAME = "PRISM Brain"
APP_VERSION = "2.0.0"
APP_SUBTITLE = "Risk Intelligence System"

# Default Values
DEFAULT_CURRENCY = "EUR"
CURRENCY_SYMBOLS = {
    "EUR": "€",
    "USD": "$",
    "GBP": "£",
    "NOK": "kr"
}

# Risk Domains with colors
RISK_DOMAINS = {
    "PHYSICAL": {"color": "#FFC000", "icon": "🌍", "description": "Climate, Energy, Materials"},
    "STRUCTURAL": {"color": "#5B9BD5", "icon": "🏛️", "description": "Geopolitical, Regulatory, Financial"},
    "OPERATIONAL": {"color": "#70AD47", "icon": "⚙️", "description": "Supply Chain, Labor, Business Continuity"},
    "DIGITAL": {"color": "#7030A0", "icon": "💻", "description": "Cybersecurity, Technology, Data"}
}

# Probability Factor Weights (must sum to 1.0)
PROBABILITY_WEIGHTS = {
    "historical_frequency": 0.30,
    "trend_direction": 0.25,
    "current_conditions": 0.25,
    "geographic_exposure": 0.20
}

# Risk Level Thresholds
RISK_LEVELS = {
    "HIGH": {"min": 0.65, "color": "#FF6B6B", "label": "High Risk"},
    "MEDIUM": {"min": 0.40, "color": "#FFE066", "label": "Medium Risk"},
    "LOW": {"min": 0.0, "color": "#69DB7C", "label": "Low Risk"}
}

# Process Scopes (top-level grouping)
PROCESS_SCOPES = {
    "A": {"name": "Physical Assets & Infrastructure", "icon": "\U0001f3ed", "color": "#FFC000"},
    "B": {"name": "External Dependencies & Supply Chain", "icon": "\U0001f517", "color": "#5B9BD5"},
    "C": {"name": "Strategic & Commercial Operations", "icon": "\U0001f4c8", "color": "#70AD47"},
    "D": {"name": "Digital, Production & Workforce Operations", "icon": "\U0001f4bb", "color": "#7030A0"},
}

# Process Categories (top-level macro-processes, grouped by scope)
PROCESS_CATEGORIES = {
    "1": "Integrity of buildings and constructions",
    "2": "Maintenance of storage conditions for raw materials, products and waste",
    "3": "Maintenance of working and production conditions",
    "4": "Equipment Integrity and Operation",
    "5": "Services provided by the natural assets exploited by the company",
    "6": "Other physical assets operated by the business necessary for its operation",
    "7": "Supply Availability and Quality - Direct Scope of Tier 1 Suppliers",
    "8": "Product and service opportunities - Direct scope of Tier 1 customers",
    "9": "Freight supply and distribution - Availability and quality of transport networks",
    "10": "Mobility of people (employees and service providers)",
    "11": "Availability and quality of power supply to sites",
    "12": "Availability and quality of gas, steam, heat or cold supply",
    "13": "Availability and quality of water supply at sites",
    "14": "Availability and quality of telecom and internet networks",
    "15": "Disposal of waste and effluents",
    "16": "Financial Operations & Treasury Management",
    "17": "Insurance & Risk Transfer Programs",
    "18": "Stability of the political, regulatory and socio-economic environment",
    "19": "Market relevance of the offer",
    "20": "Tier 1 Supplier Value Chain",
    "21": "Tier 1 Customer Value Chain",
    "22": "Value chain of infrastructures and networks powering the company",
    "23": "Sales & Commercial Operations",
    "24": "Strategic Management & Governance",
    "25": "Marketing & Brand Management",
    "26": "IT Infrastructure & Enterprise Applications",
    "27": "Cybersecurity & Information Security Operations",
    "28": "Production Operations & Manufacturing Execution",
    "29": "Quality Management & Product Integrity",
    "30": "Human Resource Management & Workforce Operations",
    "31": "Product Development & Engineering",
    "32": "Maintenance & Asset Management Operations",
}

# Mapping: which scope does each top-level process belong to?
PROCESS_SCOPE_MAP = {
    "1": "A", "2": "A", "3": "A", "4": "A", "5": "A", "6": "A",
    "7": "B", "8": "B", "9": "B", "10": "B", "11": "B", "12": "B",
    "13": "B", "14": "B", "15": "B", "16": "B", "17": "B", "18": "B",
    "19": "C", "20": "C", "21": "C", "22": "C", "23": "C", "24": "C", "25": "C",
    "26": "D", "27": "D", "28": "D", "29": "D", "30": "D", "31": "D", "32": "D",
}

# Industry Templates (pre-configured process selections using new process IDs)
INDUSTRY_TEMPLATES = {
    "Manufacturing": {
        "description": "Manufacturing company (automotive, aerospace, chemicals, CPG)",
        "default_processes": ["1", "2", "4", "7", "9", "11", "26", "28", "29", "32"]
    },
    "Energy & Utilities": {
        "description": "Energy production, distribution, and utilities",
        "default_processes": ["1", "4", "5", "11", "12", "13", "15", "26", "27", "32"]
    },
    "Financial Services": {
        "description": "Banking, insurance, investment",
        "default_processes": ["16", "17", "18", "23", "24", "26", "27", "30"]
    },
    "Technology & Services": {
        "description": "Software, IT services, and consulting",
        "default_processes": ["8", "14", "19", "23", "26", "27", "30", "31"]
    },
    "Logistics & Distribution": {
        "description": "Transport, warehousing, and distribution",
        "default_processes": ["2", "9", "10", "15", "26", "28", "32"]
    },
    "Custom": {
        "description": "Select processes manually",
        "default_processes": []
    }
}

# Database Settings
DATABASE_NAME = "prism_brain.db"

# Export Settings
EXCEL_TEMPLATE_SHEETS = [
    "Dashboard",
    "Process Inventory",
    "Risk Events",
    "Risk Matrix",
    "Risk Heatmap",
    "Methodology"
]

# Pagination
ITEMS_PER_PAGE = 20

# Session timeout (minutes)
SESSION_TIMEOUT = 60
