# PRD: Method C Event-Specific Probability Research
## For PRISM Brain Probability Engine
## February 2026

---

## 1. CONTEXT: What is PRISM Brain?

PRISM Brain is a **risk intelligence engine** that calculates the annual probability of 174 external risk events that could affect industrial companies. Think of it as answering the question: "What is the chance that [event X] happens at least once in the next 12 months?"

The engine currently computes probabilities for all 174 events using three methods:

- **Method A (33 events):** Historical frequency counting from databases (EM-DAT disasters, USGS earthquakes, FRED economic data). These are done -- they use real data.
- **Method B (26 events):** Industry survey decomposition from the Verizon DBIR and Dragos reports. These are done -- they use real survey data.
- **Method C (115 events):** Structural calibration using three sub-probabilities multiplied together. **These currently use generic family-level defaults. YOUR JOB is to replace those defaults with event-specific, evidence-based values.**

---

## 2. YOUR TASK: Replace Method C Defaults with Evidence-Based Values

### What Method C Does

For events that don't have a direct historical database, we estimate the annual probability by breaking it into three components:

```
P(event) = P(preconditions) x P(trigger) x P(implementation)
```

Where:
- **P(preconditions):** Are the structural conditions in place for this event to happen? (e.g., for "carbon tax implementation" -- is there legislative momentum, political will, and industry readiness?)
- **P(trigger):** Will a specific trigger event occur within 12 months? (e.g., an election, a climate summit, an international agreement, a crisis)
- **P(implementation):** If triggered, will it actually be implemented/realized within 12 months? (e.g., will the law pass, will the regulation take effect, will the market disruption materialize?)

### What's Wrong with the Current Defaults

Right now, all events in the same family share the same generic sub-probabilities. For example, ALL 7 STR-REG events (regulatory changes) use p_pre=0.70, p_trig=0.50, p_impl=0.60. But in reality:

- "Environmental regulation" (STR-REG-001) has VERY high preconditions (EU Green Deal pipeline is massive) -- maybe p_pre=0.85
- "Supply chain due diligence requirements" (STR-REG-006) is newer and less certain -- maybe p_pre=0.60

**Your job is to research each event individually and assign specific, justified sub-probabilities.**

---

## 3. EVIDENCE TYPES -- What Counts as Justification

Each sub-probability MUST be supported by one of these evidence types:

| Evidence Type | How to Use It | Example |
|---|---|---|
| **Legislative pipeline** | Count bills introduced vs. bills passed in observation window (2000-2024) | EU AI Act: 3 major bills introduced 2019-2024, 1 passed. P_implementation per bill ~ 0.33 |
| **Historical policy frequency** | Count analogous policy changes in observation window | REACH-style restrictions: 8 major amendments in 17 years. P ~ 0.47/year |
| **Election/political cycle** | Is there a relevant election in 2026-2027? Does election year historically increase likelihood? | US midterms 2026. WTO data: trade measures increase ~25% in US election years |
| **Expert survey / report** | Reference a specific report ranking (WEF Global Risks, Munich Re, etc.) | WEF 2025: risk ranked top 5 by >50% of respondents. P_preconditions = 0.80 |
| **Market/industry data** | Reference specific market statistics or industry reports | Gartner 2025: 78% of enterprises use multi-cloud. Cloud lock-in preconditions = 0.75 |
| **Historical incident count** | Count how many times something similar happened in 2000-2024 | 4 major rare earth supply crises in 25 years. P ~ 0.16/year |
| **Regulatory timeline** | When was the regulation announced? When does it take effect? | EU CBAM: announced 2021, transitional 2023, full implementation 2026. P_impl = 0.85 |

**If you genuinely cannot find evidence for a sub-probability after researching, set it to 0.50 and mark the evidence as `"DEFAULT_0.50_NO_DATA"`.** This makes the uncertainty visible rather than hiding it.

---

## 4. THE 115 EVENTS THAT NEED RESEARCH

Below is the complete list, organized by family. For each event, you need to provide p_pre, p_trig, p_impl, and evidence for each.

### PRIORITY 1: Structural & Strategic Events (40 events)
These have the highest impact on the probability quality because they're policy/regulatory/geopolitical events where real evidence exists.

#### STR-REG: Regulatory Changes (7 events)
| Event ID | Event Name |
|---|---|
| STR-REG-001 | Environmental Regulation (Emissions, Waste) |
| STR-REG-002 | Data Protection/Privacy Law Changes (GDPR-style) |
| STR-REG-003 | Product Safety Standards Changes |
| STR-REG-004 | Labor Law Changes (Minimum Wage, Hours) |
| STR-REG-005 | Financial Reporting/Tax Law Changes |
| STR-REG-006 | Supply Chain Due Diligence Requirements |
| STR-REG-007 | Chemical/Materials Restrictions (REACH-style) |

#### STR-ENP: Energy Policy Transition (6 events)
| Event ID | Event Name |
|---|---|
| STR-ENP-001 | Carbon pricing/tax implementation |
| STR-ENP-002 | Fossil fuel subsidy removal |
| STR-ENP-003 | Renewable energy mandates |
| STR-ENP-004 | Internal combustion engine phase-out |
| STR-ENP-005 | Building energy efficiency requirements |
| STR-ENP-006 | Green hydrogen/ammonia transition |

#### STR-TEC: Technology Policy (5 events)
| Event ID | Event Name |
|---|---|
| STR-TEC-001 | Semiconductor / Advanced Technology Export Control Expansion |
| STR-TEC-002 | AI Regulation / Algorithmic Compliance Requirements (EU AI Act) |
| STR-TEC-003 | Data Protection & Localization Mandates (GDPR, Schrems) |
| STR-TEC-004 | Forced Technology Transfer / Local Partnering Requirements |
| STR-TEC-005 | Critical Technology Investment Screening (FDI/CFIUS) |

#### STR-TRD: Trade Policy (6 events, including 1 Phase 1 event)
| Event ID | Event Name |
|---|---|
| STR-TRD-001 | Major Tariff Increases (>25%) -- NOTE: This already has Phase 1 values but could be improved |
| STR-TRD-002 | Export Controls on Critical Goods |
| STR-TRD-003 | Currency Controls or Capital Restrictions |
| STR-TRD-004 | Trade Agreement Termination |
| STR-TRD-005 | Import Quota or Licensing Requirements |
| STR-TRD-006 | Economic Sanctions on Business Partners |

#### STR-GEO: Geopolitical (6 events)
| Event ID | Event Name |
|---|---|
| STR-GEO-002 | Trade Embargo or International Sanctions |
| STR-GEO-003 | Nationalization or Asset Expropriation |
| STR-GEO-004 | Civil Unrest Disrupting Operations |
| STR-GEO-005 | State Collapse or Regime Change |
| STR-GEO-006 | Territorial Dispute (Sea Lanes, Borders) |
| STR-GEO-007 | Military Mobilization Affecting Workforce |

#### STR-ECO: Economic (2 events)
| Event ID | Event Name |
|---|---|
| STR-ECO-003 | Currency devaluation (>30% vs. USD/EUR) |
| STR-ECO-004 | Sovereign debt crisis |

#### STR-FIN: Financial System (5 events)
| Event ID | Event Name |
|---|---|
| STR-FIN-001 | SWIFT or payment system exclusion |
| STR-FIN-002 | Foreign exchange market closure |
| STR-FIN-003 | Credit market freeze (no access to financing) |
| STR-FIN-004 | Insurance market withdrawal (uninsurable risks) |
| STR-FIN-005 | Securities market suspension |

### PRIORITY 2: Digital Platform & Technology Events (21 events)

#### DIG-CLS: Cloud & Platform (6 events)
| Event ID | Event Name |
|---|---|
| DIG-CLS-001 | Hyperscaler vendor lock-in / Migration barriers |
| DIG-CLS-002 | Cloud service provider outage (>4hr, multi-region) |
| DIG-CLS-003 | Cloud pricing shock / Cost escalation (>30% YoY) |
| DIG-CLS-004 | Data residency / Sovereignty compliance failure |
| DIG-CLS-005 | Platform dependency / API deprecation |
| DIG-CLS-006 | National cloud / Data center capacity shortage |

#### DIG-HWS: Hardware Supply (7 events)
| Event ID | Event Name |
|---|---|
| DIG-HWS-001 | Advanced semiconductor access restriction (<7nm chips) |
| DIG-HWS-002 | Semiconductor supply chain disruption (fab fire, earthquake) |
| DIG-HWS-003 | Legacy chip / Mature node shortage (automotive MCUs) |
| DIG-HWS-004 | Hardware supply chain attack / Compromised components |
| DIG-HWS-005 | Critical equipment / Manufacturing tools shortage |
| DIG-HWS-006 | Industrial automation hardware dependency / Vendor lock-in |
| DIG-HWS-007 | Machine tool / Manufacturing equipment access restriction |

#### DIG-SWS: Software Supply (8 events)
| Event ID | Event Name |
|---|---|
| DIG-SWS-001 | Critical software license termination (ERP, CAD, OS) |
| DIG-SWS-002 | Open-source maintainer risk / Supply chain failure (Log4j) |
| DIG-SWS-003 | AI/ML platform dependency / Model access restriction |
| DIG-SWS-004 | AI model poisoning / Adversarial attack |
| DIG-SWS-005 | Critical software supply chain vulnerability (zero-day) |
| DIG-SWS-006 | Emerging technology access restriction (quantum, 5G) |
| DIG-SWS-007 | CAD/CAM/PLM software dependency / Vendor lock-in |
| DIG-SWS-008 | MES/SCADA/DCS software dependency (industrial control) |

### PRIORITY 3: Operational & Physical Events (54 events)

#### OPS-MAR: Maritime (5 events)
| Event ID | Event Name |
|---|---|
| OPS-MAR-001 | Major Port Congestion (>2 Week Delays) |
| OPS-MAR-003 | Port Labor Strike or Slowdown |
| OPS-MAR-004 | Shipping Line Bankruptcy or Service Suspension |
| OPS-MAR-005 | Container Shortage or Equipment Availability |
| OPS-MAR-007 | Port Cyberattack or System Failure |

#### OPS-AIR: Air Transport (5 events)
| Event ID | Event Name |
|---|---|
| OPS-AIR-001 | Air Freight Capacity Shortage |
| OPS-AIR-002 | Major Airport Closure (>24 Hours) |
| OPS-AIR-003 | Air Cargo Carrier Bankruptcy |
| OPS-AIR-004 | Aviation Fuel Shortage or Price Spike |
| OPS-AIR-005 | Airspace Closure or Flight Restrictions |

#### OPS-RLD: Road & Rail (6 events)
| Event ID | Event Name |
|---|---|
| OPS-RLD-001 | Trucking Capacity Shortage (Driver Shortage) |
| OPS-RLD-002 | Rail Freight Disruption (Derailment, Strike) |
| OPS-RLD-003 | Border Crossing Delays (>48 Hours) |
| OPS-RLD-004 | Highway/Infrastructure Closure (Accident, Weather) |
| OPS-RLD-005 | Fuel Shortage for Transport Fleet |
| OPS-RLD-006 | Truck/Rail Equipment Shortage |

#### OPS-CMP: Components (6 events)
| Event ID | Event Name |
|---|---|
| OPS-CMP-001 | Semiconductor Chip Shortage |
| OPS-CMP-002 | Battery/Energy Storage Materials Shortage |
| OPS-CMP-003 | Electronic Component Shortage (Capacitors, Resistors) |
| OPS-CMP-004 | Steel/Metals Shortage for Manufacturing |
| OPS-CMP-005 | Chemical/Polymer Feedstock Shortage |
| OPS-CMP-006 | Packaging Materials Shortage |

#### OPS-SUP: Supplier Risk (5 events)
| Event ID | Event Name |
|---|---|
| OPS-SUP-001 | Critical Supplier Bankruptcy |
| OPS-SUP-003 | Supplier Strike or Labor Disruption |
| OPS-SUP-004 | Supplier Quality Issue (Recall, Defects) |
| OPS-SUP-005 | Supplier Capacity Insufficient (Demand Surge) |
| OPS-SUP-006 | Supplier Exit from Market |

#### OPS-MFG: Manufacturing (4 events)
| Event ID | Event Name |
|---|---|
| OPS-MFG-001 | Critical Equipment Failure (>48hr Downtime) |
| OPS-MFG-003 | Production Line Contamination or Quality Failure |
| OPS-MFG-004 | Utility Failure (Power, Water, Gas) Disrupting Production |
| OPS-MFG-006 | Maintenance Shutdown Extended (Parts Unavailable) |

#### OPS-WHS: Warehouse (4 events)
| Event ID | Event Name |
|---|---|
| OPS-WHS-002 | Inventory Management System Failure |
| OPS-WHS-003 | Storage Capacity Shortage |
| OPS-WHS-004 | Warehouse Automation System Failure |
| OPS-WHS-005 | Theft or Inventory Shrinkage (>1M) |

#### PHY-ENE: Energy Supply (5 events)
| Event ID | Event Name |
|---|---|
| PHY-ENE-002 | Natural gas supply interruption |
| PHY-ENE-004 | Renewable energy output volatility |
| PHY-ENE-005 | Nuclear plant unplanned shutdown |
| PHY-ENE-006 | Grid frequency instability |
| PHY-ENE-007 | Transformer/substation failure |

#### PHY-MAT: Raw Materials (7 events)
| Event ID | Event Name |
|---|---|
| PHY-MAT-001 | Rare earth element supply disruption |
| PHY-MAT-002 | Lithium/battery materials shortage |
| PHY-MAT-004 | Semiconductor-grade silicon scarcity |
| PHY-MAT-005 | Agricultural commodity shortage (grains, oilseeds) |
| PHY-MAT-006 | Timber/wood products supply disruption |
| PHY-MAT-007 | Natural fiber shortage (cotton, wool, rubber) |
| PHY-MAT-008 | Plant-derived chemical feedstock shortage (palm oil, cellulose) |

#### PHY-WAT: Water (4 events)
| Event ID | Event Name |
|---|---|
| PHY-WAT-002 | Municipal/industrial water supply failure (>7 days) |
| PHY-WAT-003 | Wastewater treatment plant failure (>48hr) |
| PHY-WAT-004 | Industrial water contamination incident (>72hr remediation) |
| PHY-WAT-005 | Freshwater shortage for manufacturing (<80% normal supply, >=30 days) |

#### PHY-POL: Pollution (4 events)
| Event ID | Event Name |
|---|---|
| PHY-POL-003 | Soil contamination discovery (heavy metals, PFAS) |
| PHY-POL-005 | Radiation/nuclear contamination (>regulatory limits) |
| PHY-POL-006 | Electromagnetic/RF interference (>24hr, critical systems) |
| PHY-POL-007 | Noise pollution regulatory action (>10dB over limits) |

#### PHY-BIO: Biological (2 events)
| Event ID | Event Name |
|---|---|
| PHY-BIO-003 | Antimicrobial resistance (AMR) health crisis |
| PHY-BIO-005 | Occupational biological hazard (workplace exposure event) |

---

## 5. EXACT OUTPUT FORMAT

You MUST produce a single JSON file. The engine will load this file directly. Use this exact structure:

```json
{
  "metadata": {
    "created": "2026-02-20",
    "created_by": "Claude research session",
    "purpose": "Event-specific Method C sub-probabilities for PRISM Engine",
    "total_events": 115
  },
  "events": {
    "STR-REG-001": {
      "event_name": "Environmental Regulation (Emissions, Waste)",
      "p_pre": 0.85,
      "p_trig": 0.55,
      "p_impl": 0.65,
      "prior_computed": 0.3044,
      "evidence": {
        "p_preconditions": {
          "value": 0.85,
          "type": "legislative_pipeline",
          "justification": "EU Green Deal regulatory pipeline has 13 major proposals tabled 2019-2024 (ETS revision, CBAM, Nature Restoration Law, Industrial Emissions Directive revision, etc.). At least 8 reached implementation. Active pipeline remains full with ongoing amendments and new proposals.",
          "sources": ["EUR-Lex legislative tracker", "European Commission Green Deal tracker"]
        },
        "p_trigger": {
          "value": 0.55,
          "type": "historical_frequency",
          "justification": "In the period 2000-2024, 14 of 25 years saw at least one major new environmental regulation entering EU force (55%). Trigger events include COP summits, industrial accidents (Deepwater Horizon, Bhopal anniversary), and election cycle shifts.",
          "sources": ["EUR-Lex", "WTO Environmental Database"]
        },
        "p_implementation": {
          "value": 0.65,
          "type": "regulatory_timeline",
          "justification": "EU environmental directives have an average transposition period of 18-24 months. Of regulations triggered, approximately 65% take effect within the 12-month window (based on EUR-Lex transposition tracking 2015-2024).",
          "sources": ["European Commission transposition scoreboards"]
        }
      }
    },
    "STR-REG-002": {
      "event_name": "Data Protection/Privacy Law Changes (GDPR-style)",
      "p_pre": 0.75,
      "p_trig": 0.50,
      "p_impl": 0.55,
      "prior_computed": 0.2063,
      "evidence": {
        "p_preconditions": {
          "value": 0.75,
          "type": "legislative_pipeline",
          "justification": "EXPLANATION HERE",
          "sources": ["SOURCE1", "SOURCE2"]
        },
        "p_trigger": {
          "value": 0.50,
          "type": "TYPE_HERE",
          "justification": "EXPLANATION HERE",
          "sources": ["SOURCE1"]
        },
        "p_implementation": {
          "value": 0.55,
          "type": "TYPE_HERE",
          "justification": "EXPLANATION HERE",
          "sources": ["SOURCE1"]
        }
      }
    }
  }
}
```

### CRITICAL OUTPUT RULES:

1. **The `prior_computed` field MUST equal `p_pre x p_trig x p_impl`** (rounded to 4 decimal places). Double-check your math.

2. **Every sub-probability MUST have:**
   - `value`: float between 0.05 and 0.95
   - `type`: one of `"legislative_pipeline"`, `"historical_frequency"`, `"election_cycle"`, `"expert_survey"`, `"market_data"`, `"incident_count"`, `"regulatory_timeline"`, or `"DEFAULT_0.50_NO_DATA"`
   - `justification`: 1-3 sentences explaining WHY this number. Must reference specific data, not vague claims.
   - `sources`: list of 1-3 source names

3. **Plausibility checks** -- the resulting `prior_computed` should generally be:
   - Between 0.01 and 0.50 for most events
   - Below 0.10 for rare/extreme events (nuclear contamination, state collapse)
   - Above 0.20 for structurally likely events (new regulations, driver shortage)
   - If your math produces something above 0.50 or below 0.01, double-check your reasoning

4. **Do NOT over-estimate probabilities.** These are annual probabilities for events with SPECIFIC thresholds (">25% tariff increase", ">30% currency devaluation", ">48hr downtime"). The thresholds make them less likely than the generic category.

5. **Geographic scope matters.** Most events are scoped to affect European industrial companies specifically:
   - Regulatory events: EU/EEA + US + UK jurisdictions
   - Geopolitical events: Impact on TOP20 supplier countries (CN, US, GB, CH, RU, NO, JP, KR, TR, IN, BR, VN, TW, TH, SA, ID, MY, MX, UA, ZA)
   - Trade events: OECD trading partners
   - Digital/operational events: Global scope

6. **Output a single JSON file** with ALL 115 events. Do NOT split into multiple files.

---

## 6. RESEARCH APPROACH GUIDANCE

### For Regulatory Events (STR-REG, STR-ENP, STR-TEC):
- Check the EU legislative pipeline (EUR-Lex, European Commission work programme)
- Check US Congressional activity (Congress.gov)
- Check implementation timelines of recent similar regulations
- Count how many regulations of this type entered force in the last 10-15 years

### For Geopolitical Events (STR-GEO):
- Use UCDP data (ucdpapi.pcr.uu.se) for conflict history
- Check SIPRI for sanctions data
- Check PRS Group/Marsh Political Risk Map for country risk ratings
- Count historical incidents in 2000-2024

### For Trade Events (STR-TRD):
- WTO Trade Monitoring Reports (count restrictive measures per year)
- Global Trade Alert database
- Count how many years had >25% tariff changes, export controls, etc.

### For Financial Events (STR-FIN, STR-ECO):
- IMF Financial Stability Reports
- BIS Annual Reports
- Historical count of banking crises, currency crises (Reinhart-Rogoff database)

### For Digital Platform Events (DIG-CLS, DIG-HWS, DIG-SWS):
- Gartner/IDC market reports for market concentration data
- Uptime Institute for outage statistics
- Synergy Research for cloud market share data
- SIA Semiconductor Industry Association for chip supply data
- GitHub/Linux Foundation for open-source dependency data

### For Operational Events (OPS-*):
- Allianz Risk Barometer (annual)
- Munich Re NatCat Service for physical events
- Drewry/Clarksons for maritime data
- IRU/OECD for trucking/logistics data
- Industry-specific incident databases

### For Physical Events (PHY-*):
- EM-DAT (emdat.be) for disaster statistics
- EEA reports for European environmental data
- WHO/ECDC for biological hazard data
- IEA for energy data

---

## 7. IMPORTANT CONTEXT

### Time Horizon
All probabilities are **annual**: "What is the probability this event occurs at least once within a 12-month period?"

### Observation Window
Default observation window is **2000-2024 (25 years)**. If the risk didn't exist before a certain date (e.g., cloud sovereignty issues started ~2010), use a shorter window and document it.

### What "Event Occurs" Means
The event must meet the specific threshold in its name. For example:
- "Major Tariff Increases (>25%)" = a tariff increase exceeding 25% was imposed, not just any tariff change
- "Currency devaluation (>30%)" = a >30% drop, not just volatility
- "Critical Equipment Failure (>48hr Downtime)" = downtime exceeding 48 hours, not just any failure

### Layer 1 Only
You are computing the probability that this event happens ANYWHERE in the defined observation region (the world, Europe, or the relevant country set). You are NOT computing the probability that it affects a specific company. That's Layer 2, which is handled separately.

---

## 8. DELIVERABLE

A single JSON file following the schema in Section 5, containing all 115 Method C events with evidence-based sub-probabilities. Name it:

```
method_c_research_output.json
```

If you cannot complete all 115 events, prioritize in this order:
1. Priority 1: STR-* events (40 events) -- these have the most available evidence
2. Priority 2: DIG-CLS, DIG-HWS, DIG-SWS events (21 events) -- good market data available
3. Priority 3: OPS-* and PHY-* events (54 events) -- some may need DEFAULT_0.50_NO_DATA

---

## 9. QUALITY CHECKLIST

Before submitting, verify:

- [ ] All 115 events are present in the JSON
- [ ] Every `prior_computed` = `p_pre` x `p_trig` x `p_impl` (rounded to 4 decimal places)
- [ ] Every sub-probability has `value`, `type`, `justification`, and `sources`
- [ ] No sub-probability is exactly 0.50 unless marked as `"DEFAULT_0.50_NO_DATA"`
- [ ] No `prior_computed` is above 0.50 without strong justification
- [ ] No `prior_computed` is below 0.005 (very rare events should still be > 0.5%)
- [ ] The JSON is valid (no trailing commas, proper quoting)
- [ ] All event IDs match exactly (e.g., "STR-REG-001", not "STR-REG-1")
