#!/usr/bin/env python3
"""
Parse risk event data from taxonomy and detailed MD files.
Outputs comprehensive JSON with all events.
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Define the source files
TAXONOMY_FILE = "/sessions/modest-vigilant-lamport/mnt/prism-brain-v2/Risk_Family_Taxonomy_REVISED_v2.1.md"
DETAIL_FILES = [
    "/sessions/modest-vigilant-lamport/mnt/prism-brain-v2/Family_1.1_Climate_Extremes_COMPLETE.md",
    "/sessions/modest-vigilant-lamport/mnt/prism-brain-v2/Family_1.2_Energy_Supply_COMPLETE.md",
    "/sessions/modest-vigilant-lamport/mnt/prism-brain-v2/Family_1.3_Natural_Resources_COMPLETE.md",
    "/sessions/modest-vigilant-lamport/mnt/prism-brain-v2/Family_1.4_Water_Resources_COMPLETE.md",
    "/sessions/modest-vigilant-lamport/mnt/prism-brain-v2/Family_1.5_Geophysical_COMPLETE.md",
    "/sessions/modest-vigilant-lamport/mnt/prism-brain-v2/Family_1.6_Contamination_Pollution_COMPLETE.md",
    "/sessions/modest-vigilant-lamport/mnt/prism-brain-v2/Family_1.7_Biological_Pandemic_COMPLETE.md",
    "/sessions/modest-vigilant-lamport/mnt/prism-brain-v2/DOMAIN_2_STRUCTURAL_COMPLETE v2.md",
    "/sessions/modest-vigilant-lamport/mnt/prism-brain-v2/DOMAIN_3_DIGITAL_RESILIENCE_COMPLETE.md",
    "/sessions/modest-vigilant-lamport/mnt/prism-brain-v2/DOMAIN_4_OPERATIONAL_MASTER (2).md",
]

OUTPUT_FILE = "/sessions/modest-vigilant-lamport/mnt/prism-brain-v2/frontend/data/risk_events_v2.json"

# Domain mapping
DOMAIN_MAP = {
    "PHY": "PHYSICAL",
    "STR": "Structural",
    "DIG": "Digital Resilience & Technology Sovereignty",
    "OPS": "Operational"
}

# Layer 1 mapping (more concise than full names)
LAYER_1_MAP = {
    "PHY": "PHYSICAL",
    "STR": "STRUCTURAL",
    "DIG": "DIGITAL",
    "OPS": "OPERATIONAL"
}

# Family code mapping (Event code prefix -> family code)
FAMILY_CODE_MAP = {
    "PHY-CLI": "1.1",
    "PHY-ENE": "1.2",
    "PHY-MAT": "1.3",
    "PHY-WAT": "1.4",
    "PHY-GEO": "1.5",
    "PHY-POL": "1.6",
    "PHY-BIO": "1.7",
    "STR-GEO": "2.1",
    "STR-TRD": "2.2",
    "STR-REG": "2.3",
    "STR-ECO": "2.4",
    "STR-ENP": "2.5",
    "STR-TEC": "2.6",
    "STR-FIN": "2.7",
    "DIG-CIC": "3.1",
    "DIG-RDE": "3.2",
    "DIG-SCC": "3.3",
    "DIG-FSD": "3.4",
    "DIG-CLS": "3.5",
    "DIG-HWS": "3.6",
    "DIG-SWS": "3.7",
    "OPS-MAR": "4.1",
    "OPS-AIR": "4.2",
    "OPS-RLD": "4.3",
    "OPS-CMP": "4.4",
    "OPS-SUP": "4.5",
    "OPS-MFG": "4.6",
    "OPS-WHS": "4.7",
}

# Family name mapping
FAMILY_NAME_MAP = {
    "1.1": "Climate Extremes & Weather Events",
    "1.2": "Energy Supply & Grid Stability",
    "1.3": "Natural Resources & Raw Materials",
    "1.4": "Water Resources & Quality",
    "1.5": "Geophysical Disasters",
    "1.6": "Contamination & Pollution",
    "1.7": "Biological & Pandemic Risks",
    "2.1": "Geopolitical Conflict & Instability",
    "2.2": "Trade & Economic Policy Shifts",
    "2.3": "Regulatory & Compliance Changes",
    "2.4": "Macroeconomic Shocks",
    "2.5": "Energy Transition & Climate Policy",
    "2.6": "Technology Policy & Regulatory Restrictions",
    "2.7": "Financial Market Disruptions",
    "3.1": "Critical Infrastructure Cyberattacks",
    "3.2": "Ransomware, Data Breaches & Exfiltration",
    "3.3": "Supply Chain Cyberattacks",
    "3.4": "Fraud, Social Engineering & Denial-of-Service",
    "3.5": "Cloud & Platform Sovereignty",
    "3.6": "Hardware, Industrial Equipment & Semiconductor Sovereignty",
    "3.7": "Software & AI Sovereignty",
    "4.1": "Port & Maritime Logistics",
    "4.2": "Air Freight & Aviation",
    "4.3": "Road & Rail Transport",
    "4.4": "Component & Materials Shortages",
    "4.5": "Supplier & Vendor Disruptions",
    "4.6": "Manufacturing & Production Disruptions",
    "4.7": "Warehouse & Inventory Management",
}


class RiskEventParser:
    def __init__(self):
        self.events = {}
        self.events_list = []

    def read_file(self, filepath: str) -> str:
        """Read file contents."""
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            return ""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def parse_taxonomy(self) -> Dict:
        """Parse the taxonomy file to extract event IDs, names, and structure."""
        content = self.read_file(TAXONOMY_FILE)
        events = {}

        # Extract all event definitions from the taxonomy
        # Format: X. EVENT-CODE: Event Name
        pattern = r'^\d+\.\s+([A-Z]+-[A-Z]+-\d{3}):\s+(.+?)$'

        for line in content.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                event_id = match.group(1)
                event_name = match.group(2).strip()

                # Extract family code from event ID
                family_prefix = "-".join(event_id.split("-")[:2])
                family_code = FAMILY_CODE_MAP.get(family_prefix, "")
                family_name = FAMILY_NAME_MAP.get(family_code, "")

                # Extract domain
                domain_prefix = event_id.split("-")[0]
                domain = DOMAIN_MAP.get(domain_prefix, "")
                layer_1_primary = LAYER_1_MAP.get(domain_prefix, "")

                events[event_id] = {
                    "Event_ID": event_id,
                    "Event_Name": event_name,
                    "family_code": family_code,
                    "family_name": family_name,
                    "domain": domain,
                    "layer_1_primary": layer_1_primary,
                    "domain_prefix": domain_prefix,
                }

        return events

    def extract_base_rate_from_taxonomy(self, content: str, family_code: str) -> Optional[Tuple[float, str]]:
        """Extract base rate range from taxonomy file for a family."""
        # Look for "Base Rate Range:" pattern
        pattern = r'Base Rate Range:\s*([\d.]+%?)\s*-\s*([\d.]+%?)'

        for match in re.finditer(pattern, content):
            rate1_str = match.group(1).rstrip('%')
            rate2_str = match.group(2).rstrip('%')

            try:
                rate1 = float(rate1_str)
                rate2 = float(rate2_str)
                # Convert percentage to decimal, use midpoint
                midpoint = (rate1 + rate2) / 2 / 100
                return midpoint, "Midpoint of taxonomy range"
            except ValueError:
                continue

        return None, ""

    def extract_event_details(self, content: str, event_id: str) -> Dict:
        """Extract detailed information for a specific event from content."""
        event_data = {
            "Event_Description": "",
            "base_probability": None,
            "confidence_level": "",
            "Geographic_Scope": "",
            "Time_Horizon": "",
            "Affected_Industries": "",
            "Universal_Base_Rate": None,
        }

        # Create case-insensitive pattern for event ID
        event_section_pattern = rf"^#+\s*(?:Event\s+)?{re.escape(event_id)}.*?$"

        # Find the event section
        lines = content.split('\n')
        event_start = -1
        for i, line in enumerate(lines):
            if re.search(event_section_pattern, line, re.IGNORECASE):
                event_start = i
                break

        if event_start == -1:
            return event_data

        # Extract content from event start until next major section or EOF
        # Look further ahead for detailed events
        event_content_lines = []
        for i in range(event_start, min(event_start + 300, len(lines))):
            line = lines[i]
            # Stop at next event (new ### or ##)
            if i > event_start and re.match(r'^(###|##)\s+(?:Event\s+)?[A-Z]+-[A-Z]+-\d{3}', line):
                break
            event_content_lines.append(line)

        event_text = '\n'.join(event_content_lines)

        # Extract Historical Base Rate (specific percentage)
        base_rate_match = re.search(r'(?:Historical\s+)?Base Rate[:\s]+\**([\d.]+)%', event_text, re.IGNORECASE)
        if base_rate_match:
            event_data["base_probability"] = float(base_rate_match.group(1)) / 100
            event_data["Universal_Base_Rate"] = float(base_rate_match.group(1)) / 100
        else:
            # Try Universal Base Rate format
            universal_match = re.search(r'Universal Base Rate:\s*([\d.]+)%', event_text, re.IGNORECASE)
            if universal_match:
                event_data["base_probability"] = float(universal_match.group(1)) / 100
                event_data["Universal_Base_Rate"] = float(universal_match.group(1)) / 100

        # Extract Confidence Level
        conf_match = re.search(r'\*\*Confidence Level:\*\*\s*(HIGH|MEDIUM|LOW|MEDIUM-HIGH|MEDIUM-LOW)', event_text, re.IGNORECASE)
        if not conf_match:
            conf_match = re.search(r'Confidence[:\s]+\**(HIGH|MEDIUM|LOW|MEDIUM-HIGH|MEDIUM-LOW)', event_text, re.IGNORECASE)
        if conf_match:
            event_data["confidence_level"] = conf_match.group(1).upper()

        # Extract Geographic Scope
        geo_match = re.search(r'\*\*Geographic Scope:\*\*\s*(.+?)(?:\n|$)', event_text, re.IGNORECASE)
        if not geo_match:
            geo_match = re.search(r'Geographic Scope:\s*(.+?)(?:\n|$)', event_text, re.IGNORECASE)
        if geo_match:
            event_data["Geographic_Scope"] = geo_match.group(1).strip()

        # Extract Time Horizon
        time_match = re.search(r'(?:\*\*)?(?:Time Horizon|Lead Time)[:\s]+\*?(.+?)(?:\n|$)', event_text, re.IGNORECASE)
        if time_match:
            time_str = time_match.group(1).strip()
            # Extract time period if present
            if 'annual' in time_str.lower():
                event_data["Time_Horizon"] = "Annual"
            elif 'quarterly' in time_str.lower():
                event_data["Time_Horizon"] = "Quarterly"
            elif 'monthly' in time_str.lower():
                event_data["Time_Horizon"] = "Monthly"
            elif 'daily' in time_str.lower():
                event_data["Time_Horizon"] = "Daily"
            elif 'episodic' in time_str.lower():
                event_data["Time_Horizon"] = "Episodic"

        # Extract affected industries
        industries_match = re.search(
            r'(?:\*\*)?(?:Affected\s+)?Industries.*?(?:\*\*)?\s*:\s*(.+?)(?=\n\n|\*\*|Geographic|Impact Assessment|$)',
            event_text,
            re.IGNORECASE | re.DOTALL
        )
        if industries_match:
            industries_text = industries_match.group(1).strip()
            # Clean up list formatting
            industries_text = re.sub(r'^[\d\-•]\.\s+', '', industries_text, flags=re.MULTILINE)
            industries_text = re.sub(r'^\-\s+', '', industries_text, flags=re.MULTILINE)
            event_data["Affected_Industries"] = industries_text[:500]  # Limit length

        # Extract description (first paragraph or definition)
        desc_match = re.search(
            r'(?:\*\*)?(?:Description|Definition)(?:\*\*)?\s*:\s*\n*(.+?)(?=\n\n|\*\*|Probability|Formula)',
            event_text,
            re.IGNORECASE | re.DOTALL
        )
        if desc_match:
            description = desc_match.group(1).strip()
            # Clean up formatting
            description = re.sub(r'\*\*', '', description)
            # Limit to reasonable length
            event_data["Event_Description"] = description[:800]

        return event_data

    def parse_detail_files(self, events: Dict) -> None:
        """Parse all detail files and extract event information."""

        for detail_file in DETAIL_FILES:
            print(f"Processing: {detail_file}")
            content = self.read_file(detail_file)

            if not content:
                continue

            # For each event in the events dict, extract details
            for event_id, event_info in events.items():
                # Check if this file likely contains this event
                if event_id in content:
                    details = self.extract_event_details(content, event_id)

                    # Merge with existing event info
                    event_info.update(details)

                    # If no base_probability found, try to get from taxonomy
                    if event_info.get("base_probability") is None:
                        family_code = event_info.get("family_code", "")
                        if family_code:
                            # Find family section in taxonomy
                            family_section = self.extract_family_section(family_code)
                            if family_section:
                                base_rate, source = self.extract_base_rate_from_taxonomy(family_section, family_code)
                                if base_rate:
                                    event_info["base_probability"] = base_rate

    def extract_family_section(self, family_code: str) -> Optional[str]:
        """Extract a family section from the taxonomy."""
        content = self.read_file(TAXONOMY_FILE)

        # Find the family in content
        pattern = rf'###\s+Family\s+{re.escape(family_code)}.*?(?=###\s+Family|\Z)'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)

        if match:
            return match.group(0)

        return None

    def build_output_events(self, events: Dict) -> List[Dict]:
        """Build the final output event list with all required fields."""
        output_events = []

        for event_id, event_info in events.items():
            family_code = event_info.get("family_code", "")
            domain_prefix = event_info.get("domain_prefix", "")
            layer_1_primary = event_info.get("layer_1_primary", "")

            output_event = {
                "Event_ID": event_id,
                "Event_Name": event_info.get("Event_Name", ""),
                "Event_Description": event_info.get("Event_Description", ""),
                "Layer_1_Primary": layer_1_primary,
                "Layer_1_Secondary": "",
                "Layer_2_Primary": event_info.get("family_name", ""),
                "Layer_2_Secondary": "",
                "Super_Risk": "NO",
                "Geographic_Scope": event_info.get("Geographic_Scope", ""),
                "Time_Horizon": event_info.get("Time_Horizon", "Annual"),
                "Baseline_Probability": 3,  # Default placeholder
                "Baseline_Impact": 3,  # Default placeholder
                "Source_Category": event_info.get("domain", ""),
                "base_probability": event_info.get("base_probability") or 0.01,
                "base_impact": 0.5,  # Default
                "family_code": family_code,
                "family_name": event_info.get("family_name", ""),
                "confidence_level": event_info.get("confidence_level", "MEDIUM"),
                "Affected_Industries": event_info.get("Affected_Industries", ""),
            }

            output_events.append(output_event)

        return sorted(output_events, key=lambda x: x["Event_ID"])

    def save_json(self, events: List[Dict], filepath: str) -> None:
        """Save events to JSON file."""
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        output = {
            "metadata": {
                "version": "2.0",
                "generated": "2026-02-19",
                "total_events": len(events),
                "domains": 4,
                "families": 28,
            },
            "events": events
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Output saved to: {filepath}")
        print(f"Total events: {len(events)}")

    def run(self):
        """Execute the full parsing pipeline."""
        print("=" * 80)
        print("PRISM BRAIN RISK EVENT PARSER")
        print("=" * 80)

        print("\n1. Parsing taxonomy file...")
        events = self.parse_taxonomy()
        print(f"   Found {len(events)} events in taxonomy")

        print("\n2. Parsing detail files...")
        self.parse_detail_files(events)

        print("\n3. Building output...")
        output_events = self.build_output_events(events)

        print("\n4. Saving to JSON...")
        self.save_json(output_events, OUTPUT_FILE)

        print("\n" + "=" * 80)
        print("PARSING COMPLETE")
        print("=" * 80)

        return output_events


if __name__ == "__main__":
    parser = RiskEventParser()
    events = parser.run()

    # Print sample
    if events:
        print("\nSample event (first):")
        print(json.dumps(events[0], indent=2))


