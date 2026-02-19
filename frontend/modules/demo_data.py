"""
Demo data seeding for PRISM Brain.
Creates 5 demo clients with pre-filled industry data and processes
so the app is ready for testing immediately after startup.

Process IDs reference the new 2-level process framework:
  depth 1 = macro-processes (1–32)
  depth 2 = sub-processes  (e.g. 1.1, 7.3, 26.2)
"""
import logging
from modules.database import get_all_clients, create_client, add_client_process

logger = logging.getLogger(__name__)

DEMO_CLIENTS = [
    {
        "name": "AutoStahl GmbH",
        "location": "Stuttgart, Germany",
        "industry": "Manufacturing",
        "revenue": 850000000,
        "employees": 4500,
        "currency": "EUR",
        "export_percentage": 65,
        "primary_markets": "Germany, China, USA, France, UK",
        "sectors": "Automotive, Steel, Components",
        "notes": "Demo client - German automotive & steel manufacturer with global supply chain. Focus on electric vehicle components and traditional powertrain. High exposure to energy prices, semiconductor shortages, and EU regulatory changes.",
        "processes": [
            {"id": "1.1", "name": "Integrity of service buildings", "category": "A", "criticality": 120000},
            {"id": "4.1", "name": "Production equipment and machinery", "category": "A", "criticality": 420000},
            {"id": "4.2", "name": "Measurement and testing equipment", "category": "A", "criticality": 180000},
            {"id": "7.1", "name": "Direct materials and components", "category": "B", "criticality": 850000},
            {"id": "9.1", "name": "Road freight", "category": "B", "criticality": 280000},
            {"id": "11.1", "name": "Grid electricity supply", "category": "B", "criticality": 380000},
            {"id": "26.1", "name": "Network infrastructure & connectivity", "category": "D", "criticality": 210000},
            {"id": "28.1", "name": "Production planning & scheduling", "category": "D", "criticality": 350000},
            {"id": "29.1", "name": "Incoming quality control", "category": "D", "criticality": 175000},
            {"id": "32.1", "name": "Preventive maintenance programs", "category": "D", "criticality": 145000},
        ]
    },
    {
        "name": "NordPharma AS",
        "location": "Oslo, Norway",
        "industry": "Manufacturing",
        "revenue": 400000000,
        "employees": 2200,
        "currency": "EUR",
        "export_percentage": 80,
        "primary_markets": "Norway, Sweden, Germany, USA, Japan",
        "sectors": "Pharmaceuticals, Biotech, Medical Devices",
        "notes": "Demo client - Norwegian pharmaceutical company specializing in oncology and rare diseases. GMP-regulated manufacturing with cold chain logistics. High exposure to regulatory changes, clinical trial risks, and data privacy requirements.",
        "processes": [
            {"id": "2.1", "name": "Temperature-controlled storage for raw materials", "category": "A", "criticality": 520000},
            {"id": "3.1", "name": "HVAC and climate control", "category": "A", "criticality": 350000},
            {"id": "7.1", "name": "Direct materials and components", "category": "B", "criticality": 680000},
            {"id": "9.1", "name": "Road freight", "category": "B", "criticality": 180000},
            {"id": "18.1", "name": "Regulatory and legal compliance", "category": "B", "criticality": 420000},
            {"id": "26.1", "name": "Network infrastructure & connectivity", "category": "D", "criticality": 250000},
            {"id": "29.1", "name": "Incoming quality control", "category": "D", "criticality": 310000},
            {"id": "30.1", "name": "Workforce planning & scheduling", "category": "D", "criticality": 165000},
        ]
    },
    {
        "name": "FreshHarvest BV",
        "location": "Rotterdam, Netherlands",
        "industry": "Manufacturing",
        "revenue": 200000000,
        "employees": 1800,
        "currency": "EUR",
        "export_percentage": 45,
        "primary_markets": "Netherlands, Belgium, Germany, France, UK",
        "sectors": "Food Processing, Cold Chain Logistics, Agriculture",
        "notes": "Demo client - Dutch food & beverage company producing fresh and frozen products for European retail. HACCP-certified, operates cold chain from farm to shelf. High exposure to food safety regulations, climate events affecting crops, and energy costs for refrigeration.",
        "processes": [
            {"id": "2.1", "name": "Temperature-controlled storage for raw materials", "category": "A", "criticality": 540000},
            {"id": "2.2", "name": "Humidity-controlled storage conditions", "category": "A", "criticality": 210000},
            {"id": "7.1", "name": "Direct materials and components", "category": "B", "criticality": 320000},
            {"id": "9.1", "name": "Road freight", "category": "B", "criticality": 185000},
            {"id": "13.1", "name": "Municipal/industrial water supply", "category": "B", "criticality": 150000},
            {"id": "15.1", "name": "Industrial waste disposal", "category": "B", "criticality": 120000},
            {"id": "29.1", "name": "Incoming quality control", "category": "D", "criticality": 280000},
            {"id": "29.3", "name": "Regulatory compliance & certifications", "category": "D", "criticality": 95000},
        ]
    },
    {
        "name": "CyberShield Solutions Ltd",
        "location": "London, United Kingdom",
        "industry": "Technology & Services",
        "revenue": 150000000,
        "employees": 800,
        "currency": "GBP",
        "export_percentage": 55,
        "primary_markets": "UK, USA, Germany, Singapore, Australia",
        "sectors": "Cybersecurity, Cloud Services, Managed SOC",
        "notes": "Demo client - UK-based cybersecurity firm providing managed detection & response, penetration testing, and compliance consulting. Operates 24/7 SOC. High exposure to talent shortages, its own cyber risks, cloud provider dependencies, and regulatory complexity across jurisdictions.",
        "processes": [
            {"id": "8.1", "name": "SaaS/PaaS/IaaS service delivery", "category": "B", "criticality": 480000},
            {"id": "14.1", "name": "Fixed-line broadband and fiber optic", "category": "B", "criticality": 320000},
            {"id": "23.1", "name": "Sales pipeline management", "category": "C", "criticality": 210000},
            {"id": "26.1", "name": "Network infrastructure & connectivity", "category": "D", "criticality": 520000},
            {"id": "27.1", "name": "Security operations center (SOC)", "category": "D", "criticality": 480000},
            {"id": "27.2", "name": "Identity & access management (IAM)", "category": "D", "criticality": 250000},
            {"id": "30.1", "name": "Workforce planning & scheduling", "category": "D", "criticality": 195000},
        ]
    },
    {
        "name": "SolarVerde Energia SA",
        "location": "Madrid, Spain",
        "industry": "Energy & Utilities",
        "revenue": 300000000,
        "employees": 1200,
        "currency": "EUR",
        "export_percentage": 30,
        "primary_markets": "Spain, Portugal, Italy, Morocco, Chile",
        "sectors": "Solar Energy, Wind Energy, Grid Infrastructure",
        "notes": "Demo client - Spanish renewable energy company operating solar and wind farms across Southern Europe and North Africa. Manages grid-connected assets with battery storage. High exposure to energy policy changes, grid instability, critical material shortages (lithium, rare earths), and extreme weather events.",
        "processes": [
            {"id": "1.1", "name": "Integrity of service buildings", "category": "A", "criticality": 95000},
            {"id": "4.3", "name": "Safety and protection systems", "category": "A", "criticality": 310000},
            {"id": "5.1", "name": "Ecosystem services and biodiversity", "category": "A", "criticality": 140000},
            {"id": "11.1", "name": "Grid electricity supply", "category": "B", "criticality": 620000},
            {"id": "18.1", "name": "Regulatory and legal compliance", "category": "B", "criticality": 250000},
            {"id": "24.1", "name": "Strategic planning & risk governance", "category": "C", "criticality": 185000},
            {"id": "26.1", "name": "Network infrastructure & connectivity", "category": "D", "criticality": 220000},
            {"id": "32.1", "name": "Preventive maintenance programs", "category": "D", "criticality": 480000},
        ]
    },
]


def seed_demo_clients():
    """
    Create demo clients if they don't already exist.
    Safe to call multiple times - checks for existing clients by name.
    """
    try:
        existing = get_all_clients()
        if existing is None:
            existing = []

        # Build set of existing client names (handle both dict formats)
        existing_names = set()
        for c in existing:
            if isinstance(c, dict):
                name = c.get("name") or c.get("company_name", "")
                if name:
                    existing_names.add(name)

        created_count = 0
        for client_data in DEMO_CLIENTS:
            if client_data["name"] in existing_names:
                logger.info(f"Demo client '{client_data['name']}' already exists, skipping.")
                continue

            # Create the client
            client_id = create_client(
                name=client_data["name"],
                location=client_data.get("location", ""),
                industry=client_data.get("industry", ""),
                revenue=client_data.get("revenue", 0),
                employees=client_data.get("employees", 0),
                currency=client_data.get("currency", "EUR"),
                export_percentage=client_data.get("export_percentage", 0),
                primary_markets=client_data.get("primary_markets", ""),
                sectors=client_data.get("sectors", ""),
                notes=client_data.get("notes", ""),
            )

            if client_id:
                # Add processes with criticality values
                for proc in client_data.get("processes", []):
                    try:
                        add_client_process(
                            client_id=client_id,
                            process_id=proc["id"],
                            process_name=proc["name"],
                            category=proc.get("category", ""),
                            criticality_per_day=proc.get("criticality", 0),
                        )
                    except Exception as e:
                        logger.warning(f"Could not add process {proc['id']} to {client_data['name']}: {e}")

                created_count += 1
                logger.info(f"Created demo client: {client_data['name']} (id={client_id}) with {len(client_data.get('processes', []))} processes")

        if created_count > 0:
            logger.info(f"Demo data seeding complete: created {created_count} new client(s)")
        else:
            logger.info("Demo data seeding: all clients already exist, nothing to create")

    except Exception as e:
        logger.error(f"Error seeding demo clients: {e}")
