"""
================================================================================
COURSE: CAF7104 - Mini Project
PROJECT: AI Agent-Based Fertilizer Recommendation System
MODULE 3: Fertilizer Knowledge Base Population (knowledge_base/seed_data.py)
================================================================================
Purpose:
  1. Initializes SQLite database 'knowledge_base/fertilizer_kb.db'.
  2. Applies DDL schema from 'knowledge_base/schema.sql'.
  3. Populates master fertilizer specifications (Dosage, Application Method, NPK ratio).
  4. Populates agronomic crop-specific adjustment notes for all 11 crop types.
  5. Provides lookup helper functions for the LangChain agent tools.
================================================================================
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "fertilizer_kb.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

# ------------------------------------------------------------------------------
# Master Fertilizer Data (7 Classes)
# ------------------------------------------------------------------------------
FERTILIZERS_DATA = [
    (
        "Urea",
        "45-55 kg/acre",
        "Top dressing broadcast over moist soil during active vegetative growth. Avoid direct contact with wet foliage or standing deep water to prevent volatilization loss.",
        46.0, 0.0, 0.0
    ),
    (
        "DAP",
        "40-50 kg/acre",
        "Basal application at sowing or transplanting. Band placement 4-5 cm below and to the side of seed furrow ensures early root absorption and seedling vigor.",
        18.0, 46.0, 0.0
    ),
    (
        "14-35-14",
        "45-50 kg/acre",
        "Basal soil incorporation during secondary tillage. Supplies high starter phosphorus alongside balanced nitrogen and potassium for sturdy root establishment.",
        14.0, 35.0, 14.0
    ),
    (
        "28-28",
        "35-45 kg/acre",
        "Split dose application: 50% as basal during sowing and 50% top-dressed 30-40 days after emergence during peak tillering/branching.",
        28.0, 28.0, 0.0
    ),
    (
        "17-17-17",
        "45-50 kg/acre",
        "Equally balanced complex fertilizer. Broadcast as basal dressing or apply through fertigation/furrow placement during vegetative stages.",
        17.0, 17.0, 17.0
    ),
    (
        "20-20",
        "40-50 kg/acre",
        "Basal dressing during land preparation or side-banding during first intercultivation weeding cycle.",
        20.0, 20.0, 0.0
    ),
    (
        "10-26-26",
        "45-55 kg/acre",
        "Deep placement at sowing or planting. High potassium formulation ideal for strengthening cell walls, drought resistance, and fruit/pod development.",
        10.0, 26.0, 26.0
    )
]

# ------------------------------------------------------------------------------
# Crop Adjustment Guidelines (11 Crops x Fertilizers)
# ------------------------------------------------------------------------------
CROP_ADJUSTMENTS_DATA = [
    # Sugarcane
    ("Sugarcane", "Urea", "Sugarcane has high nitrogen demand; split into 3 doses (planting, 45 DAP, 90 DAP). Incorporate before earthing up."),
    ("Sugarcane", "DAP", "Apply entire quantity as basal dose in furrows before planting setts to stimulate vigorous sett root initiation."),
    ("Sugarcane", "28-28", "Recommended for early vegetative flush. Apply 50% at 30 days and 50% at 75 days post planting."),
    ("Sugarcane", "10-26-26", "Critical for sugar accumulation and stalk thickness; apply as basal or during grand growth phase earthing up."),
    ("Sugarcane", "17-17-17", "Ideal for ratoon crops; apply within 15 days after harvesting main cane followed by immediate irrigation."),
    ("Sugarcane", "20-20", "Split into two doses: 45th day and 90th day to maintain uninterrupted tillering."),
    ("Sugarcane", "14-35-14", "Deep placement in planting trenches improves initial sett rooting and cane population."),

    # Maize
    ("Maize", "Urea", "Split dose: 33% at knee-high stage (30 DAS) and 33% at tasseling stage. Avoid applying when leaves are wet."),
    ("Maize", "DAP", "Apply 100% as basal dose at sowing. Place 5 cm away from seed line to avoid seedling burn."),
    ("Maize", "28-28", "Excellent for fast-growing hybrids; apply half at sowing and remainder at 35 DAS knee-high stage."),
    ("Maize", "10-26-26", "Apply as basal fertilizer; potassium ensures strong stalk strength and prevents lodging under heavy winds."),
    ("Maize", "17-17-17", "Apply basal dose at planting for uniform emergence and robust early canopy development."),
    ("Maize", "20-20", "Apply during second intercultivation (25 DAS) to boost cob elongation and kernel count."),
    ("Maize", "14-35-14", "Basal placement in seed bed promotes vigorous primary root system and early vigor."),

    # Cotton
    ("Cotton", "Urea", "Apply in 3 split doses: at thinning (25 DAS), square initiation (50 DAS), and peak boll development (75 DAS)."),
    ("Cotton", "DAP", "Apply entire dose basally in furrows before sowing to stimulate taproot deepening."),
    ("Cotton", "10-26-26", "High potassium is essential for boll retention, fiber strength, and resistance to sucking pests."),
    ("Cotton", "28-28", "Apply during vegetative flush; stop high nitrogen after peak flowering to prevent excessive vegetative shedding."),
    ("Cotton", "17-17-17", "Apply as basal or split at early squaring for balanced vegetative and reproductive development."),
    ("Cotton", "20-20", "Top dress before flowering alongside weeding; avoid over-application during rainy overcast days."),
    ("Cotton", "14-35-14", "Basal incorporation ensures early square formation and minimizes flower bud drop."),

    # Paddy
    ("Paddy", "Urea", "Split into 3 stages: 50% at active tillering (21 DAT), 25% at panicle initiation, and 25% at booting in saturated (not flooded) soil."),
    ("Paddy", "DAP", "Apply 100% as basal dose during final puddling stage just before transplanting."),
    ("Paddy", "17-17-17", "Suitable for medium duration cultivars; apply at transplanting and early tillering."),
    ("Paddy", "10-26-26", "Ensures high grain weight, panicle filling, and lodging resistance in dwarf high-yielding rice varieties."),
    ("Paddy", "28-28", "Split application at tillering and panicle initiation enhances fertile tillers per hill."),
    ("Paddy", "20-20", "Basal broadcast at final leveling followed by light incorporation into puddle."),
    ("Paddy", "14-35-14", "Basal application promotes deep root anchoring in soft clayey puddled rice soils."),

    # Wheat
    ("Wheat", "Urea", "Split: 50% basal and 50% top-dressed at first crown root initiation (CRI) irrigation (21 DAS)."),
    ("Wheat", "DAP", "Drill along with seed using seed-cum-fertilizer drill at 4-5 cm depth for maximal phosphorus availability."),
    ("Wheat", "10-26-26", "Promotes strong straw resistance against lodging and enhances plump grain filling during terminal heat."),
    ("Wheat", "28-28", "Ideal for irrigated dwarf wheat; split between sowing and first irrigation."),
    ("Wheat", "17-17-17", "Basal drilling at sowing gives uniform seedling emergence across dry tracts."),
    ("Wheat", "20-20", "Apply prior to second irrigation (tillering stage) to enhance spikelet count."),
    ("Wheat", "14-35-14", "Deep placement during seedbed preparation protects young seedlings against cold stress."),

    # Barley
    ("Barley", "Urea", "Moderate nitrogen requirement; apply in 2 splits (sowing and 30 DAS). Excess N decreases malting quality."),
    ("Barley", "DAP", "Apply full quantity as basal dressing to foster early winter hardiness and root growth."),
    ("Barley", "10-26-26", "Potassium application enhances malt grain plumpness and minimizes stem breakage."),
    ("Barley", "17-17-17", "Balanced application at sowing produces uniform tiller density."),
    ("Barley", "28-28", "Use conservatively for feed barley; avoid high dosage for malting grades."),
    ("Barley", "20-20", "Basal application at sowing ensures steady nitrogen and phosphorus supply."),
    ("Barley", "14-35-14", "Basal placement in furrows ensures high root density in coarse textured soils."),

    # Millets
    ("Millets", "Urea", "Drought tolerant crop; apply top dress only when adequate soil moisture is present (30 DAS)."),
    ("Millets", "DAP", "Drill at sowing; millets have rapid seedling emergence and benefit from early phosphorus."),
    ("Millets", "10-26-26", "Improves drought and heat stress tolerance; apply basal at field preparation."),
    ("Millets", "17-17-17", "Ideal all-round fertilizer for dryland millets on sandy or red soils."),
    ("Millets", "28-28", "Single basal or early split application at 25 DAS boosts head size."),
    ("Millets", "20-20", "Basal dose at land preparation supports fast vegetative tillering."),
    ("Millets", "14-35-14", "Basal application fosters robust root systems to mine moisture in semi-arid zones."),

    # Pulses
    ("Pulses", "Urea", "Legumes fix atmospheric nitrogen; use only a small starter dose of Urea (under 20 kg/acre) at sowing."),
    ("Pulses", "DAP", "Prime fertilizer for pulses; high phosphorus accelerates Rhizobium root nodulation and nitrogen fixation."),
    ("Pulses", "14-35-14", "Excellent for pulses; high phosphorus and potassium maximize nodule mass and pod filling."),
    ("Pulses", "10-26-26", "Potassium and phosphorus combination prevents flower drop and enhances seed weight."),
    ("Pulses", "17-17-17", "Use as starter basal dressing in low-fertility soils."),
    ("Pulses", "20-20", "Basal application at sowing fulfills initial N and seasonal P requirements."),
    ("Pulses", "28-28", "Use restricted quantities to avoid suppressing natural biological nitrogen fixation."),

    # Ground Nuts
    ("Ground Nuts", "Urea", "Groundnut is an oilseed legume; restrict nitrogen to starter dose at sowing to avoid leafy vegetative surge."),
    ("Ground Nuts", "DAP", "Basal placement at sowing stimulates root nodule formation and early peg initiation."),
    ("Ground Nuts", "10-26-26", "Crucial for pod filling, oil percentage, and shell thickness. Apply basal with gypsum supplement."),
    ("Ground Nuts", "14-35-14", "Provides high phosphorus for robust root nodulation and moderate potassium for kernel filling."),
    ("Ground Nuts", "17-17-17", "Apply basal dose at land preparation for uniform peg penetration."),
    ("Ground Nuts", "20-20", "Basal application at sowing meets starter phosphorus and nitrogen demands."),
    ("Ground Nuts", "28-28", "Apply cautiously at sowing; avoid late nitrogen top dressing."),

    # Oil seeds
    ("Oil seeds", "Urea", "Apply in 2 splits: 50% basal and 50% at rosette/pre-flowering stage with sufficient moisture."),
    ("Oil seeds", "DAP", "Basal placement at sowing ensures strong root establishment and uniform flowering."),
    ("Oil seeds", "10-26-26", "High potassium is vital for oil synthesis, test weight, and disease resistance."),
    ("Oil seeds", "14-35-14", "Basal dressing ensures vigorous siliqua/capsule formation and seed development."),
    ("Oil seeds", "17-17-17", "Balanced nutrient supply supports uniform crop stand and prolonged flowering."),
    ("Oil seeds", "20-20", "Basal incorporation ensures early root depth and branch proliferation."),
    ("Oil seeds", "28-28", "Top dressing at early flowering stage boosts seed number per plant."),

    # Tobacco
    ("Tobacco", "Urea", "Avoid excess nitrogen late in season as it increases nicotine harshness and delays leaf curing."),
    ("Tobacco", "DAP", "Apply full phosphorus requirement at transplanting along plant ridges."),
    ("Tobacco", "10-26-26", "Sulphate-based potassium is preferred for tobacco leaf burn quality and elasticity; apply basal."),
    ("Tobacco", "17-17-17", "Apply basal in plant furrows for balanced leaf expansion."),
    ("Tobacco", "28-28", "Apply early; avoid nitrogen application after topping stage."),
    ("Tobacco", "20-20", "Basal placement supports rapid transplant recovery and early leaf spread."),
    ("Tobacco", "14-35-14", "Basal dose accelerates root establishment and early vegetative vigor.")
]


def init_and_seed_database(db_path=DB_PATH, schema_path=SCHEMA_PATH):
    """
    Initializes SQLite database and seeds with master tables and records.
    """
    print("\n" + "="*70)
    print("MODULE 3: FERTILIZER KNOWLEDGE BASE INITIALIZATION")
    print("="*70)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Step 1: Execute Schema
    print(f"\n[Step 1] Applying schema from: {schema_path}")
    with open(schema_path, "r") as f:
        schema_sql = f.read()
    cursor.executescript(schema_sql)

    # Step 2: Seed Fertilizers
    print("\n[Step 2] Seeding master fertilizers table...")
    cursor.executemany(
        """
        INSERT INTO fertilizers (name, dosage_per_acre, application_method, n_content, p_content, k_content)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            dosage_per_acre=excluded.dosage_per_acre,
            application_method=excluded.application_method,
            n_content=excluded.n_content,
            p_content=excluded.p_content,
            k_content=excluded.k_content;
        """,
        FERTILIZERS_DATA
    )
    cursor.execute("SELECT COUNT(*) FROM fertilizers")
    fert_count = cursor.fetchone()[0]
    print(f"         Total Fertilizers Seeded: {fert_count}")

    # Step 3: Seed Crop Adjustments
    print("\n[Step 3] Seeding crop adjustments table...")
    cursor.executemany(
        """
        INSERT INTO crop_adjustments (crop_type, fertilizer_name, adjustment_notes)
        VALUES (?, ?, ?)
        ON CONFLICT(crop_type, fertilizer_name) DO UPDATE SET
            adjustment_notes=excluded.adjustment_notes;
        """,
        CROP_ADJUSTMENTS_DATA
    )
    cursor.execute("SELECT COUNT(*) FROM crop_adjustments")
    adj_count = cursor.fetchone()[0]
    print(f"         Total Crop Adjustments Seeded: {adj_count}")

    conn.commit()
    conn.close()

    print(f"\n[Step 4] Verification Complete. Database ready at: {db_path}")
    print("="*70)
    print("MODULE 3: DATABASE SEEDING COMPLETED SUCCESSFULLY")
    print("="*70)


def query_fertilizer_details(fertilizer_name: str, db_path=DB_PATH) -> dict:
    """
    Helper function to query fertilizer details from SQLite database.
    """
    if not os.path.exists(db_path):
        init_and_seed_database(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name, dosage_per_acre, application_method, n_content, p_content, k_content FROM fertilizers WHERE LOWER(name) = LOWER(?)",
        (fertilizer_name.strip(),)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return {
        "name": fertilizer_name,
        "dosage_per_acre": "40-50 kg/acre (General agronomic guideline)",
        "application_method": "Apply as basal or top dressing according to local agricultural university package of practices.",
        "n_content": 0.0,
        "p_content": 0.0,
        "k_content": 0.0
    }


def query_crop_adjustment(crop_type: str, fertilizer_name: str, db_path=DB_PATH) -> str:
    """
    Helper function to query crop-specific adjustment notes from SQLite database.
    """
    if not os.path.exists(db_path):
        init_and_seed_database(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT adjustment_notes FROM crop_adjustments WHERE LOWER(crop_type) = LOWER(?) AND LOWER(fertilizer_name) = LOWER(?)",
        (crop_type.strip(), fertilizer_name.strip())
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return row[0]
    return f"For {crop_type}, apply {fertilizer_name} according to the standard local agronomic timetable, ensuring sufficient soil moisture at application."


if __name__ == "__main__":
    init_and_seed_database()
    # Test query
    sample_fert = query_fertilizer_details("Urea")
    sample_adj = query_crop_adjustment("Sugarcane", "Urea")
    print("\n[Sample Query Verification]")
    print(f"Fertilizer Details (Urea): {sample_fert}")
    print(f"Crop Adjustment (Sugarcane + Urea): {sample_adj}")
