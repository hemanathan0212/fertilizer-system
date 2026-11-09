-- ============================================================================
-- COURSE: CAF7104 - Mini Project
-- PROJECT: AI Agent-Based Fertilizer Recommendation System
-- MODULE 3: SQLite Knowledge Base Schema (knowledge_base/schema.sql)
-- ============================================================================

-- Table 1: Master Fertilizer Information
CREATE TABLE IF NOT EXISTS fertilizers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    dosage_per_acre TEXT NOT NULL,
    application_method TEXT NOT NULL,
    n_content REAL NOT NULL,
    p_content REAL NOT NULL,
    k_content REAL NOT NULL
);

-- Table 2: Crop Specific Adjustments & Management Guidelines
CREATE TABLE IF NOT EXISTS crop_adjustments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crop_type TEXT NOT NULL,
    fertilizer_name TEXT NOT NULL,
    adjustment_notes TEXT NOT NULL,
    UNIQUE(crop_type, fertilizer_name)
);

-- Indexes for rapid retrieval by LangChain tools
CREATE INDEX IF NOT EXISTS idx_fert_name ON fertilizers(name);
CREATE INDEX IF NOT EXISTS idx_crop_fert ON crop_adjustments(crop_type, fertilizer_name);
