-- PRISM Brain Database Initialization
-- This script runs automatically when PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Set timezone
SET timezone = 'UTC';

-- Grant privileges (if needed)
GRANT ALL PRIVILEGES ON DATABASE prism_brain TO prism;

-- Note: Tables are created by SQLAlchemy ORM
-- This file is for additional database setup

-- Create indexes for common queries (if not created by ORM)
-- These are already defined in models.py but can be added here as backup

-- Performance tuning for PostgreSQL
-- Adjust based on available memory
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '768MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET max_wal_size = '4GB';
