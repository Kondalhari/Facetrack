-- Database schema for Face Detection & Tracking System
-- Simplified version without pgvector (uses BYTEA for embeddings)

-- Table to store unique, registered visitors and their face embeddings
CREATE TABLE IF NOT EXISTS Visitors (
    visitor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    -- InsightFace embeddings stored as BYTEA (binary data)
    embedding BYTEA NOT NULL,
    -- Store embedding dimension and metadata
    embedding_dim INTEGER DEFAULT 512
);

-- Table to log every single entry and exit event
CREATE TABLE IF NOT EXISTS Events (
    event_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    visitor_id UUID NOT NULL REFERENCES Visitors(visitor_id) ON DELETE CASCADE,
    event_type VARCHAR(10) NOT NULL CHECK (event_type IN ('entry', 'exit')),
    -- Path on the filesystem to the saved cropped image
    cropped_image_path VARCHAR(255),
    -- Confidence score for the detection
    confidence FLOAT DEFAULT 0.0
);

-- Create index on event_type for faster queries
CREATE INDEX IF NOT EXISTS idx_events_type ON Events(event_type);

-- Create index on visitor_id for faster queries
CREATE INDEX IF NOT EXISTS idx_events_visitor ON Events(visitor_id);

-- Create index on timestamp for time-range queries
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON Events(timestamp);

-- Create index on visitor first_seen for analytics
CREATE INDEX IF NOT EXISTS idx_visitors_first_seen ON Visitors(first_seen);

-- Create view for visitor statistics
CREATE OR REPLACE VIEW visitor_stats AS
SELECT 
    v.visitor_id,
    COUNT(CASE WHEN e.event_type = 'entry' THEN 1 END) as entry_count,
    COUNT(CASE WHEN e.event_type = 'exit' THEN 1 END) as exit_count,
    COUNT(*) as total_events,
    v.first_seen,
    v.last_seen,
    MAX(e.timestamp) as last_event_time
FROM Visitors v
LEFT JOIN Events e ON v.visitor_id = e.visitor_id
GROUP BY v.visitor_id, v.first_seen, v.last_seen;

-- Create view for today's activity
CREATE OR REPLACE VIEW today_activity AS
SELECT 
    COUNT(DISTINCT visitor_id) as unique_visitors,
    COUNT(CASE WHEN event_type = 'entry' THEN 1 END) as total_entries,
    COUNT(CASE WHEN event_type = 'exit' THEN 1 END) as total_exits,
    MAX(timestamp) as last_activity
FROM Events
WHERE DATE(timestamp) = CURRENT_DATE;
