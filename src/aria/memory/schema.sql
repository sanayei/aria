-- ARIA Conversation History Database Schema
--
-- This schema defines the SQLite database structure for storing conversation
-- history, including sessions, messages, and tool calls.
--
-- Schema Version: 1
-- Created: 2024-12-16

-- ============================================================================
-- Schema Metadata
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Insert initial schema version
INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Initial schema with sessions, messages, and tool_calls');

-- ============================================================================
-- Sessions Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0),
    metadata TEXT DEFAULT '{}',  -- JSON object for additional data
    CONSTRAINT valid_metadata CHECK(json_valid(metadata))
);

-- Indexes for sessions
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_message_count ON sessions(message_count);

-- ============================================================================
-- Messages Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}',  -- JSON object for additional data
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    CONSTRAINT valid_metadata CHECK(json_valid(metadata))
);

-- Indexes for messages
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp ON messages(session_id, timestamp ASC);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC);

-- ============================================================================
-- Tool Calls Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS tool_calls (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_input TEXT NOT NULL,  -- JSON object
    tool_output TEXT,  -- JSON object, NULL if pending/error
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'success', 'error', 'denied')),
    duration_ms INTEGER CHECK(duration_ms >= 0),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
    CONSTRAINT valid_tool_input CHECK(json_valid(tool_input)),
    CONSTRAINT valid_tool_output CHECK(tool_output IS NULL OR json_valid(tool_output))
);

-- Indexes for tool_calls
CREATE INDEX IF NOT EXISTS idx_tool_calls_message_id ON tool_calls(message_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_status ON tool_calls(status);
CREATE INDEX IF NOT EXISTS idx_tool_calls_timestamp ON tool_calls(timestamp DESC);

-- ============================================================================
-- Triggers for automatic timestamp updates
-- ============================================================================

-- Update session.updated_at when session is modified
CREATE TRIGGER IF NOT EXISTS update_session_timestamp
AFTER UPDATE ON sessions
FOR EACH ROW
BEGIN
    UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Update session.updated_at when a message is added
CREATE TRIGGER IF NOT EXISTS update_session_on_message_insert
AFTER INSERT ON messages
FOR EACH ROW
BEGIN
    UPDATE sessions
    SET updated_at = CURRENT_TIMESTAMP,
        message_count = message_count + 1
    WHERE id = NEW.session_id;
END;

-- Update session.updated_at when a message is deleted
CREATE TRIGGER IF NOT EXISTS update_session_on_message_delete
AFTER DELETE ON messages
FOR EACH ROW
BEGIN
    UPDATE sessions
    SET updated_at = CURRENT_TIMESTAMP,
        message_count = message_count - 1
    WHERE id = OLD.session_id;
END;

-- ============================================================================
-- Views for common queries
-- ============================================================================

-- View for session summaries with latest message preview
CREATE VIEW IF NOT EXISTS session_summaries AS
SELECT
    s.id,
    s.title,
    s.created_at,
    s.updated_at,
    s.message_count,
    COUNT(DISTINCT tc.id) as tool_call_count,
    (
        SELECT substr(m.content, 1, 100)
        FROM messages m
        WHERE m.session_id = s.id
        ORDER BY m.timestamp DESC
        LIMIT 1
    ) as last_message_preview
FROM sessions s
LEFT JOIN messages m ON s.id = m.session_id
LEFT JOIN tool_calls tc ON m.id = tc.message_id
GROUP BY s.id
ORDER BY s.updated_at DESC;

-- View for messages with their tool calls
CREATE VIEW IF NOT EXISTS messages_with_tools AS
SELECT
    m.id,
    m.session_id,
    m.role,
    m.content,
    m.timestamp,
    m.metadata,
    json_group_array(
        CASE
            WHEN tc.id IS NOT NULL THEN json_object(
                'id', tc.id,
                'tool_name', tc.tool_name,
                'status', tc.status,
                'duration_ms', tc.duration_ms
            )
            ELSE NULL
        END
    ) FILTER (WHERE tc.id IS NOT NULL) as tool_calls
FROM messages m
LEFT JOIN tool_calls tc ON m.id = tc.message_id
GROUP BY m.id;

-- ============================================================================
-- Helper functions using JSON
-- ============================================================================

-- Note: These are example queries, not stored functions (SQLite doesn't support UDFs in SQL)
-- Use these patterns in application code:

-- Example: Get session metadata value
-- SELECT json_extract(metadata, '$.key_name') FROM sessions WHERE id = ?;

-- Example: Update session metadata
-- UPDATE sessions SET metadata = json_set(metadata, '$.key_name', 'value') WHERE id = ?;

-- Example: Search messages by content
-- SELECT * FROM messages WHERE content LIKE '%search_term%';

-- Example: Get tool call statistics by session
-- SELECT
--     s.id,
--     s.title,
--     COUNT(tc.id) as total_calls,
--     SUM(CASE WHEN tc.status = 'success' THEN 1 ELSE 0 END) as successful_calls,
--     AVG(tc.duration_ms) as avg_duration_ms
-- FROM sessions s
-- LEFT JOIN messages m ON s.id = m.session_id
-- LEFT JOIN tool_calls tc ON m.id = tc.message_id
-- GROUP BY s.id;
