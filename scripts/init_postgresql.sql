-- =============================================================================
-- PostgreSQL Migration Script
-- =============================================================================
-- Bu script, projedeki tüm SQLite veritabanlarını PostgreSQL'e taşımak için
-- gerekli schema ve tablo yapılarını oluşturur.
--
-- Mevcut SQLite DB'ler ve karşılık gelen PostgreSQL schemaları:
--
--   SQLite DB             →  PostgreSQL Schema
--   ─────────────────────────────────────────────
--   mcp_configs.db        →  app          (API/Auth/MCP tabloları)
--   history.db            →  history      (Experiment tracking)
--   adult_income.db       →  agent_data   (Kullanıcı analiz verileri)
--   financial.db          →  data_gen     (Generate edilmiş test verileri)
--
-- Kullanım:
--   psql -U <user> -d <database> -f scripts/init_postgresql.sql
--
-- NOT: Bu script idempotent'tir (IF NOT EXISTS kullanır).
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- EXTENSIONS
-- ─────────────────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────────────────────────────────────
-- SCHEMA: app
-- Kaynak: mcp_configs.db  (API core — kullanıcılar, auth, MCP, projeler vs.)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS app;

-- users tablosu
CREATE TABLE IF NOT EXISTS app.users (
    id              SERIAL PRIMARY KEY,
    email           VARCHAR(255) NOT NULL UNIQUE,
    full_name       VARCHAR(255),
    hashed_password VARCHAR(255) NOT NULL,
    is_active       BOOLEAN DEFAULT TRUE,
    is_verified     BOOLEAN DEFAULT FALSE,
    verification_token          VARCHAR(255),
    verification_token_expires  TIMESTAMP WITH TIME ZONE,
    reset_password_token        VARCHAR(255),
    reset_password_expires      TIMESTAMP WITH TIME ZONE,
    failed_login_attempts       INTEGER DEFAULT 0,
    locked_until                TIMESTAMP WITH TIME ZONE,
    token_version               INTEGER DEFAULT 1,
    created_at                  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON app.users(email);
CREATE INDEX IF NOT EXISTS idx_users_verification_token ON app.users(verification_token);
CREATE INDEX IF NOT EXISTS idx_users_reset_password_token ON app.users(reset_password_token);

-- token_blacklist tablosu
CREATE TABLE IF NOT EXISTS app.token_blacklist (
    id          SERIAL PRIMARY KEY,
    token       VARCHAR(512) NOT NULL UNIQUE,
    expires_at  TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_token_blacklist_token ON app.token_blacklist(token);

-- data_groups tablosu
CREATE TABLE IF NOT EXISTS app.data_groups (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255),
    source_type VARCHAR(50),   -- "mcp" or "csv"
    created_at  INTEGER,
    user_id     INTEGER REFERENCES app.users(id) ON DELETE CASCADE,

    -- LLM schema intelligence cache
    db_intelligence_cache       JSONB,
    raw_schema_cache            TEXT,
    schema_cache_updated_at     INTEGER,

    -- Long-term memory for Talk to Data
    long_term_memory            TEXT
);

CREATE INDEX IF NOT EXISTS idx_data_groups_name ON app.data_groups(name);

-- mcp_configs tablosu
CREATE TABLE IF NOT EXISTS app.mcp_configs (
    id                          SERIAL PRIMARY KEY,
    name                        VARCHAR(255) UNIQUE,
    type                        VARCHAR(50),    -- "stdio" or "sse"
    command                     VARCHAR(1024),
    args                        JSONB,
    url                         VARCHAR(1024),
    env                         JSONB,
    schema_cache                TEXT,
    schema_cache_updated_at     INTEGER,
    column_stats_cache          TEXT,
    column_stats_cache_updated_at INTEGER,
    data_group_id               INTEGER REFERENCES app.data_groups(id) ON DELETE CASCADE,
    user_id                     INTEGER REFERENCES app.users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_mcp_configs_name ON app.mcp_configs(name);

-- csv_files tablosu
CREATE TABLE IF NOT EXISTS app.csv_files (
    id              SERIAL PRIMARY KEY,
    filename        VARCHAR(255),
    file_path       VARCHAR(1024),
    file_size       INTEGER,
    row_count       INTEGER,
    columns_schema  JSONB,
    column_stats_cache JSONB,
    created_at      INTEGER,
    data_group_id   INTEGER REFERENCES app.data_groups(id) ON DELETE CASCADE,
    user_id         INTEGER REFERENCES app.users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_csv_files_filename ON app.csv_files(filename);

-- chat_messages tablosu
CREATE TABLE IF NOT EXISTS app.chat_messages (
    id              SERIAL PRIMARY KEY,
    data_group_id   INTEGER REFERENCES app.data_groups(id),
    role            VARCHAR(50),   -- "user" or "assistant"
    content         TEXT,
    created_at      INTEGER,

    -- Visualization columns
    sql             TEXT,
    chart           JSONB,
    columns         JSONB,
    rows            JSONB,
    warning         TEXT
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_data_group_id ON app.chat_messages(data_group_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON app.chat_messages(created_at);

-- projects tablosu
CREATE TABLE IF NOT EXISTS app.projects (
    id              SERIAL PRIMARY KEY,
    title           VARCHAR(255),
    description     TEXT,
    query           TEXT,
    data_group_id   INTEGER REFERENCES app.data_groups(id) ON DELETE SET NULL,
    user_id         INTEGER REFERENCES app.users(id) ON DELETE CASCADE,
    status          VARCHAR(50) DEFAULT 'idle',   -- idle, running, completed, failed
    report          TEXT,
    error           TEXT,
    created_at      INTEGER DEFAULT EXTRACT(EPOCH FROM NOW())::INTEGER,
    updated_at      INTEGER DEFAULT EXTRACT(EPOCH FROM NOW())::INTEGER
);

CREATE INDEX IF NOT EXISTS idx_projects_title ON app.projects(title);

-- project_steps tablosu
CREATE TABLE IF NOT EXISTS app.project_steps (
    id          SERIAL PRIMARY KEY,
    project_id  INTEGER REFERENCES app.projects(id) ON DELETE CASCADE,
    node_name   VARCHAR(255),
    step_data   JSONB,
    created_at  INTEGER DEFAULT EXTRACT(EPOCH FROM NOW())::INTEGER
);

-- background_jobs tablosu
CREATE TABLE IF NOT EXISTS app.background_jobs (
    id              SERIAL PRIMARY KEY,
    project_id      INTEGER REFERENCES app.projects(id) ON DELETE CASCADE,
    task_type       VARCHAR(255) DEFAULT 'run_project' NOT NULL,
    query           TEXT,
    data_group_id   INTEGER,
    is_resume       INTEGER DEFAULT 0,
    force_restart   INTEGER DEFAULT 0,
    status          VARCHAR(50) DEFAULT 'pending',   -- pending, running, completed, failed
    error           TEXT,
    created_at      INTEGER DEFAULT EXTRACT(EPOCH FROM NOW())::INTEGER,
    updated_at      INTEGER DEFAULT EXTRACT(EPOCH FROM NOW())::INTEGER
);

CREATE INDEX IF NOT EXISTS idx_background_jobs_status ON app.background_jobs(status);
CREATE INDEX IF NOT EXISTS idx_background_jobs_project_id ON app.background_jobs(project_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- SCHEMA: history
-- Kaynak: history.db  (Experiment tracking)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS history;

CREATE TABLE IF NOT EXISTS history.experiments (
    id                  SERIAL PRIMARY KEY,
    timestamp           TEXT,
    task                TEXT,
    intent              TEXT,
    model_name          TEXT,
    metrics             JSONB,
    hyperparameters     JSONB,
    feature_names       JSONB,
    target_name         TEXT,
    dataset_shape       TEXT
);


-- ─────────────────────────────────────────────────────────────────────────────
-- SCHEMA: agent_data
-- Kaynak: adult_income.db ve diğer kullanıcı analiz veritabanları
-- Agent'ın analiz edeceği veri setlerini barındırır.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS agent_data;

-- adult_income tablosu (varsayılan örnek dataset)
-- NOT: Bu tablonun sütunları kullanılan dataset'e göre değişir.
-- Aşağıdaki yapı "UCI Adult Income" dataset'ine göre oluşturulmuştur.
CREATE TABLE IF NOT EXISTS agent_data.adult_income (
    id                  SERIAL PRIMARY KEY,
    age                 INTEGER,
    workclass           VARCHAR(255),
    fnlwgt              INTEGER,
    education           VARCHAR(255),
    education_num       INTEGER,
    marital_status      VARCHAR(255),
    occupation          VARCHAR(255),
    relationship        VARCHAR(255),
    race                VARCHAR(255),
    sex                 VARCHAR(50),
    capital_gain        INTEGER,
    capital_loss        INTEGER,
    hours_per_week      INTEGER,
    native_country      VARCHAR(255),
    income              VARCHAR(50)
);


-- ─────────────────────────────────────────────────────────────────────────────
-- SCHEMA: data_gen
-- Kaynak: financial.db  (Generate edilmiş sentetik finansal veri)
-- generate_financial_db.py / generate_mediumdata.py scriptleri ile oluşturulur
-- ─────────────────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS data_gen;

-- accounts tablosu (500K satır — müşteri hesapları)
CREATE TABLE IF NOT EXISTS data_gen.accounts (
    account_id          INTEGER PRIMARY KEY,
    customer_name       TEXT,
    age                 INTEGER,
    gender              TEXT,
    country             TEXT,
    city                TEXT,
    account_type        TEXT,
    balance             DOUBLE PRECISION,
    credit_limit        DOUBLE PRECISION,
    open_date           TEXT,
    risk_score          DOUBLE PRECISION,
    is_active           INTEGER
);

-- merchants tablosu (50K satır — satıcılar)
CREATE TABLE IF NOT EXISTS data_gen.merchants (
    merchant_id             INTEGER PRIMARY KEY,
    merchant_name           TEXT,
    category                TEXT,
    country                 TEXT,
    city                    TEXT,
    avg_transaction_amount  DOUBLE PRECISION,
    is_blacklisted          INTEGER
);

-- transactions tablosu (10M satır — işlemler, fraud dahil)
CREATE TABLE IF NOT EXISTS data_gen.transactions (
    transaction_id      INTEGER PRIMARY KEY,
    account_id          INTEGER,
    merchant_id         INTEGER,
    amount              DOUBLE PRECISION,
    currency            TEXT,
    transaction_type    TEXT,
    channel             TEXT,
    transaction_date    TEXT,
    hour_of_day         INTEGER,
    is_foreign          INTEGER,
    is_weekend          INTEGER,
    is_fraud            INTEGER,
    fraud_reason        TEXT
);

CREATE INDEX IF NOT EXISTS idx_txn_account  ON data_gen.transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_txn_merchant ON data_gen.transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_txn_date     ON data_gen.transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_txn_fraud    ON data_gen.transactions(is_fraud);


-- ─────────────────────────────────────────────────────────────────────────────
-- SEARCH PATH ayarı (opsiyonel)
-- Uygulamanın bağlandığı kullanıcı için default search_path ayarlayabilirsiniz
-- ─────────────────────────────────────────────────────────────────────────────
-- ALTER ROLE your_app_user SET search_path TO app, history, agent_data, data_gen, public;


-- ─────────────────────────────────────────────────────────────────────────────
-- GRANTS (opsiyonel — uygulamanız için ayrı bir role oluşturduysanız)
-- ─────────────────────────────────────────────────────────────────────────────
-- GRANT USAGE ON SCHEMA app, history, agent_data, data_gen TO your_app_role;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA app TO your_app_role;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA history TO your_app_role;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA agent_data TO your_app_role;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA data_gen TO your_app_role;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA app TO your_app_role;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA history TO your_app_role;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA agent_data TO your_app_role;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA data_gen TO your_app_role;


-- =============================================================================
-- BİTTİ ✓
-- =============================================================================
-- Tüm tablolar oluşturuldu. Şimdi .env dosyanızda bağlantı bilgilerini
-- PostgreSQL'e göre güncelleyin.
-- =============================================================================
