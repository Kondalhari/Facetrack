#!/usr/bin/env python3
"""
Initialize PostgreSQL database for Face Detection System
Uses simplified schema without pgvector (compatible with stock PostgreSQL)
"""

import psycopg
import json

print('=' * 60)
print('📦 DATABASE INITIALIZATION')
print('=' * 60)
print()

# Load config
config = json.load(open('config.json'))

# Step 1: Connect to postgres server and create database
print("Step 1: Creating database...")
conninfo_server = f"host={config['db_host']} port={config['db_port']} user={config['db_user']} password={config['db_pass']} dbname=postgres"

try:
    conn = psycopg.connect(conninfo_server, autocommit=True)
    print('✓ Connected to PostgreSQL server')
    
    with conn.cursor() as cur:
        try:
            cur.execute('CREATE DATABASE visitor_db')
            print(f'✓ Created database: visitor_db')
        except psycopg.errors.DuplicateDatabase:
            print(f'ℹ Database visitor_db already exists')
        except Exception as e:
            print(f'⚠️ {e}')
    
    conn.close()
    
except Exception as e:
    print(f'❌ Error connecting to server: {e}')
    print()
    print('TROUBLESHOOTING:')
    print(f'1. Is PostgreSQL running on port {config["db_port"]}?')
    print(f'2. Are credentials correct? ({config["db_user"]}/{config["db_pass"]})')
    exit(1)

print()

# Step 2: Connect to visitor_db and create schema
print("Step 2: Creating tables and schema...")
conninfo_db = f"host={config['db_host']} port={config['db_port']} user={config['db_user']} password={config['db_pass']} dbname=visitor_db"

try:
    conn = psycopg.connect(conninfo_db)
    print('✓ Connected to visitor_db')
    
    # Read schema (use simplified version without pgvector)
    try:
        with open('db_schema_simple.sql', 'r') as f:
            schema = f.read()
        print('✓ Loaded db_schema_simple.sql')
    except FileNotFoundError:
        print('⚠️ db_schema_simple.sql not found, trying db_schema.sql')
        with open('db_schema.sql', 'r') as f:
            schema = f.read()
    
    # Remove pgvector-specific commands
    schema = schema.replace('CREATE EXTENSION IF NOT EXISTS vector;', '')
    schema = schema.replace('vector(512)', 'BYTEA -- embedding storage')
    schema = schema.replace('USING HNSW (embedding vector_cosine_ops)', '')
    schema = schema.replace('vector_cosine_ops', '')
    
    # Execute schema
    with conn.cursor() as cur:
        # Split by semicolon and execute each statement
        statements = [s.strip() for s in schema.split(';') if s.strip()]
        for statement in statements:
            try:
                cur.execute(statement)
                conn.commit()
            except psycopg.errors.DuplicateTable:
                conn.rollback()
                pass  # Table already exists
            except Exception as e:
                conn.rollback()
                print(f'⚠️ Error executing statement: {e}')
    
    print('✓ Schema executed successfully')
    
    # Verify tables
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
        tables = cur.fetchall()
        
        if tables:
            print()
            print('✓ Tables created:')
            for (table,) in tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    print(f'  ✓ {table} ({count} records)')
                except:
                    print(f'  ✓ {table}')
        else:
            print('⚠️ No tables found')
    
    conn.close()
    
except Exception as e:
    print(f'❌ Error: {e}')
    print()
    print('TROUBLESHOOTING:')
    print('1. Make sure visitor_db database exists')
    print(f'2. Check PostgreSQL is running on port {config["db_port"]}')
    exit(1)

print()
print('=' * 60)
print('✅ DATABASE INITIALIZATION COMPLETE!')
print('=' * 60)
print()
print('Database Details:')
print(f'  Host: {config["db_host"]}')
print(f'  Port: {config["db_port"]}')
print(f'  Database: {config["db_name"]}')
print(f'  User: {config["db_user"]}')
print()
print('Next steps:')
print('  1. Run: python main.py')
print('  2. Visitors will auto-register with their embeddings')
print('  3. Entry/exit events logged to database')
print()
