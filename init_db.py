#!/usr/bin/env python3
"""
Initialize PostgreSQL database with pgvector for Face Detection System
"""

import psycopg

print('=' * 60)
print('📦 DATABASE INITIALIZATION')
print('=' * 60)
print()

# Step 1: Connect to postgres server and create database
print("Step 1: Creating database...")
conninfo_server = 'host=localhost port=8055 user=postgres password=8055 dbname=postgres'

try:
    conn = psycopg.connect(conninfo_server, autocommit=True)
    print('✓ Connected to PostgreSQL server')
    
    with conn.cursor() as cur:
        try:
            cur.execute('CREATE DATABASE visitor_db')
            print('✓ Created database: visitor_db')
        except psycopg.errors.DuplicateDatabase:
            print('ℹ Database visitor_db already exists')
    
    conn.close()
    
except Exception as e:
    print(f'❌ Error connecting to server: {e}')
    print()
    print('TROUBLESHOOTING:')
    print('1. Is PostgreSQL running on port 8055?')
    print('2. Are credentials correct? (postgres/8055)')
    print('3. Try: psql -U postgres -p 8055 -c "CREATE DATABASE visitor_db;"')
    exit(1)

print()

# Step 2: Connect to visitor_db and create schema
print("Step 2: Creating tables and schema...")
conninfo_db = 'host=localhost port=8055 user=postgres password=8055 dbname=visitor_db'

try:
    conn = psycopg.connect(conninfo_db)
    print('✓ Connected to visitor_db')
    
    # Read schema
    with open('db_schema.sql', 'r') as f:
        schema = f.read()
    
    # Execute schema
    with conn.cursor() as cur:
        cur.execute(schema)
        conn.commit()
    
    print('✓ Schema executed successfully')
    
    # Verify tables
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
        tables = cur.fetchall()
        
        if tables:
            print()
            print('✓ Tables created:')
            for (table,) in tables:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                print(f'  ✓ {table} ({count} records)')
        else:
            print('⚠️ No tables found')
    
    conn.close()
    
except Exception as e:
    print(f'❌ Error: {e}')
    print()
    print('TROUBLESHOOTING:')
    print('1. Make sure visitor_db database exists')
    print('2. Check PostgreSQL is running on port 8055')
    print('3. Try manually: psql -U postgres -p 8055 -d visitor_db -f db_schema.sql')
    exit(1)

print()
print('=' * 60)
print('✅ DATABASE INITIALIZATION COMPLETE!')
print('=' * 60)
print()
print('Next steps:')
print('1. Run main.py to start detection')
print('2. Visitors will auto-register with their embeddings')
print('3. Entry/exit events logged to database')
print()
