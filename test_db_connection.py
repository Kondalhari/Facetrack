#!/usr/bin/env python3
"""
Test PostgreSQL database connection with pgvector
"""

import json
import psycopg

# Load config
config = json.load(open('config.json'))

# Build connection string
conninfo = f"host={config['db_host']} port={config['db_port']} user={config['db_user']} password={config['db_pass']} dbname={config['db_name']}"

print('=' * 60)
print('🔍 DATABASE CONNECTION TEST')
print('=' * 60)
print(f"Connecting to: {config['db_host']}:{config['db_port']}/{config['db_name']}")
print()

try:
    conn = psycopg.connect(conninfo)
    print('✅ CONNECTION SUCCESSFUL!')
    print()
    
    # Get database info
    with conn.cursor() as cur:
        cur.execute('SELECT version()')
        version = cur.fetchone()[0]
        print(f'PostgreSQL Version: {version.split(",")[0]}')
        
        # Check for pgvector extension
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        pgvector = cur.fetchone()
        if pgvector:
            print('✓ pgvector extension installed')
        else:
            print('⚠️ pgvector extension NOT installed')
        
        # Check for tables
        cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'visitors')")
        has_visitors = cur.fetchone()[0]
        if has_visitors:
            print('✓ Visitors table exists')
            # Count visitors
            cur.execute('SELECT COUNT(*) FROM visitors')
            count = cur.fetchone()[0]
            print(f'  Total visitors: {count}')
        else:
            print('⚠️ Visitors table NOT found')
            print('  → Need to run schema: psql -U postgres -d visitor_db -f db_schema.sql')
        
        cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'events')")
        has_events = cur.fetchone()[0]
        if has_events:
            print('✓ Events table exists')
            # Count events
            cur.execute('SELECT COUNT(*) FROM events')
            count = cur.fetchone()[0]
            print(f'  Total events: {count}')
        else:
            print('⚠️ Events table NOT found')
    
    conn.close()
    print()
    print('=' * 60)
    print('✅ DATABASE IS READY!')
    print('=' * 60)
    
except psycopg.OperationalError as e:
    print(f'❌ CONNECTION FAILED!')
    print(f'Error: {e}')
    print()
    print('TROUBLESHOOTING:')
    print('1. Make sure PostgreSQL is running on port 8055')
    print('2. Verify credentials: postgres/8055')
    print('3. Check if database "visitor_db" exists')
    print('4. If not, create it: createdb visitor_db')
    print('5. Run schema: psql -U postgres -d visitor_db -f db_schema.sql')
    print()
    print('Quick setup:')
    print('  psql -U postgres -p 8055 -c "CREATE DATABASE visitor_db;"')
    print('  psql -U postgres -p 8055 -d visitor_db -f db_schema.sql')
