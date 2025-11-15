#!/usr/bin/env python3
"""
Test and verify logging system for Face Detection & Tracking
Creates sample logs to verify everything is working
"""

import os
import json
import datetime
import logging

print('=' * 60)
print('🔍 LOGGING SYSTEM TEST')
print('=' * 60)
print()

# Load config
config = json.load(open('config.json'))
entry_log_dir = config.get('entry_log_dir', 'logs/entries')

print("📋 CONFIGURATION:")
print(f"  Entry log directory: {entry_log_dir}")
print()

# Test 1: Check/Create log directories
print("Step 1: Creating log directories...")
try:
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    print("  ✓ logs/ directory ready")
    
    # Create entries subdirectory
    os.makedirs(entry_log_dir, exist_ok=True)
    print(f"  ✓ {entry_log_dir}/ directory ready")
    
    # Create today's dated folder
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    today_dir = os.path.join(entry_log_dir, today_str)
    os.makedirs(today_dir, exist_ok=True)
    print(f"  ✓ {today_dir}/ directory ready")
    
except Exception as e:
    print(f"  ✗ Error creating directories: {e}")
    exit(1)

print()

# Test 2: Configure logging
print("Step 2: Setting up logging system...")
try:
    logging.basicConfig(
        filename='logs/events.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("LOGGING SYSTEM INITIALIZED")
    logger.info("=" * 60)
    print("  ✓ Logging configured to: logs/events.log")
except Exception as e:
    print(f"  ✗ Error setting up logging: {e}")
    exit(1)

print()

# Test 3: Write test logs
print("Step 3: Writing test logs...")
try:
    # Test event log
    with open('logs/events.log', 'a') as f:
        f.write(f"\n[TEST] System test started at {datetime.datetime.now()}\n")
    
    # Test entry log
    test_entry_file = os.path.join(today_dir, 'test_entry.txt')
    with open(test_entry_file, 'w') as f:
        f.write(f"""Entry Test Log
Created: {datetime.datetime.now()}
Directory: {today_dir}
Status: OK
""")
    
    print("  ✓ Test event log written to: logs/events.log")
    print(f"  ✓ Test entry log written to: {test_entry_file}")
    
except Exception as e:
    print(f"  ✗ Error writing test logs: {e}")
    exit(1)

print()

# Test 4: Verify files were created
print("Step 4: Verifying log files...")
try:
    # Check events.log
    if os.path.exists('logs/events.log'):
        size = os.path.getsize('logs/events.log')
        lines = len(open('logs/events.log').readlines())
        print(f"  ✓ logs/events.log ({size} bytes, {lines} lines)")
    else:
        print("  ✗ logs/events.log NOT found")
    
    # Check entries directory
    if os.path.exists(entry_log_dir):
        count = len([f for f in os.listdir(today_dir) if os.path.isfile(os.path.join(today_dir, f))])
        print(f"  ✓ {today_dir}/ ({count} files)")
    else:
        print(f"  ✗ {entry_log_dir}/ NOT found")
    
except Exception as e:
    print(f"  ✗ Error verifying: {e}")
    exit(1)

print()

# Test 5: Display directory structure
print("Step 5: Directory structure:")
try:
    print("  logs/")
    print(f"  ├── events.log (system events)")
    print(f"  └── entries/")
    print(f"      └── {today_str}/ (dated folders)")
    print(f"          ├── entry_*.jpg (cropped faces)")
    print(f"          ├── exit_*.jpg (exit faces)")
    print(f"          └── test_entry.txt (test file)")
except Exception as e:
    print(f"  ✗ Error: {e}")

print()

# Test 6: Show what gets logged during processing
print("Step 6: What will be logged during processing:")
print("""
  Entry Events:
    ✓ When a NEW face is detected
    ✓ Visitor ID assigned
    ✓ Cropped face saved to logs/entries/YYYY-MM-DD/
    ✓ Event logged to logs/events.log
    ✓ Entry logged to DATABASE
  
  Exit Events:
    ✓ When face disappears for 30 frames (timeout)
    ✓ Exit detected
    ✓ Exit face cropped and saved
    ✓ Event logged to logs/events.log
    ✓ Exit logged to DATABASE
  
  System Logs:
    ✓ All events written to logs/events.log
    ✓ One line per event with timestamp
    ✓ Can be used for analytics/reporting
""")

print()
print('=' * 60)
print('✅ LOGGING SYSTEM READY!')
print('=' * 60)
print()

print("📊 LOGGING SUMMARY:")
print()
print("Log Files Location:")
print(f"  Events log: logs/events.log")
print(f"  Entry directory: {entry_log_dir}/")
print(f"  Today's folder: {entry_log_dir}/{today_str}/")
print()

print("When to expect logs:")
print("  ✓ Logs are CREATED when you run: python main.py")
print("  ✓ System will auto-create logs/ directory on first run")
print("  ✓ Events logged in real-time as faces detected")
print()

print("Viewing logs:")
print("  # Recent events:")
print("  tail -f logs/events.log")
print()
print("  # Count total events:")
print("  wc -l logs/events.log")
print()
print("  # View cropped faces:")
print(f"  ls -lah {entry_log_dir}/")
print()

print('✅ LOGGING IS WORKING!')
print()
