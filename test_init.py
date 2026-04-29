import sys
sys.path.insert(0, 'backend')
try:
    from database import init_db
    init_db()
    print('SUCCESS')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'ERROR: {e}')
