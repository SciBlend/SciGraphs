# One connection interface over PostgreSQL (psycopg), MySQL and MariaDB
# (mysql-connector-python), SQLite (stdlib) and SQL Server (pymssql).

import os

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    PANDAS_REASON = None
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
    PANDAS_REASON = ("pandas is not installed; install the 'sql' extra to "
                     "enable tabular import and the database backends")
from scigraphs_core.logger import log


# Filled in at import time.
DRIVER_AVAILABLE = {
    'POSTGRESQL': False,
    'MYSQL': False,
    'SQLITE': True,  # Always available in Python
    'SQLSERVER': False,
}

try:
    import psycopg
    DRIVER_AVAILABLE['POSTGRESQL'] = True
except ImportError:
    try:
        import psycopg2 as psycopg
        DRIVER_AVAILABLE['POSTGRESQL'] = True
    except ImportError:
        psycopg = None

try:
    import mysql.connector as mysql_connector
    DRIVER_AVAILABLE['MYSQL'] = True
except ImportError:
    mysql_connector = None

import sqlite3

try:
    import pymssql
    DRIVER_AVAILABLE['SQLSERVER'] = True
except ImportError:
    pymssql = None


def get_available_drivers():
    """Database type to driver availability, as a copy."""
    return DRIVER_AVAILABLE.copy()


def get_driver_status_message():
    """One line naming which database drivers are installed."""
    messages = []
    for db_type, available in DRIVER_AVAILABLE.items():
        status = "available" if available else "not installed"
        messages.append(f"{db_type}: {status}")
    return ", ".join(messages)


def get_password_from_profile(profile):
    """Read a profile's password, from its environment variable if it uses one.

    A missing environment variable yields an empty string, not an error.
    """
    if profile.use_env_password:
        return os.environ.get(profile.env_password_var, "")
    return profile.password


def connect_postgresql(profile):
    """Connect to PostgreSQL. Returns (connection, error_message)."""
    if not DRIVER_AVAILABLE['POSTGRESQL']:
        return None, "PostgreSQL driver not installed. Install with: pip install psycopg[binary]"
    
    password = get_password_from_profile(profile)
    
    conn_params = {
        'host': profile.host,
        'port': profile.port,
        'dbname': profile.database,
        'user': profile.username,
        'password': password,
        'connect_timeout': profile.timeout,
    }
    
    if profile.use_ssl:
        conn_params['sslmode'] = profile.ssl_mode
    
    try:
        if hasattr(psycopg, 'connect'):
            conn = psycopg.connect(**conn_params)
        else:
            conn = psycopg.connect(**conn_params)
        return conn, None
    except Exception as e:
        return None, str(e)


def connect_mysql(profile):
    """Connect to MySQL or MariaDB. Returns (connection, error_message)."""
    if not DRIVER_AVAILABLE['MYSQL']:
        return None, "MySQL driver not installed. Install with: pip install mysql-connector-python"
    
    password = get_password_from_profile(profile)
    
    conn_params = {
        'host': profile.host,
        'port': profile.port,
        'database': profile.database,
        'user': profile.username,
        'password': password,
        'connection_timeout': profile.timeout,
    }
    
    if profile.use_ssl:
        conn_params['ssl_disabled'] = False
    
    try:
        conn = mysql_connector.connect(**conn_params)
        return conn, None
    except Exception as e:
        return None, str(e)


def connect_sqlite(profile):
    """Connect to SQLite. Returns (connection, error_message)."""
    db_path = profile.sqlite_path
    
    if not db_path:
        return None, "No SQLite database file specified"
    
    db_path = os.path.expanduser(db_path)
    
    if not os.path.exists(db_path):
        return None, f"SQLite file not found: {db_path}"
    
    try:
        conn = sqlite3.connect(db_path, timeout=profile.timeout)
        return conn, None
    except Exception as e:
        return None, str(e)


def connect_sqlserver(profile):
    """Connect to SQL Server. Returns (connection, error_message)."""
    if not DRIVER_AVAILABLE['SQLSERVER']:
        return None, "SQL Server driver not installed. Install with: pip install pymssql"
    
    password = get_password_from_profile(profile)
    
    try:
        conn = pymssql.connect(
            server=profile.host,
            port=str(profile.port),
            database=profile.database,
            user=profile.username,
            password=password,
            timeout=profile.timeout,
            login_timeout=profile.timeout,
        )
        return conn, None
    except Exception as e:
        return None, str(e)


def get_connection(profile):
    """Connect using whichever backend the profile names."""
    db_type = profile.db_type
    
    if db_type == 'POSTGRESQL':
        return connect_postgresql(profile)
    elif db_type == 'MYSQL':
        return connect_mysql(profile)
    elif db_type == 'SQLITE':
        return connect_sqlite(profile)
    elif db_type == 'SQLSERVER':
        return connect_sqlserver(profile)
    else:
        return None, f"Unknown database type: {db_type}"


def test_connection(profile):
    """Open a connection, run SELECT 1, close it. Returns (ok, message)."""
    log(f"Testing connection to {profile.db_type}: {profile.name}")
    
    conn, error = get_connection(profile)
    
    if conn is None:
        return False, error
    
    try:
        if profile.db_type == 'SQLITE':
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        elif profile.db_type == 'SQLSERVER':
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        else:  # PostgreSQL and MySQL
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
        
        conn.close()
        return True, f"Connected to {profile.database or profile.sqlite_path}"
    except Exception as e:
        return False, str(e)


def execute_query(profile, sql_query):
    """Run a read-only query. Returns (DataFrame, error_message).

    Anything that is not a bare SELECT is refused, as is any query mentioning a
    write or grant keyword.
    """
    sql_stripped = sql_query.strip().upper()
    if not sql_stripped.startswith('SELECT'):
        return None, "Only SELECT queries are allowed for security reasons"
    
    dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 
                          'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE']
    for keyword in dangerous_keywords:
        if keyword in sql_stripped:
            return None, f"Query contains forbidden keyword: {keyword}"
    
    log(f"Executing query on {profile.db_type}: {profile.name}")
    
    conn, error = get_connection(profile)
    
    if conn is None:
        return None, error
    
    try:
        df = pd.read_sql(sql_query, conn)
        conn.close()
        
        log(f"Query returned {len(df)} rows, {len(df.columns)} columns")
        return df, None
    except Exception as e:
        if conn:
            try:
                conn.close()
            except:
                pass
        return None, str(e)


def get_tables(profile):
    """List table names. Returns (names, error_message)."""
    if profile.db_type == 'POSTGRESQL':
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """
    elif profile.db_type == 'MYSQL':
        query = "SHOW TABLES"
    elif profile.db_type == 'SQLITE':
        query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    elif profile.db_type == 'SQLSERVER':
        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
    else:
        return [], f"Unknown database type: {profile.db_type}"
    
    df, error = execute_query(profile, query)
    
    if error:
        return [], error
    
    if df is not None and len(df) > 0:
        return df.iloc[:, 0].tolist(), None
    
    return [], None


def get_columns(profile, table_name):
    """List a table's columns as {'name', 'type'} dicts.

    Each backend answers in its own shape, so the rows are unpacked per type.
    """
    if profile.db_type == 'POSTGRESQL':
        query = f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """
    elif profile.db_type == 'MYSQL':
        query = f"DESCRIBE {table_name}"
    elif profile.db_type == 'SQLITE':
        query = f"PRAGMA table_info({table_name})"
    elif profile.db_type == 'SQLSERVER':
        query = f"""
            SELECT COLUMN_NAME, DATA_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{table_name}'
        """
    else:
        return [], f"Unknown database type: {profile.db_type}"
    
    df, error = execute_query(profile, query)
    
    if error:
        return [], error
    
    columns = []
    if df is not None and len(df) > 0:
        if profile.db_type == 'SQLITE':
            # PRAGMA gives cid, name, type, notnull, dflt_value, pk.
            for _, row in df.iterrows():
                columns.append({
                    'name': row['name'],
                    'type': row['type'],
                })
        elif profile.db_type == 'MYSQL':
            # DESCRIBE gives Field, Type, Null, Key, Default, Extra.
            for _, row in df.iterrows():
                columns.append({
                    'name': row['Field'],
                    'type': row['Type'],
                })
        else:  # PostgreSQL and SQL Server
            for _, row in df.iterrows():
                columns.append({
                    'name': row.iloc[0],
                    'type': row.iloc[1],
                })
    
    return columns, None

