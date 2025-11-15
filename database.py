import psycopg
import numpy as np
from innit_db import config_to_conninfo  # Reuse the helper (file is named `innit_db.py` in the repo)

def get_db_connection(config):
    """
    Creates and returns a new database connection.
    """
    conninfo = config_to_conninfo(config)
    try:
        conn = psycopg.connect(conninfo)
        return conn
    except psycopg.OperationalError as e:
        print(f"Database connection failed: {e}")
        return None

def register_new_visitor(conn, embedding):
    """
    Adds a new visitor to the Visitors table with their embedding.
    Returns the new visitor_id (UUID).
    """
    # Convert numpy array to list for psycopg
    embedding_list = embedding.tolist()
    
    sql = """
    INSERT INTO Visitors (embedding) 
    VALUES (%s) 
    RETURNING visitor_id;
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (embedding_list,))
            row = cur.fetchone()
            conn.commit()
            # fetchone() returns a tuple like (visitor_id,), so return the id or None
            return row[0] if row else None
    except Exception as e:
        print(f"Error registering new visitor: {e}")
        conn.rollback()
        return None

def find_visitor(conn, embedding, threshold):
    """
    Searches the database for a similar face embedding.
    Uses the HNSW index for fast search.
    Returns (visitor_id, similarity_score) or (None, 0).
    """
    # pgvector's <=> operator calculates cosine distance (0=identical, 2=opposite)
    # We convert it to cosine similarity (1=identical, -1=opposite)
    # Cosine Similarity = 1 - Cosine Distance
    
    embedding_list = embedding.tolist()
    
    sql = """
    SELECT visitor_id, (1 - (embedding <=> %s)) AS similarity
    FROM Visitors
    WHERE (1 - (embedding <=> %s)) >= %s
    ORDER BY similarity DESC
    LIMIT 1;
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (embedding_list, embedding_list, threshold))
            result = cur.fetchone()

            if result:
                # result is (visitor_id, similarity)
                visitor_id, similarity = result
                return visitor_id, similarity
            else:
                return None, 0
    except Exception as e:
        print(f"Error finding visitor: {e}")
        return None, 0

def log_event(conn, visitor_id, event_type, image_path):
    """
    Logs an 'entry' or 'exit' event to the Events table.
    """
    sql = """
    INSERT INTO Events (visitor_id, event_type, cropped_image_path)
    VALUES (%s, %s, %s);
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (visitor_id, event_type, image_path))
            conn.commit()
    except Exception as e:
        print(f"Error logging event: {e}")
        conn.rollback()