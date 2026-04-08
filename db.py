import sqlite3
import os

DB_FILE = "automl_pro.db"

# ---------------- CREATE TABLES ----------------
def create_tables():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    # Datasets table
    c.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            filename TEXT,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    conn.commit()
    conn.close()

# ---------------- USER FUNCTIONS ----------------
def add_user(username, hashed_password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()

def get_hashed_password(username):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

# ---------------- DATASET FUNCTIONS ----------------
def save_dataset(username, filename):
    os.makedirs("datasets", exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO datasets (username, filename) VALUES (?, ?)", (username, filename))
    conn.commit()
    conn.close()

# ---------------- ADMIN FUNCTIONS ----------------
def get_all_users():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, username FROM users")
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_datasets():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, username, filename FROM datasets")
    rows = c.fetchall()
    conn.close()
    return rows

def delete_user(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

def delete_dataset(dataset_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
    conn.commit()
    conn.close()