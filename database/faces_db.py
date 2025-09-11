import sqlite3, numpy as np

class FaceDB:
    def __init__(self, db_path="faces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS faces (name TEXT, embedding BLOB)")

    def add_face(self, name, embedding):
        self.conn.execute("INSERT INTO faces VALUES (?, ?)", (name, embedding.tobytes()))
        self.conn.commit()

    def find_match(self, embedding):
        cursor = self.conn.execute("SELECT name, embedding FROM faces")
        for row in cursor:
            db_name, db_embed = row[0], np.frombuffer(row[1], dtype=np.float64)
            similarity = np.dot(embedding, db_embed) / (np.linalg.norm(embedding) * np.linalg.norm(db_embed))
            if similarity > 0.9:
                return db_name
        return None
