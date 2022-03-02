from flask import Flask,render_template
import sqlite3

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('C:\\SQLLite\\twitter_database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def index():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM Tweets order by created_at DESC').fetchall()
    conn.close()
    return render_template('list.html', posts=posts)
    

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)