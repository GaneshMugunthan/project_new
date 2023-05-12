from flask import Flask, render_template, request
import sqlite3

# Connect to the database
# conn = sqlite3.connect('bookings.db')

# Create tables
# c = conn.cursor()
# c.execute('CREATE TABLE rooms (id INTEGER PRIMARY KEY, name TEXT, location TEXT, capacity INTEGER)')
# c.execute('CREATE TABLE bookings (id INTEGER PRIMARY KEY, room_id INTEGER, start_time TEXT, end_time TEXT, organizer TEXT, FOREIGN KEY (room_id) REFERENCES rooms (id))')

# Commit the changes
# conn.commit()

# Close the connection
# conn.close()

# conn = sqlite3.connect('database.db')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/book_room', methods=['POST'])
def book_room():
    room = request.form['room']
    start_time = request.form['start_time']
    end_time = request.form['end_time']
    organizer = request.form['organizer']

    # Insert the booking information into the database
    conn = sqlite3.connect('bookings.db')
    c = conn.cursor()
    c.execute('INSERT INTO bookings (room_id, start_time, end_time, organizer) VALUES (?, ?, ?, ?)', (room, start_time, end_time, organizer))
    conn.commit()
    conn.close()
    print("Booked")
    return 'Booking submitted successfully!'

@app.route('/bookings')
def show_bookings():
    conn = sqlite3.connect('bookings.db')
    c = conn.cursor()
    c.execute('SELECT * FROM bookings')
    bookings = c.fetchall()
    conn.close()
    return render_template('bookings.html', bookings=bookings)



if __name__ == '__main__':
    app.debug=True
    app.run(host='127.0.0.1', port=4000)
