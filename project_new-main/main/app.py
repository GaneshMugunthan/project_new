from flask import Flask, render_template, request
import sqlite3
import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib



#------------------------------
#TABLE CREATION
# # Connect to the database
# conn = sqlite3.connect('bookings.db')

# # Create tables
# c = conn.cursor()
# c.execute('CREATE TABLE rooms (id INTEGER PRIMARY KEY, name TEXT, location TEXT, capacity INTEGER)')
# c.execute('CREATE TABLE bookings (id INTEGER PRIMARY KEY, room_id INTEGER, start_time TEXT, end_time TEXT, organizer TEXT, FOREIGN KEY (room_id) REFERENCES rooms (id))')
# conn.commit()
# conn.close()
# conn = sqlite3.connect('database.db')
#---------------------------------

app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#--------------------------Index page----------------------------------
@app.route('/')
def main():
    return render_template('index.html')

#--------------------------login/register/student detail---------------
# Create a connection to the SQLite database
conn = sqlite3.connect('database.db')
c = conn.cursor()

# Create a table to store user registration data
c.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fullName TEXT NOT NULL,
                userName TEXT NOT NULL,
                email TEXT NOT NULL,
                password TEXT NOT NULL,
                mobileNumber INTEGER NOT NULL,
                address TEXT NOT NULL,
                state TEXT NOT NULL,
                country TEXT NOT NULL,
                alternativeNumber INTEGER NOT NULL,
                role TEXT NOT NULL,
                year TEXT,
                class TEXT,
                department TEXT,
                designation TEXT,
                age INTEGER NOT NULL,
                dob DATE NOT NULL
            )''')
conn.commit()
# Define a function to insert user data into the database
def insert_user(full_name, username, email, password, mobile_number, address, state, country, alternative_number, role, year=None, class_=None, department=None, designation=None, age=None, dob=None):
    c.execute('''INSERT INTO users (fullName, userName, email, password, mobileNumber, address, state, country, alternativeNumber, role, year, class, department, designation, age, dob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (full_name, username, email, password, mobile_number, address, state, country, alternative_number, role, year, class_, department, designation, age, dob))
    conn.commit()

@app.route('/login_page')
def login_page():
    return render_template('login.html')

@app.route('/login',methods=['GET','POST'])
def login():
    
    
    pass

@app.route('/register_page')
def register_page():
    return render_template('register.html')

@app.route('/adduser')
def register():
    if request.method == 'POST':
        full_name = request.form.get('fullName')
        username = request.form.get('userName')
        email = request.form.get('email')
        password = request.form.get('password')
        mobile_number = request.form.get('mobileNumber')
        address = request.form.get('address')
        state = request.form.get('state')
        country = request.form.get('country')
        alternative_number = request.form.get('alternativeNumber')
        role = request.form.get('role')
        year = request.form.get('year') if role == 'student' else None
        class_ = request.form.get('class') if role == 'student' else None
        department = request.form.get('department') if role == 'student' else None
        designation = request.form.get('designation') if role == 'teacher' else None
        age = request.form.get('age')
        dob = request.form.get('dob')
        
        insert_user(full_name, username, email, password, mobile_number, address, state, country, alternative_number, role, year, class_, department, designation, age, dob)
        
        return render_template('main.html')
    return render_template('register.html')
    

#--------------------------Camera operation----------------------------
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    if img!=[]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')



#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#-------------------------------------------------------


#### Our main page
@app.route('/a')
def home1():
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            add_attendance(identified_person)
            cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#--------------------------Booking operation --------------------------
@app.route('/book_room', methods=['POST'])
def book_room():
    room = request.form['room']
    start_time = request.form['start_time']
    end_time = request.form['end_time']
    organizer = request.form['organizer']

    # Insert the booking information into the database
    conn = sqlite3.connect('bookings.db')
    
    
    # Create tablesflask 
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

#--------------------------Main----------------------------------------
#--------------------------admin page----------------------------------
#--------------------------teacher/student page------------------------



#-------------------------------------

   
if __name__ == '__main__':
    app.debug=True
    app.run(host='127.0.0.1', port=4000)