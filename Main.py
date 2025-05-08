import streamlit as st
import mysql.connector
import cv2
import numpy as np
from skimage.filters import threshold_local
import tensorflow as tf
import pytesseract
import re
import time
import datetime

# Configure Tesseract OCR Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Establish Database Connection
def get_db_connection():
    return mysql.connector.connect(user='root', password='', host='localhost', database='5numberfacedb')

# Load TensorFlow Model
class NeuralNetwork:
    def _init_(self):
        self.model_file = "./model/binary_128_0.50_ver3.pb"
        self.label_file = "./model/binary_128_0.50_labels_ver2.txt"
        self.label = self.load_label(self.label_file)
        self.graph = self.load_graph(self.model_file)
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def load_graph(self, modelFile):
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with open(modelFile, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
        return graph

    def load_label(self, labelFile):
        label = []
        proto_as_ascii_lines = tf.io.gfile.GFile(labelFile).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

# License Plate Detection Functions
def segment_chars(plate_img, fixed_width=400):
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 29, offset=15, method='gaussian')
    thresh = (V > T).astype('uint8') * 255
    charCandidates = np.zeros(thresh.shape, dtype='uint8')
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 14 and h > 20:
            hull = cv2.convexHull(c)
            cv2.drawContours(charCandidates, [hull], -1, 255, -1)
    return charCandidates

# Streamlit App
st.title("Vehicle License Plate Recognition")

menu = ["Home", "Admin Login", "User Login", "New User"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Welcome to the Vehicle License Plate Recognition System")

elif choice == "Admin Login":
    username = st.text_input("Username", key="admin_username")
    password = st.text_input("Password", type="password", key="admin_password")
    if st.button("Login", key="admin_login_button"):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM regtb WHERE username=%s AND password=%s", (username, password))
        data = cursor.fetchone()
        conn.close()
        if data:
            st.success("Welcome, Admin!")
        else:
            st.error("Invalid Username or Password")

elif choice == "User Login":
    username = st.text_input("Username", key="user_username")
    password = st.text_input("Password", type="password", key="user_password")
    if st.button("Login", key="user_login_button"):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM regtb WHERE username=%s AND password=%s", (username, password))
        data = cursor.fetchone()
        conn.close()
        if data:
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid Username or Password")

elif choice == "New User":
    with st.form("user_form"):
        vno = st.text_input("Vehicle Number", key="new_user_vno")
        name = st.text_input("Name", key="new_user_name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="new_user_gender")
        age = st.number_input("Age", min_value=1, max_value=120, key="new_user_age")
        email = st.text_input("Email", key="new_user_email")
        pnumber = st.text_input("Phone Number", key="new_user_phone")
        address = st.text_area("Address", key="new_user_address")
        uname = st.text_input("Username", key="new_user_uname")
        password = st.text_input("Password", type="password", key="new_user_password")
        submitted = st.form_submit_button("Register", key="new_user_submit")

    if submitted:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO regtb (VehicleNo, Name, Gender, Age, Email, PhoneNumber, Address, Username, Password) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (vno, name, gender, age, email, pnumber, address, uname, password)
        )
        conn.commit()
        conn.close()
        st.success("User Registered Successfully!")

if st.button("Start Plate Detection", key="start_detection_button"):
    st.write("Starting Webcam...")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            st.image(frame, channels="BGR")
            plate_text = pytesseract.image_to_string(frame, config='--psm 6')
            st.write(f"Detected Plate: {plate_text}")
        if st.button("Stop", key="stop_detection_button"):
            break
    cap.release()