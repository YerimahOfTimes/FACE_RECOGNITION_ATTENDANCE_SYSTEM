import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Face Recognition Attendance", layout="wide")

st.title("📊 Face Recognition Attendance Dashboard")

file_path = "attendance.csv"

# Check if file exists
if not os.path.exists(file_path):
    st.warning("No attendance file found yet.")
    st.stop()

# Load data
df = pd.read_csv(file_path)

# Show raw data
st.subheader("📋 Attendance Records")
st.dataframe(df, use_container_width=True)

# =========================
# FILTERS
# =========================
st.subheader("🔎 Filter Records")

col1, col2 = st.columns(2)

with col1:
    name_filter = st.text_input("Filter by Name")

with col2:
    date_filter = st.text_input("Filter by Date (YYYY-MM-DD)")

filtered_df = df.copy()

if name_filter:
    filtered_df = filtered_df[filtered_df["Name"].str.contains(
        name_filter, case=False)]

if date_filter:
    filtered_df = filtered_df[filtered_df["Date"] == date_filter]

st.dataframe(filtered_df, use_container_width=True)

# =========================
# STATS
# =========================
st.subheader("📈 Quick Stats")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Unique People", df["Name"].nunique())

if "Date" in df.columns:
    col3.metric("Days Tracked", df["Date"].nunique())

# =========================
# DOWNLOAD BUTTON
# =========================
st.subheader("⬇️ Export Data")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Attendance CSV",
    csv,
    "attendance.csv",
    "text/csv"
)
