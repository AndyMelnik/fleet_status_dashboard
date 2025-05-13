import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timezone
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Fleet Status Dashboard")

# Sidebar: DB connection
st.sidebar.title("Database Connection")
hostname = st.sidebar.text_input("Hostname", value="localhost")
database = st.sidebar.text_input("Database Name")
user = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
port = st.sidebar.text_input("Port", value="5432")
connect_button = st.sidebar.button("Connect")

@st.cache_resource
def get_engine(user, password, hostname, port, database):
    engine_url = f'postgresql://{user}:{password}@{hostname}:{port}/{database}'
    return create_engine(engine_url)

# Function to fetch tracking, registered objects, and latest status info
def fetch_data(engine):
    tracking_query = """
        SELECT 
            o.object_label,
            tdc.device_time,
            tdc.latitude / 1e7::decimal AS latitude,
            tdc.longitude / 1e7::decimal AS longitude,
            tdc.speed / 100 AS speed_n,
            tdc.altitude
        FROM 
            raw_telematics_data.tracking_data_core AS tdc
        JOIN 
            raw_business_data.devices AS d ON d.device_id = tdc.device_id
        JOIN 
            raw_business_data.objects AS o ON o.device_id = d.device_id
        WHERE 
            tdc.device_time >= NOW() - INTERVAL '15 minutes'
        ORDER BY 
            tdc.device_time DESC;
    """

    object_query = """
        SELECT DISTINCT 
            object_label
        FROM 
            raw_business_data.objects
        WHERE 
            object_label IS NOT NULL AND is_deleted IS NOT TRUE
        ORDER BY 
            object_label;
    """

    connection_status_query = """
        WITH latest_speeds AS (
            SELECT 
                tdc.device_id,
                tdc.device_time,
                tdc.speed,
                ROW_NUMBER() OVER (PARTITION BY tdc.device_id ORDER BY tdc.device_time DESC) AS rn
            FROM 
                raw_telematics_data.tracking_data_core AS tdc
        )
        SELECT 
            o.object_label,
            ls.speed AS the_latest_speed,
            ls.device_time AS last_device_time
        FROM 
            latest_speeds AS ls
        JOIN 
            raw_business_data.devices AS d ON d.device_id = ls.device_id
        JOIN 
            raw_business_data.objects AS o ON o.device_id = d.device_id
        WHERE 
            ls.rn = 1;
    """

    tracking_df = pd.read_sql(tracking_query, engine)
    object_df = pd.read_sql(object_query, engine)
    status_df = pd.read_sql(connection_status_query, engine)

    tracking_df['device_time'] = pd.to_datetime(tracking_df['device_time'], utc=True)
    status_df['last_device_time'] = pd.to_datetime(status_df['last_device_time'], utc=True)

    return tracking_df, object_df, status_df

# Establish connection on Connect
if connect_button:
    try:
        engine = get_engine(user, password, hostname, port, database)
        st.session_state.engine = engine
        st.session_state.connected = True
        st.success("Connected to the database.")
    except Exception as e:
        st.error(f"Connection failed: {e}")

# UI and logic if connected
if st.session_state.get("connected"):
    engine = st.session_state.engine

    with st.form(key="params_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_idle_speed = st.slider("Max Idle Speed (km/h)", 0, 10, 2)
        with col2:
            min_idle_detection = st.slider("Min Idle Detection (minutes)", 0, 10, 3)
        with col3:
            gps_not_updated_min = st.slider("GPS Not Updated Min (minutes)", 0, 10, 5)
        with col4:
            gps_not_updated_max = st.slider("GPS Not Updated Max (minutes)", gps_not_updated_min, 15, 5)

        update_button = st.form_submit_button("Update")

    if update_button:
        tracking_df, object_df, status_df = fetch_data(engine)

        current_time = datetime.now(timezone.utc)
        valid_object_labels = set(object_df['object_label'])

        tracking_df = tracking_df[tracking_df['object_label'].isin(valid_object_labels)].copy()
        status_df = status_df[status_df['object_label'].isin(valid_object_labels)].copy()

        # Movement status logic
        latest_df = tracking_df.sort_values("device_time", ascending=False).groupby("object_label", as_index=False).first()
        full_latest_df = pd.merge(object_df, latest_df, on="object_label", how="left")

        def classify_movement(row):
            speed = row.get('speed_n', 0)
            time = row.get('device_time')
            if pd.isna(time):
                return 'No Data'
            time_diff = (current_time - time).total_seconds() / 60
            if speed > max_idle_speed:
                return 'Moving'
            elif time_diff < min_idle_detection:
                return 'Stopped'
            else:
                return 'Parked'

        full_latest_df['moving_status'] = full_latest_df.apply(classify_movement, axis=1)

        # Connection status logic
        full_status_df = pd.merge(object_df, status_df, on="object_label", how="left")

        def classify_connection(row):
            time = row.get('last_device_time')
            if pd.isna(time):
                return 'No Signal'
            diff = (current_time - time).total_seconds() / 60
            if diff <= gps_not_updated_min:
                return 'Online'
            elif gps_not_updated_min < diff <= gps_not_updated_max:
                return 'Standby'
            else:
                return 'Offline'

        full_status_df['connection_status'] = full_status_df.apply(classify_connection, axis=1)

        # Merge both together
        merged_df = pd.merge(full_latest_df, full_status_df[['object_label', 'connection_status', 'last_device_time']], on='object_label', how='left')

        # Metrics
        total_objects = object_df['object_label'].nunique()
        moving_count = (merged_df['moving_status'] == 'Moving').sum()
        stopped_count = (merged_df['moving_status'] == 'Stopped').sum()
        parked_count = (merged_df['moving_status'] == 'Parked').sum()
        no_data_count = (merged_df['moving_status'] == 'No Data').sum()

        online_count = (merged_df['connection_status'] == 'Online').sum()
        standby_count = (merged_df['connection_status'] == 'Standby').sum()
        offline_count = (merged_df['connection_status'] == 'Offline').sum()
        no_signal_count = (merged_df['connection_status'] == 'No Signal').sum()

        # Display metrics
        ind1, ind2, ind3, ind4 = st.columns(4)
        ind1.metric("Total Registered Objects", total_objects)
        ind2.metric("Moving", moving_count)
        ind3.metric("Stopped", stopped_count)
        ind4.metric("Parked", parked_count)

        cs1, cs2, cs3, cs4 = st.columns(4)
        cs1.metric("No Signal", no_signal_count)
        cs2.metric("Online", online_count)
        cs3.metric("Standby", standby_count)
        cs4.metric("Offline", offline_count)

        # Charts
        pie1_col, pie2_col = st.columns(2)
        with pie1_col:
            st.plotly_chart(px.pie(merged_df, names='moving_status', title='Movement Status Distribution'))
        with pie2_col:
            st.plotly_chart(px.pie(merged_df, names='connection_status', title='Connection Status Distribution'))

        # Table
        display_df = merged_df[[
            'object_label', 'latitude', 'longitude', 'speed_n', 'device_time',
            'last_device_time', 'connection_status', 'moving_status'
        ]]
        display_df.columns = [
            'Object Label', 'Last Latitude', 'Last Longitude', 'Last Speed',
            'Last Tracking Time', 'Last Device Time', 'Connection Status', 'Moving Status'
        ]
        st.dataframe(display_df, use_container_width=True)
else:
    st.info("Please connect to the database to begin.")
