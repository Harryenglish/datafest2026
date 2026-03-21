import streamlit as st
import pandas as pd
import pydeck as pdk
import re
import time

st.set_page_config(layout="wide", page_title="Kansas Diagnosis 2022-2025")

# 1. FINAL 10 CITIES
KANSAS_GEO = {
    'junction city': [39.0286, -96.8314], 'manhattan': [39.1836, -96.5717],
    'wamego': [39.2025, -96.3050], 'emporia': [38.4039, -96.1817],
    'osage city': [38.6339, -95.8261], 'carbondale': [38.8236, -95.6883],
    'topeka': [39.0473, -95.6752], 'meriden': [39.1917, -95.5683],
    'oskaloosa': [39.2158, -95.3130], 'lawrence': [38.9717, -95.2353]
}

PILLAR_COLOR = [0, 255, 255, 200]

def get_chapter(code):
    c = str(code).upper()
    if "IMO" in c or "NAN" in c or c == "-1": return "Other/Unspecified"
    match = re.search(r'([A-Z])(\d{2})', c)
    if not match: return "Other/Misc"
    l, n = match.group(1), int(match.group(2))
    mapping = {'A': 'Infectious', 'B': 'Infectious', 'E': 'Endocrine', 'F': 'Mental', 
               'G': 'Nervous', 'I': 'Circulatory', 'J': 'Respiratory', 'K': 'Digestive', 
               'M': 'Musculoskeletal', 'N': 'Genitourinary', 'Z': 'Health Status'}
    if l in mapping: return mapping[l]
    if l == 'C' or (l == 'D' and n <= 49): return "Neoplasms"
    if l == 'D' and n >= 50: return "Blood/Immune"
    if l in 'ST': return "Injury"
    return "Other/Misc"

@st.cache_data
def load_and_preprocess():
    # Load only necessary columns to save RAM
    diag = pd.read_csv('diagnosis.csv', usecols=['DiagnosisKey', 'GroupCode'], dtype=str)
    link = pd.read_csv('encounters.csv', usecols=['Date', 'DepartmentKey', 'PrimaryDiagnosisKey'], dtype=str)
    dept = pd.read_csv('departments.csv', usecols=['DepartmentKey', 'city'], dtype=str)

    # Convert dates once
    link['Date'] = pd.to_datetime(link['Date'], errors='coerce', format='mixed')
    link = link.dropna(subset=['Date'])
    link = link[(link['Date'].dt.year >= 2022) & (link['Date'].dt.year <= 2025)]
    
    # Create Year-Month string for easy grouping (e.g., "2022-01")
    link['YearMonth'] = link['Date'].dt.strftime('%Y-%m')

    # Merge and filter for our 10 cities
    df = link.merge(dept, on='DepartmentKey', how='inner')
    df['city_clean'] = df['city'].astype(str).str.lower().str.strip()
    df = df[df['city_clean'].isin(KANSAS_GEO.keys())]
    
    df = df.merge(diag, left_on='PrimaryDiagnosisKey', right_on='DiagnosisKey', how='left')
    df['Chapter'] = df['GroupCode'].apply(get_chapter)
    
    # PRE-AGGREGATE: This is the secret to speed. 
    # We turn millions of rows into a few thousand rows.
    monthly_summary = df.groupby(['YearMonth', 'city_clean', 'Chapter']).size().reset_index(name='count')
    full_summary = df.groupby(['city_clean', 'Chapter']).size().reset_index(name='count')
    
    return monthly_summary, full_summary

try:
    monthly_data, full_data = load_and_preprocess()
    all_months = sorted(monthly_data['YearMonth'].unique())

    st.sidebar.header("Navigation")
    view_mode = st.sidebar.radio("Select View:", ["Full Period Summary (2022-2025)", "Monthly Timelapse"])
    
    scale = st.sidebar.slider("Pillar Height Multiplier", 1, 5000, 1500)

    if view_mode == "MonthlyTimelapse":
        # ... (Timelapse logic below)
        pass 

    # --- MODE 1: FULL SUMMARY ---
    if view_mode == "Full Period Summary (2022-2025)":
        st.title("Kansas Medical Distribution: 2022-2025 Total")
        
        viz_list = []
        for city, coords in KANSAS_GEO.items():
            city_stats = full_data[full_data['city_clean'] == city]
            total = city_stats['count'].sum()
            table_html = ""
            if total > 0:
                top_5 = city_stats.sort_values(by='count', ascending=False).head(5)
                for _, r in top_5.iterrows():
                    perc = (r['count']/total)*100
                    table_html += f"<div style='display:flex; justify-content:space-between;'><span>{r['Chapter']}</span><b>{perc:.1f}%</b></div>"
            
            viz_list.append({"city": city.title(), "lat": coords[0], "lon": coords[1], "total": int(total), "table_html": table_html})

        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer("ColumnLayer", data=pd.DataFrame(viz_list), get_position="[lon, lat]", get_elevation="total", elevation_scale=scale/10, get_fill_color=PILLAR_COLOR, radius=3500, pickable=True)],
            initial_view_state=pdk.ViewState(latitude=38.8, longitude=-96.0, zoom=7, pitch=45),
            tooltip={"html": "<b>{city}</b><br>Total Records: {total}<hr>{table_html}"}
        ))

    # --- MODE 2: TIMELAPSE ---
    else:
        selected_month = st.sidebar.select_slider("Manual Month", options=all_months)
        play = st.sidebar.button("▶ Start 120s Timelapse")

        # Container for the moving pillars
        placeholder = st.empty()
        
        # Determine which months to show (one or all in sequence)
        loop_months = all_months if play else [selected_month]
        wait_time = 120.0 / len(all_months) if play else 0

        for m in loop_months:
            m_df = monthly_data[monthly_data['YearMonth'] == m]
            viz_list = []
            for city, coords in KANSAS_GEO.items():
                city_total = m_df[m_df['city_clean'] == city]['count'].sum()
                viz_list.append({"city": city.title(), "lat": coords[0], "lon": coords[1], "total": int(city_total)})
            
            with placeholder.container():
                st.subheader(f"Data for: {m}")
                st.pydeck_chart(pdk.Deck(
                    layers=[pdk.Layer("ColumnLayer", data=pd.DataFrame(viz_list), get_position="[lon, lat]", 
                                      get_elevation="total", elevation_scale=scale, 
                                      get_fill_color=PILLAR_COLOR, radius=3500, pickable=True)],
                    initial_view_state=pdk.ViewState(latitude=38.8, longitude=-96.0, zoom=7, pitch=45),
                    tooltip={"text": "{city}\nMonthly Records: {total}"}
                ))
            if play:
                time.sleep(wait_time)

except Exception as e:
    st.error(f"Error: {e}")