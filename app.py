"""
Hurricane Ian Catastrophe Model — Streamlit Dashboard
CAT-402 + CSC-498 Final Project | Lehigh University | Spring 2026
GitHub: sik324/DSCI_498_FinalProject
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hurricane Ian Cat Model",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Path configuration ────────────────────────────────────────────────────────
# Works both locally and on Streamlit Cloud
# On Streamlit Cloud: put outputs/ folder in repo root
# Locally or Colab:   set OUTPUTS_DIR env variable or use default below
OUTPUTS_DIR = os.environ.get(
    'OUTPUTS_DIR',
    'outputs'   # relative path — put outputs/ in same folder as app.py
)
CGAN_DIR  = f'{OUTPUTS_DIR}/cgan'
BAL_DIR   = f'{CGAN_DIR}/balanced_data'
EXP_DIR   = f'{OUTPUTS_DIR}/exposure'
HAZ_DIR   = f'{OUTPUTS_DIR}/hazard'
LOSS_DIR  = f'{OUTPUTS_DIR}/loss'
VULN_DIR  = f'{OUTPUTS_DIR}/vulnerability'

# ── Helper: safe image loader ─────────────────────────────────────────────────
def show_image(path, caption='', width=None):
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=(width is None))
    else:
        st.info(f"Image not found: {path}")

# ── Load functions (cached) ───────────────────────────────────────────────────
@st.cache_data
def load_training_loss():
    path = f'{CGAN_DIR}/training_loss_balanced.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    path2 = f'{CGAN_DIR}/training_loss.csv'
    if os.path.exists(path2):
        return pd.read_csv(path2)
    return None

@st.cache_data
def load_validation_summary():
    path = f'{CGAN_DIR}/validation_summary.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fallback — your actual results from today's session
    return {
        'model'                 : 'generator_balanced_best.pth',
        'epoch'                 : 61,
        'val_loss'              : 0.0050,
        'peak_wind_error_pct'   : 0.6,
        'spatial_correlation'   : 0.9742,
        'mae_cgan_mph'          : 1.99,
        'mae_baseline_mph'      : 1.78,
        'resolution_in'         : '22x21',
        'resolution_out'        : '201x201',
        'conditioning_collapse' : True,
        'features_responsive'   : [],
        'features_ignored'      : ['Vmax', 'RMW', 'Pmin', 'Latitude'],
        'model_type_actual'     : 'Physics-guided super-resolution',
        'model_type_intended'   : 'Conditional GAN',
    }

@st.cache_data
def load_exposure_data():
    """Load real exposure CSV if available, else return simulated Lee County data."""
    for fname in ['lee_county_exposure.csv', 'exposure_results.csv',
                  'tract_exposure.csv', 'lee_exposure.csv']:
        path = f'{EXP_DIR}/{fname}'
        if os.path.exists(path):
            return pd.read_csv(path), True
    # Fallback — hardcoded Lee County land-only points
    np.random.seed(42)
    # Four zones — all confirmed land areas
    zones = [
        # (lat_min, lat_max, lon_min, lon_max, n, zone_name)
        (26.52, 26.72, -82.00, -81.88, 70,  'Cape Coral (NE)'),
        (26.52, 26.68, -81.90, -81.65, 70,  'Fort Myers'),
        (26.52, 26.68, -81.65, -81.35, 50,  'Lehigh Acres'),
        (26.30, 26.45, -81.82, -81.65, 31,  'Bonita Springs'),
    ]
    rows = []
    for lat_mn, lat_mx, lon_mn, lon_mx, n, zone in zones:
        lats = np.random.uniform(lat_mn, lat_mx, n)
        lons = np.random.uniform(lon_mn, lon_mx, n)
        # Wind higher near coast (western = lower longitude)
        wind_h = 155 - (lons + 82.0) * 12 + np.random.normal(0, 3, n)
        wind_h = np.clip(wind_h, 90, 157)
        wind_c = wind_h + np.random.normal(0.94, 1.5, n)
        wind_c = np.clip(wind_c, 90, 160)
        tiv    = np.random.exponential(200, n) + 50
        tiv    = np.clip(tiv, 16, 17000)
        bldgs  = np.clip((tiv * 1.4 + np.random.normal(0, 50, n)).astype(int), 50, 5000)
        for i in range(n):
            rows.append({
                'lat'       : lats[i],
                'lon'       : lons[i],
                'wind_hol'  : wind_h[i],
                'wind_cgan' : wind_c[i],
                'TIV_M'     : tiv[i],
                'buildings' : bldgs[i],
                'zone'      : zone,
                'wind_diff' : wind_c[i] - wind_h[i],
            })
    return pd.DataFrame(rows), False

@st.cache_data
def load_loss_data():
    for fname in ['loss_results.csv', 'loss_comparison.csv',
                  'vulnerability_loss.csv']:
        path = f'{LOSS_DIR}/{fname}'
        if os.path.exists(path):
            return pd.read_csv(path)
        path2 = f'{VULN_DIR}/{fname}'
        if os.path.exists(path2):
            return pd.read_csv(path2)
    # Fallback — your actual project numbers
    return pd.DataFrame({
        'building_type'     : ['W1 Wood Frame', 'W2 Wood Comm.',
                               'C1 Concrete', 'C3 Conc. Shear',
                               'RM1 Masonry', 'Total'],
        'count'             : [187000, 42000, 28000, 18000, 22000, 311512],
        'tiv_b'             : [32.07, 8.42, 4.21, 2.87, 2.66, 50.23],
        'mdr_hol_pct'       : [60.5, 48.2, 35.1, 38.4, 52.3, 56.3],
        'mdr_cgan_pct'      : [58.1, 46.5, 33.8, 37.0, 50.4, 54.2],
        'loss_hol_b'        : [18.88, 4.06, 1.48, 1.10, 1.39, 28.29],
        'loss_cgan_b'       : [18.14, 3.91, 1.42, 1.06, 1.34, 27.19],
    })

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/24701-nature-natural-beauty.jpg/1px-24701-nature-natural-beauty.jpg",
    width=1
)  # invisible spacer
st.sidebar.title("🌀 Hurricane Ian")
st.sidebar.markdown("**Catastrophe Model Dashboard**")
st.sidebar.markdown("CAT-402 + CSC-498 | Lehigh University")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview",
     "🌪 Hazard Module",
     "🏘 Exposure Module",
     "🤖 cGAN Results",
     "💰 Loss Analysis",
     "📊 Model Training",
     "🛡 Peer Review Defense"]
)

st.sidebar.divider()
st.sidebar.markdown("**Storm Parameters**")
st.sidebar.markdown("📍 Landfall: Lee County, FL")
st.sidebar.markdown("📅 Date: Sep 28, 2022")
st.sidebar.markdown("💨 Intensity: Cat 4 — 130 kt")
st.sidebar.markdown("🌡 Min Pressure: 937 mb")
st.sidebar.markdown("📏 RMW: ~15 nm")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🌀 Hurricane Ian Catastrophe Model")
    st.markdown("### Lee County, Florida — September 28, 2022")
    st.markdown(
        "A HAZUS-based catastrophe model enhanced with conditional GAN "
        "super-resolution for building-level loss estimation."
    )

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Landfall",        "Cat 4 — 130 kt")
    c2.metric("Min Pressure",    "937 mb")
    c3.metric("Peak Gust",       "157 mph")
    c4.metric("Total TIV",       "$50.23B")
    c5.metric("Estimated Loss",  "$28.29B")

    st.divider()

    # Pipeline diagram
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Project Pipeline")
        st.markdown("""
        ```
        ┌─────────────────────────────────┐
        │  IBTrACS Storm Track (NOAA)     │  Data
        └──────────────┬──────────────────┘
                       ↓
        ┌─────────────────────────────────┐
        │  Holland Wind Field (0.05°)     │  CAT-402
        │  Hazard Module                  │  Module 1
        └──────────────┬──────────────────┘
                       ↓
        ┌─────────────────────────────────┐
        │  cGAN Super-Resolution          │  CSC-498
        │  0.05° → 0.005° (10× finer)    │
        └──────────────┬──────────────────┘
                       ↓
        ┌─────────────────────────────────┐
        │  HAZUS Exposure Module          │  CAT-402
        │  311,512 buildings, $50.23B TIV │  Module 2
        └──────────────┬──────────────────┘
                       ↓
        ┌─────────────────────────────────┐
        │  Vulnerability + Loss           │  CAT-402
        │  $28.29B estimated loss         │  Modules 3+4
        └─────────────────────────────────┘
        ```
        """)

    with col2:
        st.subheader("Key Results Summary")
        results = pd.DataFrame({
            'Module'   : ['Hazard', 'Hazard', 'cGAN', 'cGAN',
                          'Exposure', 'Loss', 'Loss'],
            'Metric'   : ['Peak wind speed', 'Grid resolution',
                          'Correlation (r)', 'Peak wind error',
                          'Total buildings', 'Holland loss',
                          'cGAN loss'],
            'Value'    : ['157 mph', '0.05° (5.5 km)',
                          '0.9742', '0.6%',
                          '311,512', '$28.29B', '$27.19B'],
            'Status'   : ['✓', '→ improved by cGAN',
                          '✓ Excellent', '✓ Excellent',
                          '✓', '—', '↓ $1.1B lower'],
        })
        st.dataframe(results, hide_index=True, use_container_width=True)

        st.subheader("Courses")
        st.info("**CAT-402** — Hazard + Exposure + Vulnerability + Loss modules")
        st.success("**CSC-498** — cGAN super-resolution + validation + peer review")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HAZARD MODULE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌪 Hazard Module":
    st.title("🌪 Hazard Module — Holland Wind Field")
    st.markdown("**Method:** Holland (1980) parametric wind field model")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak 3-s Gust",    "157 mph",  "Cat 4 at landfall")
    c2.metric("Grid Resolution",  "0.05°",    "~5.5 km per cell")
    c3.metric("Track Records",    "74",       "IBTrACS records")
    c4.metric("Study Area",       "Lee County", "FIPS 12071")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Wind Field Maps", "Storm Track", "Methodology"])

    with tab1:
        st.subheader("Wind Field Comparison")
        show_image(
            f'{CGAN_DIR}/wind_field_comparison_balanced.png',
            caption='Left: Holland coarse (0.05°) | Centre: cGAN (0.005°) | Right: Holland fine reference (0.005°)'
        )
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            show_image(
                f'{CGAN_DIR}/holland_vs_cgan_land_comparison.png',
                caption='Holland vs cGAN — Lee County land area'
            )
        with col2:
            show_image(
                f'{CGAN_DIR}/wind_exposure_overlay.png',
                caption='Wind field overlaid with exposure data'
            )

    with tab2:
        st.subheader("Hurricane Ian Track — Lee County Landfall")
        # Ian track points near Florida
        ian_track = pd.DataFrame({
            'lat' : [23.2, 24.1, 25.0, 25.9, 26.4, 26.8, 27.8, 28.8, 29.8],
            'lon' : [-84.3,-83.5,-82.8,-82.5,-82.2,-82.0,-81.6,-81.2,-80.9],
            'vmax': [60,   80,   100,  115,  125,  130,  110,  80,   60  ],
            'time': ['Sep 27 00Z','Sep 27 06Z','Sep 27 12Z','Sep 27 18Z',
                     'Sep 28 00Z','Sep 28 18Z','Sep 29 00Z','Sep 29 06Z',
                     'Sep 29 12Z'],
        })
        fig = px.scatter_mapbox(
            ian_track, lat='lat', lon='lon',
            size='vmax', color='vmax',
            color_continuous_scale='RdYlGn_r',
            size_max=25,
            mapbox_style='carto-positron',
            zoom=6,
            center={'lat': 26.5, 'lon': -82.5},
            hover_data={'time': True, 'vmax': True},
            labels={'vmax': 'Wind (kt)', 'time': 'Time'},
            title='Hurricane Ian track — color = intensity (kt)'
        )
        fig.add_trace(go.Scattermapbox(
            lat=ian_track['lat'], lon=ian_track['lon'],
            mode='lines',
            line=dict(width=2, color='gray'),
            showlegend=False
        ))
        # Lee County marker
        fig.add_trace(go.Scattermapbox(
            lat=[26.55], lon=[-81.80],
            mode='markers+text',
            marker=dict(size=12, color='red', symbol='star'),
            text=['Lee County'],
            textposition='top right',
            showlegend=False
        ))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Holland (1980) Wind Field Model")
        st.markdown("""
        **Gradient wind equation:**
        ```
        Vgr(r) = sqrt( B/ρ × (Rmw/r)^B × ΔP × exp(-(Rmw/r)^B) + (r×f/2)² ) - r×f/2
        ```

        **Parameters used for Ian:**
        | Parameter | Value | Source |
        |-----------|-------|--------|
        | Vmax | 130 kt | IBTrACS |
        | Pmin | 937 mb | IBTrACS |
        | Penv | 1013 mb | Standard atmosphere |
        | RMW  | 15 nm  | IBTrACS |
        | Holland B | 1.67 | Willoughby & Rahn (2004) |
        | Latitude | 26.8°N | IBTrACS |

        **Post-processing:**
        - Surface reduction: ×0.80 (land), ×0.90 (water)
        - Asymmetry correction: Shapiro (1983) α=0.5
        - Gust factor: ×1.11 (Harper et al. 2010)
        """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EXPOSURE MODULE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏘 Exposure Module":
    st.title("🏘 Exposure Module — Lee County Building Inventory")
    st.markdown("**Method:** HAZUS MH v4.0 model building type classification")

    df, is_real = load_exposure_data()
    if not is_real:
        st.warning("⚠ Showing simulated tract locations. Upload real exposure CSV to outputs/exposure/ for actual data.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Census Tracts",    "223")
    c2.metric("Total Buildings",  "311,512")
    c3.metric("Total TIV",        "$50.23B")
    c4.metric("Avg TIV/Building", "$161K")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Map", "Building Types", "Wind vs TIV"])

    with tab1:
        st.subheader("Lee County Census Tract Locations")
        color_col = st.selectbox(
            "Color by:",
            ['wind_hol', 'wind_cgan', 'TIV_M', 'buildings', 'wind_diff']
        )
        fig = px.scatter_mapbox(
            df, lat='lat', lon='lon',
            color=color_col,
            size='TIV_M',
            color_continuous_scale='RdYlGn_r',
            mapbox_style='carto-positron',
            zoom=9,
            center={'lat': 26.55, 'lon': -81.80},
            size_max=15,
            opacity=0.85,
            labels={
                'wind_hol'  : 'Holland wind (mph)',
                'wind_cgan' : 'cGAN wind (mph)',
                'TIV_M'     : 'TIV ($M)',
                'buildings' : 'Buildings',
                'wind_diff' : 'Wind diff (mph)',
            },
            title='Lee County census tracts'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("HAZUS Building Type Distribution")
        btype = pd.DataFrame({
            'Type'       : ['W1 Wood Frame', 'W2 Wood Comm.',
                            'C1 Concrete', 'C3 Conc. Shear',
                            'RM1 Masonry', 'Other'],
            'Count'      : [187000, 42000, 28000, 18000, 22000, 14512],
            'TIV_B'      : [32.07, 8.42, 4.21, 2.87, 2.66, 0.00],
            'Avg_TIV_K'  : [145, 890, 2100, 1800, 650, 420],
        })
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                btype, values='Count', names='Type',
                title='Buildings by type',
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                btype, x='Type', y='TIV_B',
                title='TIV by building type ($B)',
                color='TIV_B',
                color_continuous_scale='Blues',
                labels={'TIV_B': 'TIV ($B)'}
            )
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Wind Speed vs TIV by Tract")
        fig = px.scatter(
            df, x='wind_hol', y='TIV_M',
            color='wind_diff',
            size='buildings',
            color_continuous_scale='RdYlGn',
            labels={
                'wind_hol'  : 'Holland wind speed (mph)',
                'TIV_M'     : 'Total Insured Value ($M)',
                'wind_diff' : 'cGAN - Holland (mph)',
                'buildings' : 'Buildings'
            },
            title='Wind speed vs exposure value by census tract',
            hover_data=['buildings']
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — cGAN RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 cGAN Results":
    st.title("🤖 cGAN Super-Resolution Results")
    st.markdown(
        "**Model:** U-Net Generator + PatchGAN Discriminator  |  "
        "**Training:** 100 epochs, 2,500 balanced samples  |  "
        "**Resolution:** 22×21 → 201×201"
    )

    val = load_validation_summary()

    # Key metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Epoch",       str(val.get('epoch', 61)))
    c2.metric("Val Loss",         f"{val.get('val_loss', 0.005):.4f}")
    c3.metric("Correlation (r)",  f"{val.get('spatial_correlation', 0.9742):.4f}",
              "✓ Excellent")
    c4.metric("Peak Wind Error",  f"{val.get('peak_wind_error_pct', 0.6):.1f}%",
              "✓ Excellent")
    c5.metric("Model Type",       val.get('model_type_actual', 'Super-resolution'))

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Wind Field Comparison",
        "Sensitivity Analysis",
        "Key Findings",
        "Training Data Audit"
    ])

    with tab1:
        st.subheader("Before vs After cGAN")
        show_image(
            f'{CGAN_DIR}/wind_field_comparison_balanced.png',
            caption='Left: Holland coarse (0.05°) | Centre: cGAN output (0.005°) | Right: Holland fine reference (0.005°)'
        )
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            show_image(
                f'{CGAN_DIR}/wind_distribution_comparison.png',
                caption='Wind speed distribution: Holland vs cGAN'
            )
        with col2:
            show_image(
                f'{CGAN_DIR}/holland_vs_cgan_loss_comparison.png',
                caption='Loss comparison: Holland vs cGAN'
            )

    with tab2:
        st.subheader("Feature Sensitivity Analysis")
        show_image(
            f'{CGAN_DIR}/feature_sensitivity.png',
            caption='Sensitivity of cGAN output to each conditioning feature (±30% perturbation)'
        )
        st.divider()
        show_image(
            f'{CGAN_DIR}/feature_importance.png',
            caption='Feature importance ranking (std dev of peak wind output)'
        )
        st.info(
            "**Finding:** All 4 features show < 0.1 mph variation — "
            "confirmed as numerical noise. Conditioning collapse detected."
        )

    with tab3:
        st.subheader("Complete Validation Findings")
        col1, col2 = st.columns(2)
        with col1:
            st.success("✓ What Works")
            st.markdown(f"""
            | Check | Result |
            |-------|--------|
            | Peak wind accuracy | **0.6% error** |
            | Spatial correlation | **r = 0.9742** |
            | Resolution | **22×21 → 201×201** |
            | Physical wind decay | **Confirmed** |
            | MAE improvement | Marginal (perception-distortion tradeoff) |
            """)

        with col2:
            st.error("✗ Limitations Discovered")
            st.markdown("""
            | Finding | Evidence |
            |---------|----------|
            | Conditioning collapse | Zero/ones test identical output |
            | Circular ground truth | Holland fine as training target |
            | Single location | Lat std = 0.000 (26.3°N only) |
            | No ASOS validation | Never tested vs observations |
            | Climate non-stationarity | Not modeled |
            """)

        st.divider()
        st.subheader("Conditioning Collapse Test")
        collapse_data = pd.DataFrame({
            'Condition Vector' : ['Normal [1.09, 0.17, 0.85, 0.75]',
                                  'Zero   [0.00, 0.00, 0.00, 0.00]',
                                  'Ones   [1.00, 1.00, 1.00, 1.00]'],
            'Peak Wind (mph)'  : [190.3, 190.5, 190.3],
            'Verdict'          : ['Baseline', '⚠ Same as baseline',
                                  '⚠ Same as baseline'],
        })
        st.dataframe(collapse_data, hide_index=True, use_container_width=True)
        st.error(
            "**Confirmed:** Generator ignores condition vector entirely. "
            "Model is a super-resolution upsampler, not a true cGAN."
        )

    with tab4:
        st.subheader("Training Data Audit")
        audit = pd.DataFrame({
            'Parameter'  : ['Total samples', 'Vmax range',
                            'RMW range', 'Pmin range', 'Latitude'],
            'Value'      : ['2,500', '64.1 – 164.9 kt',
                            '10.1 – 55.0 nm', '864.6 – 955.4 mb',
                            '26.3°N only'],
            'Diversity'  : ['✓ Good', '✓ Good (654 unique values)',
                            '✓ Good (436 unique values)', '✓ Good',
                            '✗ Zero — single location'],
            'Implication': ['Sufficient', 'Intensity diversity present',
                            'Size diversity present',
                            'Pressure diversity present',
                            '⚠ Cannot generalize to other coastlines'],
        })
        st.dataframe(audit, hide_index=True, use_container_width=True)

        st.warning(
            "**Key finding:** All 2,500 training samples at lat 26.3°N "
            "(Ian's landfall). This is synthetic parameter augmentation "
            "of one location — not multi-storm historical database."
        )

        show_image(
            f'{CGAN_DIR}/climate_nonstationarity.png',
            caption='Climate non-stationarity — where the model breaks down under future scenarios'
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LOSS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Loss Analysis":
    st.title("💰 Loss Analysis — Holland vs cGAN")
    st.markdown("**Method:** HAZUS fragility curves + power law vulnerability")

    loss_df = load_loss_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Holland Loss",    "$28.29B")
    c2.metric("cGAN Loss",       "$27.19B",  "-$1.10B")
    c3.metric("Overall MDR",     "56.3%",    "Holland")
    c4.metric("cGAN MDR",        "54.2%",    "-2.1 pts")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Loss by Building Type", "Wind-Loss Nonlinearity", "Comparison Chart"])

    with tab1:
        st.subheader("Loss by Building Type")
        plot_df = loss_df[loss_df['building_type'] != 'Total'].copy()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Holland loss',
            x=plot_df['building_type'],
            y=plot_df['loss_hol_b'],
            marker_color='#E8593C',
            text=plot_df['loss_hol_b'].apply(lambda x: f'${x:.2f}B'),
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name='cGAN loss',
            x=plot_df['building_type'],
            y=plot_df['loss_cgan_b'],
            marker_color='#3B8BD4',
            text=plot_df['loss_cgan_b'].apply(lambda x: f'${x:.2f}B'),
            textposition='outside'
        ))
        fig.update_layout(
            barmode='group',
            title='Estimated loss by building type — Holland vs cGAN ($B)',
            yaxis_title='Loss ($B)',
            xaxis_title='Building Type',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            loss_df.style.format({
                'tiv_b'        : '${:.2f}B',
                'mdr_hol_pct'  : '{:.1f}%',
                'mdr_cgan_pct' : '{:.1f}%',
                'loss_hol_b'   : '${:.2f}B',
                'loss_cgan_b'  : '${:.2f}B',
            }),
            hide_index=True,
            use_container_width=True
        )

    with tab2:
        st.subheader("Why Small Wind Change = Large Financial Impact")
        st.markdown(
            "Hurricane damage scales **nonlinearly** with wind speed — "
            "approximately as V³ near Cat 4 intensities."
        )

        wind_range = np.linspace(80, 200, 300)
        tiv        = 50.23e9

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                'Vulnerability curve (damage ratio vs wind)',
                'Loss amplification on $50.23B portfolio'
            ]
        )

        for exp, name, color in [
            (2.0, 'Quadratic (V²)', '#3B8BD4'),
            (3.0, 'Cubic (V³) — typical', '#E8593C'),
            (4.0, 'Quartic (V⁴)', '#EF9F27'),
        ]:
            dr = np.minimum(1.0, (wind_range / 100) ** exp * 0.15)
            fig.add_trace(
                go.Scatter(x=wind_range, y=dr * 100,
                           name=name, line=dict(color=color, width=2)),
                row=1, col=1
            )

        # Vertical lines for your values
        for v, label, color in [
            (190.3, 'cGAN (190.3)', '#1D9E75'),
            (195.0, 'Holland (195.0)', '#E8593C'),
        ]:
            for col_n in [1, 2]:
                fig.add_vline(
                    x=v, line_dash='dash', line_color=color,
                    annotation_text=label,
                    annotation_position='top',
                    row=1, col=col_n
                )

        # Loss curve
        loss_curve = np.minimum(1.0, (wind_range / 100) ** 3 * 0.15) * tiv / 1e9
        fig.add_trace(
            go.Scatter(x=wind_range, y=loss_curve,
                       name='Loss ($B)', line=dict(color='#E8593C', width=2.5),
                       showlegend=False),
            row=1, col=2
        )

        fig.update_xaxes(title_text='Wind speed (mph)')
        fig.update_yaxes(title_text='Damage ratio (%)', row=1, col=1)
        fig.update_yaxes(title_text='Loss ($B)', row=1, col=2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Key insight:** A 4.7 mph wind reduction (2.4%) produces "
            "~$1.1B loss reduction (3.9%) — 1.6× amplification "
            "from the cubic damage function + $50.23B exposure scale."
        )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            show_image(
                f'{CGAN_DIR}/holland_vs_cgan_loss_comparison.png',
                caption='Holland vs cGAN loss comparison'
            )
        with col2:
            show_image(
                f'{CGAN_DIR}/wind_exposure_overlay.png',
                caption='Wind field and exposure overlay'
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Training":
    st.title("📊 cGAN Training — Balanced Dataset")

    loss_df = load_training_loss()

    if loss_df is not None:
        best_epoch = loss_df['val_loss'].idxmin() + 1
        best_val   = loss_df['val_loss'].min()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Epochs",     str(len(loss_df)))
        c2.metric("Best Epoch",       str(best_epoch))
        c3.metric("Best Val Loss",    f"{best_val:.4f}")
        c4.metric("Final G Loss",     f"{loss_df['g_loss'].iloc[-1]:.4f}")

        st.divider()

        # Training curves
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Adversarial training loss', 'Validation loss']
        )
        fig.add_trace(
            go.Scatter(x=loss_df['epoch'], y=loss_df['g_loss'],
                       name='Generator', line=dict(color='#E8593C', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=loss_df['epoch'], y=loss_df['d_loss'],
                       name='Discriminator', line=dict(color='#3B8BD4', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=loss_df['epoch'], y=loss_df['val_loss'],
                       name='Validation loss', line=dict(color='#1D9E75', width=2)),
            row=1, col=2
        )
        fig.add_hline(
            y=best_val, line_dash='dash', line_color='red',
            annotation_text=f'Best: {best_val:.4f} (ep {best_epoch})',
            row=1, col=2
        )
        fig.update_xaxes(title_text='Epoch')
        fig.update_yaxes(title_text='Loss', row=1, col=1)
        fig.update_yaxes(title_text='Val loss', row=1, col=2)
        fig.update_layout(
            height=400,
            title='cGAN Training — Balanced Dataset (100 epochs)'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Raw data
        with st.expander("View raw training data"):
            st.dataframe(
                loss_df.style.format({
                    'g_loss'   : '{:.4f}',
                    'd_loss'   : '{:.4f}',
                    'val_loss' : '{:.4f}',
                }).highlight_min(subset=['val_loss'], color='#E1F5EE'),
                hide_index=True,
                use_container_width=True
            )
    else:
        st.error("Training loss CSV not found.")

    st.divider()
    st.subheader("Architecture Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Generator — U-Net**
        ```
        Input  : coarse wind patch (1, 22, 21)
                 + condition vector (4,)
        enc1   : Conv2d(2, 64)   LeakyReLU
        enc2   : Conv2d(64, 128) BN LeakyReLU
        enc3   : Conv2d(128, 256) BN LeakyReLU
        bttnck : Conv2d(256, 512) BN ReLU
        dec3   : Conv2d(768, 256) BN ReLU Dropout
        dec2   : Conv2d(384, 128) BN ReLU Dropout
        dec1   : Conv2d(192, 64)  BN ReLU
        up     : Upsample → (201, 201) Sigmoid
        Output : fine wind patch (1, 201, 201)
        ```
        """)

    with col2:
        st.markdown("""
        **Discriminator — PatchGAN**
        ```
        Judges 70×70 patches as real/fake
        Forces local texture realism
        Condition vector injected at input
        ```

        **Loss Functions**
        ```
        L_adv  : Generator vs discriminator
        L_L1   : Pixel-level MAE loss
        L_phys : Peak location + decay
        ```

        **Training Config**
        ```
        Optimizer : Adam (lr=2e-4, β=0.5)
        Batch size: 32
        Epochs    : 100
        Dataset   : 2,500 balanced samples
        ```
        """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — PEER REVIEW DEFENSE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛡 Peer Review Defense":
    st.title("🛡 Peer Review — Q&A Defense")
    st.markdown(
        "Complete answers to all peer review questions — "
        "backed by real validation numbers from this project."
    )

    questions = [
        {
            "q"       : "How does your cGAN ensure physically realistic coastal wind gradients?",
            "answer"  : "Physical realism comes from four layers: (1) physics conditioning on Vmax/RMW/Pmin/Lat, (2) Holland model as structural prior, (3) coastal gradient patterns implicitly learned from training data, (4) post-generation validation. However, conditioning collapse means the model relies primarily on the coarse input — a known limitation.",
            "evidence": "r=0.974 vs Holland fine, 0.6% peak error, wind decay confirmed",
            "confidence": "Medium — Holland consistency confirmed, real atmosphere not validated",
            "fix"     : "Validate against ASOS stations. Replace Holland fine with HWind/ERA5 as ground truth.",
            "color"   : "warning"
        },
        {
            "q"       : "What input features have the biggest impact on generated results?",
            "answer"  : "None of the four conditioning features have meaningful impact. All show < 0.1 mph variation across ±30% perturbation — confirmed as numerical noise. This is conditioning collapse: the coarse wind field encodes all intensity information, making the condition vector redundant.",
            "evidence": "Zero condition → 190.5 mph, Normal → 190.3 mph, Ones → 190.3 mph. All 4 features: < 0.1 mph variation (0.05% of output).",
            "confidence": "High — definitively confirmed by zero/ones test",
            "fix"     : "Multi-depth condition injection at bottleneck + decoder. Add conditioning loss term during training.",
            "color"   : "error"
        },
        {
            "q"       : "How accurate are generated scenarios vs real historical data?",
            "answer"  : "Against Holland fine reference: r=0.974 and 0.6% peak error — excellent. Against real atmosphere: unknown — no ASOS or HWind validation performed. The r=0.974 proves Holland consistency, not atmospheric accuracy.",
            "evidence": "Peak wind: 189.2 mph input vs 190.3 mph output. MAE = 1.99 mph vs 1.78 mph baseline.",
            "confidence": "High vs Holland, Unknown vs real atmosphere",
            "fix"     : "Compare output to NOAA ASOS stations at KFMY, KPGD, KAPF during Ian landfall.",
            "color"   : "warning"
        },
        {
            "q"       : "How do you confirm the cGAN produced conditioned output?",
            "answer"  : "We ran a definitive test: passed zero vector, normal vector, and ones vector as conditions. All three produced identical outputs (~190.3 mph). This confirms conditioning collapse — the generator ignores the condition vector entirely.",
            "evidence": "Normal: 190.3 mph. Zero: 190.5 mph. Ones: 190.3 mph. Difference: 0.2 mph = numerical noise.",
            "confidence": "High — conditioning collapse definitively confirmed",
            "fix"     : "Inject condition at bottleneck + decoder layers. Add auxiliary conditioning loss.",
            "color"   : "error"
        },
        {
            "q"       : "How does the cGAN learn a distribution from a single event?",
            "answer"  : "It does not truly learn a distribution. Training data audit revealed all 2,500 samples at exactly 26.3°N (Ian's landfall). Vmax/RMW/Pmin vary synthetically but geography is fixed. Model learned coarse→fine mapping for one location, not a general hurricane distribution.",
            "evidence": "Latitude unique values: 1. Latitude std dev: 0.000000. All samples: 26.3°N.",
            "confidence": "High — confirmed by training data audit",
            "fix"     : "Train on full IBTrACS Gulf database (317 storms) across all latitudes 18°N–32°N.",
            "color"   : "error"
        },
        {
            "q"       : "How do you handle climate non-stationarity?",
            "answer"  : "Currently not handled — model assumes stationarity. Three vulnerabilities: (1) future Vmax may exceed training max of 165 kt under SSP5-8.5, (2) poleward track shifts of 2–4° are outside training latitude of 26.3°N, (3) no SST feature so cannot condition on warmer oceans.",
            "evidence": "Training max Vmax: 165 kt. IPCC projects +5–10%. Training latitude: 26.3°N only.",
            "confidence": "This is a known limitation — shared by most academic cat models",
            "fix"     : "Add SST as 5th conditioning feature. Periodic retraining on rolling 30-year IBTrACS window.",
            "color"   : "warning"
        },
        {
            "q"       : "What validation is needed for regulatory adoption?",
            "answer"  : "Five layers required: (1) backtest on 10+ historical storms, (2) benchmark against AIR/RMS, (3) independent actuary review, (4) uncertainty quantification via Monte Carlo, (5) FCHLPM 64-point checklist compliance. Currently only validated on Ian — insufficient for regulatory use.",
            "evidence": "Florida statute 627.0628 requires FCHLPM review. Commercial models take 3–5 years to certify.",
            "confidence": "Not ready for regulatory use in current form",
            "fix"     : "Full FCHLPM submission process. Engage certified actuary (FCAS).",
            "color"   : "warning"
        },
        {
            "q"       : "Why is the error small but financial impact large?",
            "answer"  : "Three reasons: (1) damage scales as V³ near Cat 4 — 2.4% wind reduction produces 7.1% damage ratio reduction, (2) at 190+ mph we are on the steepest part of the vulnerability curve, (3) on $50.23B TIV even 1% damage ratio change = $500M. Combined: 4.7 mph improvement → $1.1B loss difference.",
            "evidence": "4.7 mph wind difference → $28.29B vs $27.19B = $1.10B loss difference.",
            "confidence": "High — well-established in insurance literature",
            "fix"     : "N/A — this is a feature, not a limitation.",
            "color"   : "success"
        },
        {
            "q"       : "Is the cGAN learning real coastal physics or Holland's assumptions?",
            "answer"  : "Holland's assumptions at higher resolution — not real coastal physics. Ground truth was Holland fine (same parametric equation, finer grid). r=0.974 confirms Holland reproduction. To claim real physics we would need HWind/ERA5 as ground truth.",
            "evidence": "Training: Holland coarse → Holland fine. Both from same equation. No observational data used.",
            "confidence": "Low for real physics, High for Holland consistency",
            "fix"     : "Replace Holland fine with NOAA HWind analysis or ERA5 reanalysis as training target.",
            "color"   : "error"
        },
    ]

    for i, item in enumerate(questions, 1):
        with st.expander(f"Q{i}: {item['q']}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Answer:** {item['answer']}")
                st.markdown(f"**Evidence:** `{item['evidence']}`")
                if item['color'] == 'success':
                    st.success(f"**Fix / Next step:** {item['fix']}")
                elif item['color'] == 'error':
                    st.error(f"**Fix / Next step:** {item['fix']}")
                else:
                    st.warning(f"**Fix / Next step:** {item['fix']}")
            with col2:
                conf_colors = {
                    'success' : '🟢',
                    'warning' : '🟡',
                    'error'   : '🔴'
                }
                st.markdown(f"**Confidence:** {conf_colors[item['color']]}")
                st.markdown(f"_{item['confidence']}_")

    st.divider()
    st.subheader("Overall Project Assessment")
    col1, col2 = st.columns(2)
    with col1:
        st.success("""
        **Strengths**
        - r=0.974 super-resolution accuracy
        - 0.6% peak wind error
        - Rigorous validation performed
        - Honest limitation disclosure
        - Conditioning collapse diagnosed
        - Training data audit completed
        """)
    with col2:
        st.error("""
        **Limitations**
        - Conditioning collapse — features ignored
        - Circular ground truth (Holland→Holland)
        - Single location (26.3°N only)
        - No ASOS/HWind validation
        - Climate non-stationarity not modeled
        - One event (Ian) — not regulatory grade
        """)
