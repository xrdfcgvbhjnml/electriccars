import pandas as pd
import numpy as np 
import requests
import plotly.express as px
import folium
from ipywidgets import interact, Dropdown
from folium.plugins import MarkerCluster
import streamlit as st
import re
from streamlit_folium import st_folium
import numpy as np
import plotly.graph_objects as go
import itertools
from datetime import timedelta
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Pagina instellingen ---
st.set_page_config(page_title="Auto Dashboard", layout="wide")

# --- Navigatie ---
st.sidebar.title("📂 Navigatie")
pagina = st.sidebar.radio(
    "Kies een sectie:",
    ["🚗 Auto Dashboard", "⚡ Laadpalen Kaart"]
)

# --- Data inladen ---
@st.cache_data
def load_data():
    cars = pd.read_pickle("cars.pkl")
    cars['datum_eerste_toelating'] = pd.to_datetime(cars['datum_eerste_toelating'], errors='coerce')
    cars['jaar_maand'] = cars['datum_eerste_toelating'].dt.to_period('M').astype(str)
    return cars

cars = load_data()

data_openchargemap = requests.get("https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=1000&compact=true&verbose=false&key=5d087822-ce71-42b0-a231-67209f0900a2")

Laadpalen = pd.json_normalize(data_openchargemap.json())

df = pd.json_normalize(Laadpalen.Connections)
df2 = pd.json_normalize(df[0])
Laadpalen = pd.concat([Laadpalen, df2], axis=1)

if pagina == "🚗 Auto Dashboard":
    st.title("🚗 Auto Dashboard")

    # --- Totale aantallen per merk ---
    totaal_per_merk = cars.groupby('merk').size().sort_values(ascending=False)
    top5 = totaal_per_merk.nlargest(5).index
    top10 = totaal_per_merk.nlargest(10).index

    autos_per_merk_per_maand = (
        cars.groupby(['jaar_maand', 'merk'])
            .size()
            .reset_index(name='aantal_autos')
    )

    # --- Normalisatie-functie voor model ---
    def normalize_model(name):
        if pd.isna(name):
            return None
        name = name.upper().strip()
        match = re.match(r'(MODEL [A-Z0-9]+)', name)
        if match:
            return match.group(1)
        match = re.match(r'(ID\.?\s*\d)', name)
        if match:
            return match.group(1).replace(' ', '').upper()
        match = re.match(r'(E[-\s]?\d+)', name)
        if match:
            return match.group(1).replace(' ', '').upper()
        return name.split()[0]

    cars['model_basis'] = cars['handelsbenaming'].apply(normalize_model)

    # --- Functies voor grafieken ---
    def plot_top_merks(top_labels, titel):
        filtered = autos_per_merk_per_maand[autos_per_merk_per_maand['merk'].isin(top_labels)]
        fig = px.scatter(
            filtered,
            x='jaar_maand',
            y='aantal_autos',
            color='merk',
            title=titel,
        )
        fig.update_traces(mode='lines+markers', marker=dict(size=6, opacity=0.7), line=dict(width=0.5))
        fig.update_layout(template='plotly_white', xaxis=dict(categoryorder='category ascending'), 
                          xaxis_title='Maand', yaxis_title='Aantal auto\'s')
        st.plotly_chart(fig, use_container_width=True)

    def plot_merk_trends(merknaam):
        merk_df = cars[cars['merk'].str.upper() == merknaam.upper()]
        if merk_df.empty:
            st.warning(f"⚠️ Geen resultaten gevonden voor merk: {merknaam}")
            return

        per_model_per_maand = (
            merk_df.groupby(['jaar_maand', 'model_basis'])
                .size()
                .reset_index(name='aantal_autos')
        )

        fig = px.scatter(
            per_model_per_maand,
            x='jaar_maand',
            y='aantal_autos',
            color='model_basis',
            title=f"Aantal auto's per model van {merknaam} door de maanden heen",
        )
        fig.update_traces(mode='lines+markers', marker=dict(size=6, opacity=0.7), line=dict(width=0.5))
        fig.update_layout(template='plotly_white', xaxis=dict(categoryorder='category ascending'),
                          xaxis_title='Maand', yaxis_title='Aantal auto\'s')
        st.plotly_chart(fig, use_container_width=True)

    # --- Keuze tussen Top 5 / 10 ---
    keuze = st.radio(
        "📊 Kies welke merken te tonen:",
        ["Top 5 merken", "Top 10 merken"],
        horizontal=True
    )

    if keuze == "Top 5 merken":
        plot_top_merks(top5, "📈 Aantal auto's per merk (Top 5) door de maanden heen")
    else:
        plot_top_merks(top10, "📈 Aantal auto's per merk (Top 10) door de maanden heen")

    # --- Modeltrends sectie ---
    st.markdown("---")
    st.markdown("### 🔍 Modeltrends per merk")

    merk_opties = sorted(cars['merk'].dropna().unique())
    merk_naam = st.selectbox("Selecteer een merk om modeltrends te bekijken:", merk_opties, index=0)
    plot_merk_trends(merk_naam)

    # --- Extra visualisatie: Inrichting ---
    st.markdown("---")
    st.markdown("### 🚘 Aantal auto's per inrichting (carrosserie)")

    color_sequence = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6C5B7B",
    "#355C7D", "#F67280", "#99B898", "#E84A5F", "#FECEAB", "#3E606F"
]


    # ✅ Checkbox voor log-schaal
    log_scale = st.checkbox("Logaritmische schaal gebruiken", value=True)

    fig_inrichting = px.histogram(
        cars,
        x='inrichting',
        color='inrichting',
        color_discrete_sequence=color_sequence,
        title="Aantal auto’s per inrichting",
        log_y=log_scale
    )
    fig_inrichting.update_layout(
        xaxis_title='Carrosserie',
        yaxis_title="Aantal auto's (log)" if log_scale else "Aantal auto's",
        template='plotly_white'
    )
    st.plotly_chart(fig_inrichting, use_container_width=True)

    def plot_correlation_optimized(df, x_col, y_col, title=None, max_points=100):
        """Plot een scatterplot met trendline tussen twee numerieke kolommen."""
        # Converteer kolommen naar numeriek
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

        # Filter geldige waarden
        df_filtered = df[(df[x_col] > 0) & (df[y_col] > 0)].copy()
        if df_filtered.empty:
            st.warning("⚠️ Geen geldige data om te plotten voor deze combinatie.")
            return

        # Groepeer identieke waarden
        df_grouped = (
            df_filtered.groupby([x_col, y_col])
                    .size()
                    .reset_index(name='aantal')
        )

        # Downsample bij teveel punten
        if len(df_grouped) > max_points:
            df_grouped = df_grouped.sample(
                n=max_points,
                weights='aantal',
                random_state=42
            )

        # Trendline berekenen
        x = df_grouped[x_col].values
        y = df_grouped[y_col].values
        coeffs = np.polyfit(x, y, 1)
        trendline = np.polyval(coeffs, x)

        # Marker grootte proportioneel aan aantal
        sizes = np.clip(df_grouped['aantal'], 1, 7500)
        sizeref = 2. * max(sizes) / (50. ** 2)

        # Plot maken
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=sizes,
                color=sizes,
                colorscale='Plotly3',
                showscale=True,
                sizemode='area',
                sizeref=sizeref,
            ),
            text=[f"Aantal: {a}" for a in df_grouped['aantal']],
            name='Data'
        ))

        # Trendline toevoegen
        fig.add_trace(go.Scatter(
            x=x,
            y=trendline,
            mode='lines',
            line=dict(color='red'),
            name='Trendline'
        ))

        # Layout
        fig.update_layout(
            title=title or f"{y_col} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            template='plotly_white',
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)


    # --- STREAMLIT UI ---
    st.title("📊 Vergelijking voertuigvariabelen")

    # Beschikbare kolommen
    selected_columns = [
        'massa_ledig_voertuig',
        'vermogen_massarijklaar',
        'lengte',
        'breedte',
        'hoogte_voertuig'
    ]
    available_cols = [col for col in selected_columns if col in cars.columns]

    # Maak ALLE unieke combinaties
    combinations = list(itertools.combinations(available_cols, 2))

    # Regels: als 'vermogen_massarijklaar' in de combinatie zit, dan is dat de Y-as
    adjusted_combos = []
    for x, y in combinations:
        if 'vermogen_massarijklaar' in (x, y):
            if y == 'vermogen_massarijklaar':
                adjusted_combos.append((x, y))
            else:
                adjusted_combos.append((y, x))  # swap zodat vermogen_massarijklaar altijd Y wordt
        else:
            adjusted_combos.append((x, y))

    # Labels voor dropdown
    combo_labels = [f"{y} vs {x}" for x, y in adjusted_combos]

    # Dropdown voor alle combinaties
    selected_label = st.selectbox("Kies een variabelencombinatie", combo_labels)

    # Haal de gekozen combinatie op
    selected_index = combo_labels.index(selected_label)
    x_col, y_col = adjusted_combos[selected_index]

    # Plot
    st.markdown("---")
    plot_correlation_optimized(cars, x_col, y_col, f"📈 {y_col} tegenover {x_col}")

    # Eerst numerieke kolommen schoonmaken
    cars2 = cars.copy()
    cars2['catalogusprijs'] = pd.to_numeric(cars2['catalogusprijs'], errors='coerce')
    cars2['massa_ledig_voertuig'] = pd.to_numeric(cars2['massa_ledig_voertuig'], errors='coerce')
    cars2['vermogen_massarijklaar'] = pd.to_numeric(cars2['vermogen_massarijklaar'], errors='coerce')
    cars2['lengte'] = pd.to_numeric(cars2['lengte'], errors='coerce')
    cars2['breedte'] = pd.to_numeric(cars2['breedte'], errors='coerce')
    cars2['hoogte_voertuig'] = pd.to_numeric(cars2['hoogte_voertuig'], errors='coerce')
    
    # Verwijder rijen zonder prijs of met veel missende waarden
    cars2 = cars2.dropna(subset=['catalogusprijs', 'massa_ledig_voertuig', 'vermogen_massarijklaar'])
    
    # Feature selectie
    X = cars2[['massa_ledig_voertuig', 'vermogen_massarijklaar', 'lengte', 'breedte', 'hoogte_voertuig']]
    y = cars2['catalogusprijs']
    
    X = X.dropna()
    y = y.loc[X.index]
    
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # model --> andere models laten hetzelfde zien
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # prediction
    y_pred = model.predict(X_test)
    
    # Evaluatie
    #print("R²:", r2_score(y_test, y_pred))
    #print("MAE:", mean_absolute_error(y_test, y_pred))
    
    # Coëfficiënten
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })
    
    # plot
    plot_df = pd.DataFrame({
        "Actual Price": y_test,
        "Predicted Price": y_pred
    })
    
    plot_df["Error Value"] = np.abs(plot_df["Predicted Price"] - plot_df["Actual Price"])
    
    fig = px.scatter(
        plot_df,
        x="Actual Price",
        y="Predicted Price",    
        color="Error Value", 
        color_continuous_scale="RdYlGn_r", 
        opacity=0.6,
        title="Predicted vs. Actual Car Prices"
    )
    
    fig.add_scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode="lines",
        name="Regression Line",
        line=dict(dash="dot")
    )
    
    fig.update_layout(
        xaxis_title="Actual Price (in €)",
        yaxis_title="Predicted Price (in €)",
        legend_title=None
    )

    # Maak mooie regressieformule
    coef_lines = [f"{coef:.2f} × {feat}" for feat, coef in zip(X.columns, model.coef_)]
    formula_text = " +\n".join(coef_lines)
    intercept_text = f"{model.intercept_:.2f}"

    # Markdown string met mooie opmaak
    markdown_str = f"""
    **Regressieformule:**    
    **{formula_text} + {intercept_text}**
    """

    # Toon boven de plot
    st.markdown(markdown_str)
    fig.update_yaxes(range=[0,150000])
    fig.update_xaxes(range=[0,300000])
    st.plotly_chart(fig, use_container_width=True)

elif pagina == "⚡ Laadpalen Kaart":
    st.title("⚡ Laadpalen")
    st.markdown("Bekijk de locaties van laadpalen in Nederland, gefilterd per provincie.")

    tab1, tab2 = st.tabs(["📍 Laadpalen Kaart", "📊 Laadpaalgebruik"])

    with tab1:

        # 📍 Coördinaten van provincies
        provincie_locaties = {
            "Alle provincies": {"center": [52.2129919, 5.2793703], "zoom": 7},   # Midden Nederland
            "Groningen": {"center": [53.2194, 6.5665], "zoom": 10},
            "Friesland": {"center": [53.1642, 5.7818], "zoom": 10},
            "Drenthe": {"center": [52.9480, 6.6231], "zoom": 10},
            "Overijssel": {"center": [52.4380, 6.5010], "zoom": 10},
            "Flevoland": {"center": [52.5279, 5.5953], "zoom": 10},
            "Gelderland": {"center": [52.0452, 5.8718], "zoom": 10},
            "Utrecht": {"center": [52.0907, 5.1214], "zoom": 11},
            "Noord-Holland": {"center": [52.5200, 4.7885], "zoom": 9},
            "Zuid-Holland": {"center": [51.9961, 4.5597], "zoom": 10},
            "Zeeland": {"center": [51.4940, 3.8490], "zoom": 10},
            "Noord-Brabant": {"center": [51.4827, 5.2322], "zoom":10},
            "Limburg": {"center": [51.4427, 6.0600], "zoom": 9}
        }

        # Filter de Laadpalen DataFrame op geldige coördinaten
        Laadpaal_locatie = Laadpalen[[
            "AddressInfo.AddressLine1",
            "AddressInfo.Latitude",
            "AddressInfo.Longitude",
            "PowerKW", "DateCreated", "UsageCost", "NumberOfPoints"
        ]].dropna(subset=["AddressInfo.Latitude", "AddressInfo.Longitude"])

        # Dropdown voor provincies
        provincie = st.selectbox(
            "📍 Kies een provincie:",
            options=list(provincie_locaties.keys()),
            index=0
        )

        # Kaart functie
        def maak_kaart(provincie):
            loc = provincie_locaties.get(provincie, provincie_locaties["Alle provincies"])
            m = folium.Map(location=loc["center"], zoom_start=loc["zoom"], tiles="CartoDB positron")

            cluster = MarkerCluster().add_to(m)

            # Voeg alle laadpalen toe
            for _, r in Laadpaal_locatie.iterrows():
                folium.CircleMarker(
                    location=[r["AddressInfo.Latitude"], r["AddressInfo.Longitude"]],
                    radius=3,
                    color="blue",
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"{r['AddressInfo.AddressLine1']}<br>Vermogen: {r['PowerKW']} kW <br> Datum toegevoegd: {r['DateCreated']} <br> Kosten: {r['UsageCost']} <br> Aantal oplaadpunten: {r['NumberOfPoints']}"
                ).add_to(cluster)

            return m

        # Toon kaart in Streamlit
        m = maak_kaart(provincie)
        st_folium(m, width=1200, height=700)

        # 📊 Nieuwe grafieken: laadpaal-vermogens
        paal = pd.read_pickle("Charging_data.pkl")
        power_data = Laadpalen[['PowerKW']].dropna().copy()
        power_data['PowerKw_cat'] = power_data['PowerKW'].apply(
            lambda x: 'Laag' if x < 11 else ('Middel' if x < 22 else 'Hoog'))
        
        color_sequence2 = ["#00B4D8", "#FFD166", "#EF476F"]

        
        log_scale = st.checkbox("Logaritmische schaal gebruiken", key='1', value=True)
        
        # Scatterplot - maximaal laadvermogen
        fig_hist = px.histogram(
            power_data,
            x='PowerKW',
            title="🔋 Verdeling van maximaal laadvermogen per laadpaal",
            log_y=log_scale
        )
        fig_hist.update_layout(
            xaxis_title="Maximaal laadvermogen (kW)",
            yaxis_title="Aantal observaties",
            template="plotly_white"
        )
        st.plotly_chart(fig_hist, use_container_width=True)


        # Histogram - categorieën laadvermogen
        fig_hist = px.histogram(
            power_data,
            x='PowerKw_cat',
            color='PowerKw_cat',
            color_discrete_sequence=color_sequence2,
            title="⚙️ Aantal laadpalen per vermogenscategorie",
        )
        fig_hist.update_layout(
            xaxis_title="Categorie laadvermogen",
            yaxis_title="Aantal laadpalen",
            template="plotly_white",
            showlegend=False
        )

        fig_update_xaxis = fig_hist.update_xaxes(categoryorder='array', categoryarray=['Laag', 'Middel', 'Hoog'])
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.markdown("### ⚡ Laadpaal gebruiksanalyse")

        # Zet datums om
        paal["start_time"] = pd.to_datetime(paal["start_time"], errors='coerce')
        paal["exit_time"] = pd.to_datetime(paal["exit_time"], errors='coerce')
        paal = paal.dropna(subset=["start_time", "exit_time"])

        # Zet charging_duration om naar timedelta
        paal["charging_duration"] = pd.to_timedelta(paal["charging_duration"], errors='coerce')
        paal = paal[paal["charging_duration"] >= pd.Timedelta(0)]

        # Bereken connected_duration
        paal["connected_duration"] = paal["exit_time"] - paal["start_time"]

        # Filter nogmaals
        paal = paal.dropna(subset=["connected_duration"])

        # Kolommen naar minuten
        paal["charging_duration_min"] = paal["charging_duration"].dt.total_seconds() / 60
        paal["connected_duration_min"] = paal["connected_duration"].dt.total_seconds() / 60

        # Volledig opgeladen?
        paal["FullyCharged?"] = abs(paal["charging_duration_min"] - paal["connected_duration_min"]) != 0

        # Scatterplot
        LDscatter = px.scatter(
            paal,
            x="charging_duration_min",
            y="connected_duration_min",
            color="FullyCharged?",
            color_discrete_sequence=['rgb(50, 168, 82)', 'rgb(186, 31, 28)'],
            opacity=0.75,
            labels={
                "connected_duration_min": "Tijd verbonden (minuten)",
                "charging_duration_min": "Tijd aan het opladen (minuten)",
                "FullyCharged?": "Volledig opgeladen?"
            },
            title="⏱️ Vergelijking verbonden tijd vs laadtijd"
        )
        st.plotly_chart(LDscatter, use_container_width=True)

        LDscatter = px.scatter(
            paal,
            x="charging_duration_min",
            y='energy_delivered [kWh]',
            color='charging_duration_min',
            color_continuous_scale='Algae',
            opacity=0.9,
            labels={
                "charging_duration_min": "Tijd aan het opladen (minuten)",
                "energy_delivered [kWh]": "Geleverde energie (kWh)"
            },
            title="⚡ Vergelijking laadtijd vs geleverde energie"
        )
        st.plotly_chart(LDscatter, use_container_width=True)

        paal['max_power_cat'] = paal['max_charging_power [kW]'].apply(
            lambda x: 'Laag' if x < 11 else ('Middel' if x < 22 else 'Hoog'))
        
        log_scale2 = st.checkbox("Logaritmische schaal gebruiken", key='2', value=True)
        
        # Histogram - maximaal laadvermogen
        fig_hist = px.histogram(
            paal,
            x='max_charging_power [kW]',
            title="🔋 Verdeling van maximaal laadvermogen per laadpaal",
            log_y=log_scale2
        )
        fig_hist.update_layout(
            xaxis_title="Maximaal laadvermogen (kW)",
            yaxis_title="Aantal observaties",
            template="plotly_white"
        )
        st.plotly_chart(fig_hist, use_container_width=True)


        # Histogram - categorieën laadvermogen
        fig_hist = px.histogram(
            paal,
            x='max_power_cat',
            color='max_power_cat',
            color_discrete_sequence=color_sequence2,
            title="⚙️ Aantal laadpalen per vermogenscategorie",
        )
        fig_hist.update_layout(
            xaxis_title="Categorie laadvermogen",
            yaxis_title="Aantal laadpalen",
            template="plotly_white",
            showlegend=False
        )

        fig_update_xaxis = fig_hist.update_xaxes(categoryorder='array', categoryarray=['Laag', 'Middel', 'Hoog'])
        st.plotly_chart(fig_hist, use_container_width=True)

        
