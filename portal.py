import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#____________________________________________________________________________________________________________________________________________
# Load geospatial and material data
df_geospatial = pd.read_csv("synthetic_geospatial_data.csv")
df_materials = pd.read_csv("construction_materials_data.csv")
df_environment = pd.read_csv("synthetic_environmental_data.csv")
gdf_geospatial = gpd.GeoDataFrame(df_geospatial, geometry=gpd.points_from_xy(df_geospatial.Longitude, df_geospatial.Latitude))


#___________________________________________MACHINE LEARNING MODELS______________________________________________________________________________________
# Train the models
X = df_materials[["CO2_Emissions", "Recyclability", "Energy_Efficiency", "Availability", "Durability", "Aesthetic_Value"]]
y_cost = df_materials["Cost"]
y_eco = df_materials["Eco_Friendly"]

X_train, X_test, y_cost_train, y_cost_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
_, _, y_eco_train, y_eco_test = train_test_split(X, y_eco, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_cost_train)

clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train, y_eco_train)

#___________________________________________________STREAMLIT___________________________________________________________________________
# Set page title and favicon
st.set_page_config(page_title="Jaipur Urban Planning", page_icon="🏙️", layout="wide")

# Initialize session state
if 'selected_point' not in st.session_state:
    st.session_state['selected_point'] = None

if 'page' not in st.session_state:
    st.session_state['page'] = "Home"

# Define a function to create the map
def create_map(filtered_gdf):
    center = [26.9124, 75.7873]
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    category_colors = {
        "Park": "green",
        "Residential": "red",
        "Commercial": "blue",
        "School": "orange",
        "Library": "purple",
        "Government Office": "yellow",
        "Hospital": "pink",
        "Utility": "gray"
    }

    for idx, row in filtered_gdf.iterrows():
        color = category_colors.get(row["Land Use"], "black")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"{row['Land Use']}<br>Lat: {row.geometry.y}<br>Lon: {row.geometry.x}"
        ).add_to(m)

    return m

# Create the main page layout
col1,col2=st.columns([4,1])

with col2:
    st.image("logo.jpg",width=130)
with col1:
    st.markdown("<h1 style='font-size: 50px; text-align: Center;'>JUPADP : JAIPUR URBAN PLANNING AND DEVELOPMENT PLATFORM </h1>", unsafe_allow_html=True)

# Horizontal Menu Bar with Icons
selected_option = option_menu(
    menu_title=None,
    options=["Home", "Infrastructure", "Building Placement", "About"],
    icons=["house", "building", "tools", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Handle page navigation
if selected_option == "Home":
    st.session_state['page'] = "Home"
    st.subheader("Welcome to the Jaipur Urban Planning and Development Optimization Portal.")
    col1, col2 = st.columns([2, 2])
    with col1:
        st.image("img.jpg")
    with col2:
        try:
            with open('text.txt', 'r') as file:
                file_contents = file.read()
            st.write(file_contents)
        except Exception as e:
            st.write("An error occurred while reading the file:")
            st.write(e)



elif selected_option == "Infrastructure":
    st.session_state['page'] = "Infrastructure"
    st.subheader("Infrastructure Categories")

    with st.sidebar:
        infra_option = st.selectbox(
            "Select Infrastructure Type",
            ["All", "Park", "Commercial", "Residential", "School", "Library", "Government Office", "Hospital", "Utility"]
        )

    # Filter the geospatial data based on the selected infrastructure type
    if infra_option == "All":
        filtered_gdf = gdf_geospatial
    else:
        filtered_gdf = gdf_geospatial[gdf_geospatial["Land Use"] == infra_option]

    
    m = create_map(filtered_gdf)
    output = st_folium(m, height=450, width=900)

    if output['last_clicked']:
        st.session_state['selected_point'] = output['last_clicked']
        st.session_state['page'] = "Building Placement"
        st.experimental_rerun()

    if st.session_state['selected_point']:
        st.write(f"Selected Location: {st.session_state['selected_point']}")

elif selected_option == "Building Placement" or st.session_state['page'] == "Building Placement":
    if st.session_state['selected_point']:
        #st.write(f"Selected Location: {st.session_state['selected_point']}")

        # Get the selected location's coordinates
        selected_lat = st.session_state['selected_point']['lat']
        selected_lon = st.session_state['selected_point']['lng']

        # Find the closest environmental data point
        df_environment['Distance'] = ((df_environment['Latitude'] - selected_lat)**2 + (df_environment['Longitude'] - selected_lon)**2)**0.5
        closest_env_data = df_environment.loc[df_environment['Distance'].idxmin()]

        # Display environmental data for the selected location in columns
        st.markdown("<h1 style='font-size: 40px; text-align: Center;'>Environmental Data for Selected Location</h1>", unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        # CO2 Emissions
        co2_level_str = closest_env_data['CO2_Emissions']
        co2_status = "High" if float(co2_level_str) > 200 else "Moderate" if float(co2_level_str) > 100 else "Low"
        co2_icon = "🌳" if co2_status == "Low" else "🚗" if co2_status == "Moderate" else "🔥"
        col1.metric("CO2 Emissions", f"{co2_icon} {co2_status}")

        # Traffic Congestion
        traffic_value = closest_env_data['Traffic_Congestion']
        traffic_status = traffic_value.capitalize() 
        traffic_icon = "⬇️" if traffic_status == "Good" else "⚠️" if traffic_status == "Average" else "🔺"
        col2.metric("Traffic Congestion", f"{traffic_icon} {traffic_status}")


        # Water Availability
        water_level_str = closest_env_data['Water_Availability']
        water_status = water_level_str.capitalize()  
        water_icon = "💧" if water_status == "Good" else "🟡" if water_status == "Average" else "❌"
        col3.metric("Water Availability", f"{water_icon} {water_status}")

        # Waste Management
        waste_level_str = closest_env_data['Waste_Management']
        waste_status = waste_level_str.capitalize()  
        waste_icon = "✅" if waste_status == "Good" else "⚠️" if waste_status == "Average" else "❌"
        col4.metric("Waste Management", f"{waste_icon} {waste_status}")


        # Population Estimates
        population_level = closest_env_data['Population_Estimates']
        population_status = "Low" if population_level < 1100 else "Moderate" if population_level < 1600 else "High"
        population_icon = "👶" if population_status == "Low" else "👨‍👩‍👦‍👦" if population_status == "Moderate" else "🏙️"
        col5.metric("Population Estimates", f"{population_icon} {population_status}")


        # Buttons to access cost and eco-friendly material
        st.subheader("Get Recommendations")
        col1,col2=st.columns([4,1])
        with col2:
            if st.button("Predict Construction Cost"):
                # Prepare input for the model based on environmental data
                model_input = pd.DataFrame([[
                    closest_env_data["CO2_Emissions"],
                    closest_env_data["Recyclability"],
                    closest_env_data["Energy_Efficiency"],
                    closest_env_data["Availability"],
                    closest_env_data["Durability"],
                    closest_env_data["Aesthetic_Value"]
                ]], columns=["CO2_Emissions", "Recyclability", "Energy_Efficiency", "Availability", "Durability", "Aesthetic_Value"])

                # Predict cost
                predicted_cost = reg_model.predict(model_input)[0]
                st.success(f"Predicted Construction Cost: Rs. {predicted_cost:.2f} Lacs")
        with col1:
            if st.button("Recommend Eco-Friendly Materials"):
                # Prepare input for the model based on environmental data
                model_input = pd.DataFrame([[
                    closest_env_data["CO2_Emissions"],
                    closest_env_data["Recyclability"],
                    closest_env_data["Energy_Efficiency"],
                    closest_env_data["Availability"],
                    closest_env_data["Durability"],
                    closest_env_data["Aesthetic_Value"]
                ]], columns=["CO2_Emissions", "Recyclability", "Energy_Efficiency", "Availability", "Durability", "Aesthetic_Value"])

                # Predict eco-friendliness
                predicted_eco = clf_model.predict(model_input)[0]
                #st.write(f"Is the neighbourhood Eco-Friendly? -- {'Yes' if predicted_eco == 1 else 'No'}")

                # Display recommended materials (for demonstration, we show the top 5 materials with the lowest cost)
                eco_friendly_materials = df_materials[df_materials["Eco_Friendly"] == 1]
                recommended_materials = eco_friendly_materials.sort_values(by="Cost").head(5)
                st.write("Recommended Eco-Friendly Materials (Top 5 with Lowest Cost in Lacs):")
                st.dataframe(recommended_materials)

    else:
        st.write("Please select a location on the Infrastructure page.")

elif selected_option == "About":
    st.session_state['page'] = "About"
    st.header("About")
    with open('about.txt', 'r') as file:
                file_contents = file.read()
    st.success(file_contents)

st.markdown("---")
st.markdown("Created by: Kamayani 👩‍💻")