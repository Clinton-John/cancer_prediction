
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def get_clean_data():
    cancer_df = pd.read_csv('Data/breast_cancer_data.csv')
    cancer_df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    cancer_df.replace({'diagnosis':{'M':1, 'B':0}}, inplace=True)
    return cancer_df



def add_sidebar():
    st.sidebar.header("Cell Nuclear Measurements")
    # to get the column names of the dataset, import the cleaned data function which passes the cleaned data
    data = get_clean_data()
    
    #create a lists of the label and keys where the keys are the columns from our dataset
    slider_labels = [
        ("Mean Radius","radius_mean"),
        ("Mean Texture", "texture_mean"),
        ("Mean Perimeter", "perimeter_mean"),
        ("Mean Area", "area_mean"),
        ("Mean Smoothness", "smoothness_mean"),
        ("Mean Compactness", "compactness_mean"),
        ("Mean Concavity", "concavity_mean"),
        ("Mean Concave Points", "concave points_mean"),
        ("Mean Symmetry", "symmetry_mean"),
        ("Mean Fractal Dimension", "fractal_dimension_mean"),
        ("Radius SE", "radius_se"),
        ("Texture SE", "texture_se"),
        ("Perimeter SE", "perimeter_se"),
        ("Area SE", "area_se"),
        ("Smoothness SE", "smoothness_se"),
        ("Compactness SE", "compactness_se"),
        ("Concavity SE", "concavity_se"),
        ("Concave Points SE", "concave points_se"),
        ("Symmetry SE", "symmetry_se"),
        ("Fractal Dimension SE", "fractal_dimension_se"),
        ("Worst Radius", "radius_worst"),
        ("Worst Texture", "texture_worst"),
        ("Worst Perimeter", "perimeter_worst"),
        ("Worst Area", "area_worst"),
        ("Worst Smoothness", "smoothness_worst"),
        ("Worst Compactness", "compactness_worst"),
        ("Worst Concavity", "concavity_worst"),
        ("Worst Concave Points", "concave points_worst"),
        ("Worst Symmetry", "symmetry_worst"),
        ("Worst Fractal Dimension", "fractal_dimension_worst")
    ]
     
    #create a new key value pair which will have the key as the name and the value as the selected value from the slider

    input_dict = {}
    #loop through the labels that we have set, creating each as a slider with the maximum value from the columns
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider( #here the key is the name of the column from the dataset
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value = float(data[key].mean())
        )

    return input_dict

#using the plotly library to come up with the interactive charts. from the plotly documentation
def get_radar_chart(input_data):
    import plotly.graph_objects as go

    # these are the categories from which we want to add the values from the graph
    categories = ['Radius','Texture','Perimeter','Area','Smoothness','Compactness', 'Concavity', 
    'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        #list containing the values that have been added by the scientists
        r=[
            input_data['radius_mean'],input_data['texture_mean'],input_data['perimeter_mean'],input_data['area_mean'],input_data['smoothness_mean'],input_data['compactness_mean'],input_data['concavity_mean'],input_data['concave points_mean'],input_data['symmetry_mean'],input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'],input_data['texture_se'],input_data['perimeter_se'],input_data['area_se'],input_data['smoothness_se'],input_data['compactness_se'],input_data['concavity_se'],input_data['concave points_se'],input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],input_data['texture_worst'],input_data['perimeter_worst'],input_data['area_worst'],input_data['smoothness_worst'],input_data['compactness_worst'],input_data['concavity_worst'],input_data['concave points_worst'],input_data['symmetry_worst'],input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))
    
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 5]
        )),
    showlegend=True
    )
    #streamlit uses its own function to add the plotly elements to the chart hence no need to use fig.show() instead return the fig
    return fig



def main():
    #initial page configurations that apply to the whole project 
    st.set_page_config(
        page_title="Breast Cancer Prediction",
        page_icon=':female_Doctor:',
        layout = 'wide',
        initial_sidebar_state = 'expanded'
    )

    #creating a function that adds the sidebar then calling it inside the main function for effieciency
    input_data = add_sidebar()

    # st.write(input_data)
    


    with st.container():
        st.title("Predictor For Breast Cancer")
        st.write("The Above Application is designed for use in Medical Institutions to predict whether a the tissue samples of a patient is cancer affected or the samples are safe")
    
    #creating the columns after the description is set
    col1, col2 = st.columns([4,1])
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        st.write("Column2 of Our Prediction Model")



if __name__ == '__main__':
    main()
