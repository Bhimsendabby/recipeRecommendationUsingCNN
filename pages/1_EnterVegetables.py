import streamlit as st
from model2 import get_recommendations

#Set the page configuration for multiple page
st.set_page_config(page_title="User Input Recommendation", page_icon="📈")

#Markdown to print the required content
st.markdown("# Enter the Vegetables Names To Get Recommendation Of Recipes")
st.sidebar.header("Enter Vegetables Manually")
st.write("""This demo illustrates where a user can enter the vegetables names in the input box""")

# For taking the text input for the recommendation instead of image
vegs = st.text_input("Enter some text",placeholder="Enter the Vegetables names by comma separated values")

#split the vegetables
st.write("Enter",vegs)

st.markdown(""" <style> .font {
        font-size:100px;} 
        </style> """, unsafe_allow_html=True)

if vegs:
    splitedVeg = vegs.split(",")
    #To submit the entered vegetables
    textResult = st.button('Recommend')
    st.text("Recommended Recipes Are....")
    if textResult:
        st.write("")
        st.write("Classifying...")
        f = get_recommendations(splitedVeg)
        # style for the table
        th_props = [
            ('font-size', '14px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('color', '#6d6d6d'),
            ('background-color', '#f7ffff')
        ]

        #font size for the table text
        td_props = [
            ('font-size', '18px')
        ]

        #dict object for styling
        styles = [
            dict(selector="th", props=th_props),
            dict(selector="td", props=td_props),
        ]

        # table
        f = f.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
        st.table(f)
