import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("data/titanic.csv")
df.columns = df.columns.str.lower()
df["class"] = df["pclass"].map({1: "First", 2: "Second", 3: "Third"})
df["survival"] = df["survived"].map({0: "Died", 1: "Survived"})
df["port"] = df["embarked"].map({"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"})
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["title"] = df["name"].str.extract(r",\s*([^\.]+)\.", expand=False)
df["age_group"] = pd.cut(df["age"], [0, 12, 18, 35, 60, 80],
                         labels=["child", "teen", "young adult", "adult", "senior"])

ORDERS = {"class": ["First", "Second", "Third"], "survival": ["Died", "Survived"],
          "age_group": ["child", "teen", "young adult", "adult", "senior"]}
COLORS = {"Died": "#D95D4A", "Survived": "#0F766E"}

st.title("Titanic survival prediction App")
st.caption("This app is a demo of a Streamlit app that predicts survival on the Titanic based on passenger data.")
st.image("utils/The-Titanic.jpg")

show_df = st.checkbox("Show the dataset")
if show_df:
    st.dataframe(df.head(10),hide_index=True)
    
classes=st.multiselect("Select class", df["class"].unique(), default=df["class"].unique())
gender=st.radio("Select gender", df["sex"].unique(),index=0,horizontal=True)

data=df[(df["class"].isin(classes)) & (df["sex"]==gender)]


fig = px.histogram(df, x="family_size", nbins=11, labels={"family_size": "Family size (incl. the passenger)"},
                   title="How big were the families on board?")
fig.update_layout(bargap=0.1, yaxis_title="Passengers")
st.plotly_chart(fig, use_container_width=True)



c1,c2=st.columns(2)

with c1:
    fig2 = px.histogram(data, x="age", nbins=30, marginal="box",
                   labels={"age": "Age"}, title="Age of the passengers")
    st.plotly_chart(fig2, use_container_width=True)

outcome = data["survival"].value_counts().reset_index()
with c2:
    fig3 = px.pie(outcome, names="survival", values="count", hole=0.5, color="survival",
             color_discrete_map=COLORS, title="Died vs survived")
    st.plotly_chart(fig3, use_container_width=True)