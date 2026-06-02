# ============================================================
# Task 5: Interactive Business Dashboard in Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Global Superstore Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# Custom White Theme CSS
# ============================================================

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    [data-testid="stHeader"] {
        background-color: #ffffff !important;
    }

    [data-testid="stSidebar"] {
        background-color: #f2f6fc !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #000000 !important;
    }

    .dashboard-title {
        text-align: center;
        color: #0b74d1 !important;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .dashboard-subtitle {
        text-align: center;
        color: #333333 !important;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .stMetric {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #d0d7de;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.10);
    }

    [data-testid="stMetricValue"] {
        color: #0b74d1 !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] label {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #0b74d1 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #d0d7de !important;
        border-radius: 16px !important;
        padding: 16px !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08) !important;
    }

    .stDataFrame {
        background-color: #ffffff !important;
    }

    hr {
        border: 1px solid #d0d7de;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Helper Function for Chart Styling
# ============================================================

def style_chart(fig, title):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black", size=13),
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=26, color="black"),
            x=0.02,
            xanchor="left"
        ),
        xaxis=dict(
            title_font=dict(color="black", size=14),
            tickfont=dict(color="black", size=12)
        ),
        yaxis=dict(
            title_font=dict(color="black", size=14),
            tickfont=dict(color="black", size=12)
        ),
        legend=dict(
            font=dict(color="black", size=12)
        )
    )
    return fig

# ============================================================
# Load Dataset
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("superstore.csv", encoding="latin1")
    return df

df = load_data()

# ============================================================
# Data Cleaning
# ============================================================

df.columns = df.columns.str.strip()

df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

df["Order.Date"] = pd.to_datetime(df["Order.Date"], errors="coerce")

df = df.dropna(subset=["Sales", "Profit", "Order.Date"])

# ============================================================
# Dashboard Title
# ============================================================

st.markdown(
    '<div class="dashboard-title">📊 Global Superstore Business Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="dashboard-subtitle">
    Interactive dashboard for analyzing sales, profit, customer performance, 
    and segment-wise business insights using the Global Superstore dataset.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Sidebar Filters
# ============================================================

st.sidebar.header("🔍 Filter Options")

region_filter = st.sidebar.multiselect(
    "Select Region",
    options=sorted(df["Region"].dropna().unique()),
    default=sorted(df["Region"].dropna().unique())[:5]
)

category_filter = st.sidebar.multiselect(
    "Select Category",
    options=sorted(df["Category"].dropna().unique()),
    default=sorted(df["Category"].dropna().unique())
)

subcategory_filter = st.sidebar.multiselect(
    "Select Sub-Category",
    options=sorted(df["Sub.Category"].dropna().unique()),
    default=sorted(df["Sub.Category"].dropna().unique())[:8]
)

filtered_df = df[
    (df["Region"].isin(region_filter)) &
    (df["Category"].isin(category_filter)) &
    (df["Sub.Category"].isin(subcategory_filter))
]

if filtered_df.empty:
    st.warning("No data available for the selected filters. Please change the filter options.")
    st.stop()

# ============================================================
# KPI Cards
# ============================================================

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order.ID"].nunique()
total_customers = filtered_df["Customer.Name"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
col2.metric("📈 Total Profit", f"${total_profit:,.2f}")
col3.metric("🧾 Total Orders", f"{total_orders:,}")
col4.metric("👥 Total Customers", f"{total_customers:,}")

st.markdown("---")

# ============================================================
# Sales and Profit by Category
# ============================================================

category_summary = (
    filtered_df.groupby("Category", as_index=False)[["Sales", "Profit"]]
    .sum()
    .sort_values(by="Sales", ascending=False)
)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        fig_sales_category = px.bar(
            category_summary,
            x="Category",
            y="Sales",
            title="Total Sales by Category",
            text_auto=True,
            color="Category",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig_sales_category = style_chart(
            fig_sales_category,
            "Total Sales by Category"
        )

        st.plotly_chart(fig_sales_category, use_container_width=True, theme=None)

with col2:
    with st.container(border=True):
        fig_profit_category = px.bar(
            category_summary,
            x="Category",
            y="Profit",
            title="Total Profit by Category",
            text_auto=True,
            color="Category",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        fig_profit_category = style_chart(
            fig_profit_category,
            "Total Profit by Category"
        )

        st.plotly_chart(fig_profit_category, use_container_width=True, theme=None)

# ============================================================
# Profit by Region and Sales by Segment
# ============================================================

region_profit = (
    filtered_df.groupby("Region", as_index=False)["Profit"]
    .sum()
    .sort_values(by="Profit", ascending=False)
)

segment_sales = (
    filtered_df.groupby("Segment", as_index=False)["Sales"]
    .sum()
    .sort_values(by="Sales", ascending=False)
)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        fig_profit_region = px.bar(
            region_profit,
            x="Region",
            y="Profit",
            title="Profit by Region",
            text_auto=True,
            color="Profit",
            color_continuous_scale="Blues"
        )

        fig_profit_region = style_chart(
            fig_profit_region,
            "Profit by Region"
        )

        st.plotly_chart(fig_profit_region, use_container_width=True, theme=None)

with col2:
    with st.container(border=True):
        fig_segment_sales = px.pie(
            segment_sales,
            names="Segment",
            values="Sales",
            title="Sales by Segment",
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        fig_segment_sales.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black", size=13),
            title=dict(
                text="<b>Sales by Segment</b>",
                font=dict(size=26, color="black"),
                x=0.02,
                xanchor="left"
            ),
            legend=dict(font=dict(color="black", size=12))
        )

        st.plotly_chart(fig_segment_sales, use_container_width=True, theme=None)

# ============================================================
# Top 5 Customers by Sales
# ============================================================

top_customers = (
    filtered_df.groupby("Customer.Name", as_index=False)["Sales"]
    .sum()
    .sort_values(by="Sales", ascending=False)
    .head(5)
)

with st.container(border=True):
    fig_top_customers = px.bar(
        top_customers,
        x="Sales",
        y="Customer.Name",
        orientation="h",
        title="Top 5 Customers by Sales",
        text_auto=True,
        color="Sales",
        color_continuous_scale="Teal"
    )

    fig_top_customers = style_chart(
        fig_top_customers,
        "Top 5 Customers by Sales"
    )

    fig_top_customers.update_layout(
        yaxis_categoryorder="total ascending"
    )

    st.plotly_chart(fig_top_customers, use_container_width=True, theme=None)

# ============================================================
# Sales and Profit Over Time
# ============================================================

time_data = (
    filtered_df.groupby("Order.Date", as_index=False)[["Sales", "Profit"]]
    .sum()
    .sort_values("Order.Date")
)

with st.container(border=True):
    fig_time = px.line(
        time_data,
        x="Order.Date",
        y=["Sales", "Profit"],
        title="Sales and Profit Trend Over Time",
        markers=True
    )

    fig_time = style_chart(
        fig_time,
        "Sales and Profit Trend Over Time"
    )

    st.plotly_chart(fig_time, use_container_width=True, theme=None)

# ============================================================
# Sales vs Profit Scatter Plot
# ============================================================

with st.container(border=True):
    fig_scatter = px.scatter(
        filtered_df,
        x="Sales",
        y="Profit",
        color="Category",
        size="Quantity",
        hover_data=["Customer.Name", "Region", "Sub.Category"],
        title="Sales vs Profit Analysis",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig_scatter = style_chart(
        fig_scatter,
        "Sales vs Profit Analysis"
    )

    st.plotly_chart(fig_scatter, use_container_width=True, theme=None)


# ============================================================
# Download Filtered Data
# ============================================================

with st.expander("📋 View Filtered Dataset"):
    st.dataframe(filtered_df.head(100), use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name="filtered_superstore_data.csv",
        mime="text/csv"
    )

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown("### ✅ Dashboard Created by Muhammad Sufyan Shahid")
st.markdown("Data Science & Analytics Internship Project - DevelopersHub Corporation")