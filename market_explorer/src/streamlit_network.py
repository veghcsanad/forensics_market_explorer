import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from itertools import combinations
from networkx.algorithms.community import greedy_modularity_communities
from scipy.stats import wasserstein_distance
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from bokeh.io import output_notebook, show, save
from bokeh.models import Plot, Range1d, Circle, HoverTool, MultiLine, ColumnDataSource
from bokeh.plotting import from_networkx, figure
from bokeh.palettes import Category20
from streamlit_bokeh import streamlit_bokeh


def page_config(title, layout="wide"):
    st.set_page_config(layout=layout, page_title=title, page_icon="🕸️")
    st.title(title)

def upload_file():
    uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV file", type="csv")
    return uploaded_file

def process_data(file):
    df = pd.read_csv(file)
    if "Strain" not in df:
        st.error("CSV file must contain a 'Strain' column.")
        return None
    if "Description" not in df:
        st.error("CSV file must contain a 'Description' column.")
        return None
    if "price_in_eur" not in df:
        st.error("CSV file must contain a 'price_in_eur' column.")
        return None
    if "seller" not in df:
        st.error("CSV file must contain a 'seller' column.")
        return None
    if "destination" not in df:
        st.error("CSV file must contain a 'destination' column.")
        return None
    if "Effects" not in df:
        st.error("CSV file must contain an 'Effects' column.")
        return None
    if "Flavor" not in df:
        st.error("CSV file must contain a 'Flavor' column.")
        return None

    grouped = df.groupby("seller").agg({
        "Strain": lambda x: ', '.join(set(str(i) for i in x if pd.notnull(i))),
        "name": lambda x: ' '.join(set(str(i) for i in x if pd.notnull(i))),
        "price_in_eur": list,
        "destination": lambda x: ', '.join(set(str(i) for i in x if pd.notnull(i))),
        "Effects": lambda x: ', '.join(set(str(i) for i in x if pd.notnull(i))),
        "Flavor": lambda x: ', '.join(set(str(i) for i in x if pd.notnull(i))),
    }).reset_index()
    grouped["price_in_eur"] = grouped["price_in_eur"].apply(
        lambda x: [
            round(float(str(i).replace(' ', '').replace(',', '.')), 2)
            for i in x if pd.notnull(i)
        ]
    )
    grouped = grouped.rename(columns={"name": "Description"})
    return grouped

def similarity_weights_sidebar():
    st.sidebar.header("🔧 Similarity Weights")
    weight_strain = st.sidebar.slider("Strain", 0.0, 1.0, 0.2)
    weight_desc = st.sidebar.slider("Description", 0.0, 1.0, 0.3)
    weight_price = st.sidebar.slider("Price", 0.0, 1.0, 0.1)
    weight_dest = st.sidebar.slider("Destination", 0.0, 1.0, 0.1)
    weight_effects = st.sidebar.slider("Effects", 0.0, 1.0, 0.15)
    weight_flavor = st.sidebar.slider("Flavor", 0.0, 1.0, 0.15)
    threshold = st.sidebar.slider("🔗 Similarity Threshold", 0.0, 1.0, 0.6)
    remove_isolates = st.sidebar.checkbox("Remove Isolated Nodes", value=False)
    return weight_strain, weight_desc, weight_price, weight_dest, weight_effects, weight_flavor, threshold, remove_isolates

def compute_tfidf_similarity(df, column):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[column])
    return cosine_similarity(tfidf_matrix)

def compute_jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def compute_wasserstein_similarity(df, column):
    prices = df[column]
    n = len(prices)
    similarity_matrix = np.zeros((n, n))

    # Normalize Wasserstein distances to similarities (e.g., 1 / (1 + dist))
    for i in range(n):
        for j in range(n):
            dist = wasserstein_distance(prices[i], prices[j])
            similarity_matrix[i, j] = 1 / (1 + dist)  # Invert to get similarity

    return similarity_matrix

def compute_overall_similarity(df, weight_strain, weight_desc, weight_price, weight_dest, weight_effects, weight_flavor):
    # Categorical features via Jaccard
    def pairwise_jaccard(col):
        sim = np.zeros((len(df), len(df)))
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                s = compute_jaccard_similarity(df.at[i, col], df.at[j, col])
                sim[i, j] = sim[j, i] = s
            sim[i, i] = 1.0
        return sim

    strain_sim = pairwise_jaccard("Strain")
    dest_sim = pairwise_jaccard("destination")
    effects_sim = pairwise_jaccard("Effects")
    flavor_sim = pairwise_jaccard("Flavor")

    # Description: average of TF-IDF and embedding similarity
    desc_sim = compute_tfidf_similarity(df, "Description")

    # Price via Wasserstein distance
    price_sim = compute_wasserstein_similarity(df, "price_in_eur")

    # Weighted combination
    overall_sim = (
        weight_strain * strain_sim +
        weight_desc * desc_sim +
        weight_price * price_sim +
        weight_dest * dest_sim +
        weight_effects * effects_sim +
        weight_flavor * flavor_sim
    )
    # Normalize the overall similarity matrix
    overall_sim = (overall_sim - np.min(overall_sim)) / (np.max(overall_sim) - np.min(overall_sim))

    return overall_sim

def create_graph(df, similarity_matrix, threshold):
    G = nx.Graph()
    sellers = df["seller"].tolist()
    G.add_nodes_from(sellers)

    for i in range(len(sellers)):
        for j in range(i + 1, len(sellers)):
            if similarity_matrix[i, j] >= threshold:
                G.add_edge(sellers[i], sellers[j], weight=similarity_matrix[i, j])

    return G

def visualize_graph(G, df, searched_node=None):
    title = "Black Market Seller Network"
    HOVER_TOOLTIPS = [("Seller", "@index")]

    plot = figure(
        title=title,
        tooltips=HOVER_TOOLTIPS,
        tools="pan,wheel_zoom,save,reset",
        active_scroll='wheel_zoom',
        x_range=Range1d(-10, 10),
        y_range=Range1d(-10, 10),
    )

    # Compute layout
    pos = nx.spring_layout(G, scale=10, center=(0, 0))

    # Attach attributes from df to graph nodes
    for node in G.nodes:
        if node in df.index:
            for col in df.columns:
                G.nodes[node][col] = df.loc[node, col]

    # Node attributes
    node_attrs = df[df["seller"].isin(G.nodes)].copy()
    node_attrs["index"] = node_attrs.index

    # Communities
    communities = get_communities(G)
    node_attrs["community"] = node_attrs["seller"].map(communities)
    unique_communities = sorted(node_attrs["community"].dropna().unique())
    community_colors = {comm: Category20[20][i % 20] for i, comm in enumerate(unique_communities)}
    node_attrs["color"] = node_attrs["community"].map(community_colors)

    # Node size by number of strains
    node_attrs["size"] = node_attrs["Strain"].apply(lambda x: len([s for s in str(x).split(",") if s.strip()]))

    # Filter by searched_node
    if searched_node and searched_node in communities:
        searched_community = communities[searched_node]
        nodes_to_keep = node_attrs[node_attrs["community"] == searched_community]["seller"].tolist()
        G = G.subgraph(nodes_to_keep)
        pos = nx.spring_layout(G, scale=10, center=(0, 0))  # Recompute layout
        node_attrs = node_attrs[node_attrs["seller"].isin(G.nodes)].copy()

    # Recompute sizes and colors after potential filtering
    sizes = node_attrs["size"].values.astype(float)
    scaled_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min()) if sizes.max() > sizes.min() else np.zeros_like(sizes)
    colors = node_attrs["color"].copy()

    # Create network graph
    network_graph = from_networkx(G, pos)
    network_graph.node_renderer.data_source.data["size"] = scaled_sizes
    network_graph.node_renderer.data_source.data["color"] = colors.values
    network_graph.node_renderer.data_source.data["seller"] = node_attrs["seller"].tolist()

    network_graph.node_renderer.glyph = Circle(radius=0.1, fill_color="color", name="seller")
    if searched_node:
        # Highlight the searched node in red
        seller_indices = network_graph.node_renderer.data_source.data.get("index", node_attrs["seller"].tolist())
        node_colors = ["#0000FF"] * len(seller_indices)  # Default to blue
        for idx, seller in enumerate(seller_indices):
            if seller == searched_node:
                node_colors[idx] = "#FF0000"  # Red
        network_graph.node_renderer.data_source.data["color"] = node_colors

    network_graph.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=1.5)

    plot.renderers.append(network_graph)
    streamlit_bokeh(plot, use_container_width=True, theme="streamlit", key=f"network_graph_{searched_node}")

def get_communities(G):
    partition = greedy_modularity_communities(G)
    communities = {}
    for i, community in enumerate(partition):
        for node in community:
            communities[node] = i
    return communities

def remove_isolated_nodes(G):
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    return G

# Main app
def main():
    page_config(title="🕸️ Black Market Seller Network Explorer", layout="wide")
    uploaded_file = upload_file()
    if uploaded_file:
        df = process_data(uploaded_file)
        if df is None:
            st.stop()
        weight_strain, weight_desc, weight_price, weight_dest, weight_effects, weight_flavor, threshold, remove_isolates = similarity_weights_sidebar()
        similarity_matrix = compute_overall_similarity(df, weight_strain, weight_desc, weight_price, weight_dest, weight_effects, weight_flavor)
        seller_network = create_graph(df, similarity_matrix, threshold)
        if remove_isolates:
            seller_network = remove_isolated_nodes(seller_network)
        tab1, tab2, tab3, tab4 = st.tabs(["Investigate Data", "Network View", "Look for seller", "Explore communities"])
        with tab1:
            st.dataframe(df, use_container_width=True)
        with tab2:
            visualize_graph(seller_network, df)
            partition = get_communities(seller_network)
            st.markdown(
            f"**🧠 Nodes:** {seller_network.number_of_nodes()} | "
            f"**🕸️ Edges:** {seller_network.number_of_edges()} | "
            f"**🔎 Communities:** {len(set(partition.values())) if partition else 0}"
            )
        with tab3:
            # Create a searchable dropdown menu of unique sellers
            # Only show sellers that are not isolated in the current network
            non_isolates = list(seller_network.nodes)
            search_seller = st.selectbox(
                "🔍 Search for a seller",
                options=sorted(non_isolates),
                index=None,
                placeholder="Select a seller"
            )

            if search_seller:
                seller_data = df[df["seller"] == search_seller]
                st.dataframe(seller_data, use_container_width=True)
                visualize_graph(seller_network, df, searched_node=search_seller)
        with tab4:
            # Create a searchable dropdown menu of unique communities
            # Show community selectbox with sellers in parentheses
            community_options = []
            for comm in sorted(set(partition.values())):
                sellers_in_comm = [node for node, c in partition.items() if c == comm]
                label = f"{comm} ({', '.join(sellers_in_comm)})"
                community_options.append((label, comm))
            search_community = st.selectbox(
                "🔍 Select a community",
                options=community_options,
                index=None,
                format_func=lambda x: x[0] if x else "",
                placeholder="Select a community"
            )
            if search_community:
                search_community = search_community[1]

            if search_community:
                community_nodes = [node for node, comm in partition.items() if comm == search_community]
                community_data = df[df["seller"].isin(community_nodes)]
                st.dataframe(community_data, use_container_width=True)
                visualize_graph(seller_network, df, searched_node=community_data["seller"].tolist()[0])
                # Community statistics
                st.subheader("Community Statistics")

                # Most common strains
                all_strains = community_data["Strain"].str.split(",").explode().str.strip()
                common_strains = all_strains.value_counts().head(5)

                # Average price
                all_prices = community_data["price_in_eur"].explode()
                avg_price = pd.to_numeric(all_prices, errors="coerce").mean()

                # Most common flavors
                all_flavors = community_data["Flavor"].str.split(",").explode().str.strip()
                common_flavors = all_flavors.value_counts().head(5)

                # Most common effects
                all_effects = community_data["Effects"].str.split(",").explode().str.strip()
                common_effects = all_effects.value_counts().head(5)

                st.markdown(f"**Most common strains:** {', '.join(common_strains.index)}")
                st.markdown(f"**Average price:** €{avg_price:.2f}")
                st.markdown(f"**Common flavors:** {', '.join(common_flavors.index)}")
                st.markdown(f"**Common effects:** {', '.join(common_effects.index)}")

                # Central/influential sellers
                subgraph = seller_network.subgraph(community_nodes)
                betweenness = nx.betweenness_centrality(subgraph)

                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]

                st.markdown("**Top sellers:**")
                for seller, score in top_betweenness:
                    st.markdown(f"- {seller} (betweenness score: {score:.3f})")

if __name__ == "__main__":
    main()