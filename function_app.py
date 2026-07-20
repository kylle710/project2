"""
Phase 2 Cloud Dashboard — Azure Function (Backend)
Owner: Alan 

This function replaces the Phase 1 Azurite-based lambda_function.py.
Instead of reading All_Diets.csv from the local Azurite emulator, it reads
the same dataset from a real Azure Blob Storage container.

This matches the real frontend template (UI-for-project2.html), which has
THREE separate "API Data Interaction" buttons, so this backend exposes
three routes rather than one bundled endpoint:

    GET /api/nutritional-insights   -> bar chart, heatmap, pie chart
    GET /api/recipes                -> paginated recipe table + scatter points
    GET /api/clusters                -> KMeans clustering of recipes by macros

All three accept ?diet_type=<name> (or "all") to match the dropdown filter.
/api/recipes also accepts ?search=<text>, ?page=<n>, ?page_size=<n>.
/api/clusters also accepts ?k=<n> (number of clusters, default 4).

Environment variables required (set these in the Function App's
Configuration / Application Settings — see RUNBOOK.md):
    AZURE_STORAGE_CONNECTION_STRING
    BLOB_CONTAINER_NAME   (default: "datasets")
    BLOB_NAME             (default: "All_Diets.csv")
"""

import io
import json
import logging
import math
import os
import time

import azure.functions as func
import pandas as pd
from azure.storage.blob import BlobServiceClient
from sklearn.cluster import KMeans

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}

# Module-level cache so we don't re-download the blob on every single
# request while testing. In a real production app you'd add a TTL;
# for this project, re-deploying clears it, which is good enough.
_dataset_cache: pd.DataFrame | None = None


def load_dataset() -> pd.DataFrame:
    """Download All_Diets.csv from Azure Blob Storage and load it into a DataFrame."""
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache.copy()

    connect_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container_name = os.environ.get("BLOB_CONTAINER_NAME", "datasets")
    blob_name = os.environ.get("BLOB_NAME", "All_Diets.csv")

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    stream = blob_client.download_blob().readall()
    df = pd.read_csv(io.BytesIO(stream))

    # Same cleaning approach as Phase 1: fill missing numeric values with the column mean
    numeric_cols = ["Protein(g)", "Carbs(g)", "Fat(g)"]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    _dataset_cache = df
    return df.copy()


def apply_filters(df: pd.DataFrame, diet_type: str | None = None, search: str | None = None) -> pd.DataFrame:
    """Shared filtering logic used by all three routes, so they stay consistent."""
    if diet_type and diet_type.lower() != "all":
        df = df[df["Diet_type"].str.lower() == diet_type.lower()]
    if search:
        df = df[df["Recipe_name"].str.contains(search, case=False, na=False)]
    return df


def json_response(payload: dict, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(payload),
        mimetype="application/json",
        headers=CORS_HEADERS,
        status_code=status,
    )


def error_response(message: str, status: int = 400, detail: str | None = None) -> func.HttpResponse:
    body = {"error": message}
    if detail:
        body["detail"] = detail
    return json_response(body, status)


# ---------------------------------------------------------------------------
# Route 1: "Get Nutritional Insights" button
# Feeds: Bar Chart, Heatmap, Pie Chart
# ---------------------------------------------------------------------------
@app.route(route="nutritional-insights", methods=["GET"])
def nutritional_insights(req: func.HttpRequest) -> func.HttpResponse:
    start = time.perf_counter()
    try:
        diet_type = req.params.get("diet_type")
        df = load_dataset()
        filtered = apply_filters(df, diet_type=diet_type)

        if filtered.empty:
            raise ValueError(f"No data found for diet_type='{diet_type}'")

        # Bar chart: average macros per diet type (respects the filter)
        avg_macros = filtered.groupby("Diet_type")[["Protein(g)", "Carbs(g)", "Fat(g)"]].mean().round(2)
        bar_chart = {
            "labels": avg_macros.index.tolist(),
            "protein": avg_macros["Protein(g)"].tolist(),
            "carbs": avg_macros["Carbs(g)"].tolist(),
            "fat": avg_macros["Fat(g)"].tolist(),
        }

        # Heatmap: correlation between the three macronutrients (respects the filter)
        corr = filtered[["Protein(g)", "Carbs(g)", "Fat(g)"]].corr().round(2)
        heatmap = {
            "metrics": ["Protein", "Carbs", "Fat"],
            "values": corr.values.tolist(),
        }

        # Pie chart: recipe count distribution across ALL diet types — intentionally
        # uses the unfiltered dataset so the chart stays meaningful even when a
        # specific diet is selected elsewhere on the dashboard.
        counts = df["Diet_type"].value_counts()
        pie_chart = {
            "labels": counts.index.tolist(),
            "values": counts.values.tolist(),
        }

        result = {
            "barChart": bar_chart,
            "heatmap": heatmap,
            "pieChart": pie_chart,
            "filterOptions": {"dietTypes": sorted(df["Diet_type"].unique().tolist())},
            "rowCount": int(len(filtered)),
        }
        result["executionTimeMs"] = round((time.perf_counter() - start) * 1000, 2)
        return json_response(result)

    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logging.exception("Error processing nutritional insights")
        return error_response("Internal server error", 500, str(e))


# ---------------------------------------------------------------------------
# Route 2: "Get Recipes" button (+ Search box, Diet dropdown, Pagination)
# Feeds: Scatter Plot, the implicit recipe table, Pagination controls
# ---------------------------------------------------------------------------
@app.route(route="recipes", methods=["GET"])
def get_recipes(req: func.HttpRequest) -> func.HttpResponse:
    start = time.perf_counter()
    try:
        diet_type = req.params.get("diet_type")
        search = req.params.get("search")

        try:
            page = int(req.params.get("page", 1))
            page_size = int(req.params.get("page_size", 10))
        except ValueError:
            raise ValueError("page and page_size must be integers")

        if page < 1 or page_size < 1:
            raise ValueError("page and page_size must be positive integers")

        df = load_dataset()
        filtered = apply_filters(df, diet_type=diet_type, search=search)
        total = len(filtered)
        total_pages = max(1, math.ceil(total / page_size))

        start_idx = (page - 1) * page_size
        page_df = filtered.iloc[start_idx: start_idx + page_size]

        recipes = [
            {
                "recipeName": row["Recipe_name"],
                "dietType": row["Diet_type"],
                "cuisineType": row["Cuisine_type"],
                "protein": round(float(row["Protein(g)"]), 2),
                "carbs": round(float(row["Carbs(g)"]), 2),
                "fat": round(float(row["Fat(g)"]), 2),
            }
            for _, row in page_df.iterrows()
        ]

        # Scatter plot needs the relationship across the whole filtered set, not
        # just the current page — sample up to 300 points so the chart stays
        # representative without shipping the entire dataset on every request.
        sample_n = min(300, total)
        sample_df = filtered.sample(n=sample_n, random_state=42) if sample_n > 0 else filtered
        scatter_points = [
            {
                "recipeName": row["Recipe_name"],
                "dietType": row["Diet_type"],
                "protein": round(float(row["Protein(g)"]), 2),
                "carbs": round(float(row["Carbs(g)"]), 2),
            }
            for _, row in sample_df.iterrows()
        ]

        result = {
            "recipes": recipes,
            "page": page,
            "pageSize": page_size,
            "totalRecipes": total,
            "totalPages": total_pages,
            "scatterPoints": scatter_points,
        }
        result["executionTimeMs"] = round((time.perf_counter() - start) * 1000, 2)
        return json_response(result)

    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logging.exception("Error processing recipes request")
        return error_response("Internal server error", 500, str(e))


# ---------------------------------------------------------------------------
# Route 3: "Get Clusters" button
# Groups recipes by macronutrient profile using KMeans (unsupervised — does
# not use the Diet_type label), so it surfaces nutritional groupings that
# may cut across the labeled diet types.
# ---------------------------------------------------------------------------
@app.route(route="clusters", methods=["GET"])
def get_clusters(req: func.HttpRequest) -> func.HttpResponse:
    start = time.perf_counter()
    try:
        diet_type = req.params.get("diet_type")

        try:
            k = int(req.params.get("k", 4))
        except ValueError:
            raise ValueError("k must be an integer")

        if k < 2 or k > 10:
            raise ValueError("k must be between 2 and 10")

        df = load_dataset()
        filtered = apply_filters(df, diet_type=diet_type)

        if len(filtered) < k:
            raise ValueError(f"Not enough recipes ({len(filtered)}) to form {k} clusters")

        features = filtered[["Protein(g)", "Carbs(g)", "Fat(g)"]].to_numpy()
        means = features.mean(axis=0)
        stds = features.std(axis=0)
        stds[stds == 0] = 1.0  # avoid divide-by-zero if a column is constant
        scaled = (features - means) / stds

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)

        filtered = filtered.copy()
        filtered["cluster_id"] = labels

        cluster_summaries = []
        for cluster_id in range(k):
            subset = filtered[filtered["cluster_id"] == cluster_id]
            if subset.empty:
                continue
            cluster_summaries.append({
                "clusterId": int(cluster_id),
                "size": int(len(subset)),
                "avgProtein": round(float(subset["Protein(g)"].mean()), 2),
                "avgCarbs": round(float(subset["Carbs(g)"].mean()), 2),
                "avgFat": round(float(subset["Fat(g)"].mean()), 2),
                "dominantDietType": subset["Diet_type"].mode().iloc[0],
            })

        sample_n = min(300, len(filtered))
        sample_df = filtered.sample(n=sample_n, random_state=42)
        points = [
            {
                "recipeName": row["Recipe_name"],
                "dietType": row["Diet_type"],
                "cuisineType": row["Cuisine_type"],
                "protein": round(float(row["Protein(g)"]), 2),
                "carbs": round(float(row["Carbs(g)"]), 2),
                "fat": round(float(row["Fat(g)"]), 2),
                "clusterId": int(row["cluster_id"]),
            }
            for _, row in sample_df.iterrows()
        ]

        result = {
            "k": k,
            "clusters": cluster_summaries,
            "points": points,
        }
        result["executionTimeMs"] = round((time.perf_counter() - start) * 1000, 2)
        return json_response(result)

    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logging.exception("Error processing clusters request")
        return error_response("Internal server error", 500, str(e))