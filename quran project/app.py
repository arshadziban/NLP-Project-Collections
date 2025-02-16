from flask import Flask, request, render_template, jsonify
import pandas as pd

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("quran_dataset_cleaned.csv")

# Function to find all matches
def find_matching_verses(query):
    # Filter verses that contain the query
    matches = df[df["Processed_Text"].str.contains(query, case=False, na=False)]
    return matches.to_dict(orient="records")  # Convert to list of dictionaries

# Fetch suggestions based on query
@app.route('/suggestions', methods=['GET'])
def suggestions():
    query = request.args.get('q', '')
    suggestions = find_matching_verses(query)
    return jsonify({"suggestions": suggestions})

@app.route("/", methods=["GET", "POST"])
def search_verse():
    results = []
    selected_verse = None

    if request.method == "POST":
        query = request.form["query"]
        results = find_matching_verses(query)
    elif request.args.get("verse_id"):
        verse_id = request.args.get("verse_id")
        selected_verse = df.loc[int(verse_id)].to_dict()  # Fetch the full verse details by index

    return render_template("index.html", query=request.form.get("query"), results=results, selected_verse=selected_verse)

if __name__ == "__main__":
    app.run(debug=True)