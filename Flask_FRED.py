from flask import Flask, render_template

app = Flask("Flask_FRED")

@app.route("/")
def index():
    base_url = "https://fred.stlouisfed.org/graph/graph-landing.php?g="
    width_param = "&width=100%"
    codes = [ "1dSVc", "1dsRu", "1dVQ6","1dFFw", "1dsBt", "1dHWE","1dHW3", "1dYnv", "1dYo0", "1dYo9", "1dYos", "1dYpa", "1dYpq", "1dYpB", "1dYpL", "1dYpS", "1dYpX", "1dYqi", "1dYqn", "1dYqA", "1dYqL","1dYqR", "1dYqX", "1dYqZ", "1dYr4", "1dYrD", "1dNYU", "1dYsN", "1dYsT", "1dYsZ", "1dYt9", "1dYtl", "1dYts", "16n5n", "1dYz8", "1dYzF", "1dYBL", "1dYzi", "1dYAa", "1dYAg"  ]  # FRED 코드 리스트

    embed_codes = []
    for code in codes:
        full_src = f"{base_url}{code}{width_param}"
        embed_codes.append(f'<iframe src="{full_src}" scrolling="no" frameborder="0" style="overflow:hidden; object-fit="contain" allowTransparency="true" loading="lazy"></iframe>')

    return render_template("template_FRED.html", embed_codes=embed_codes)


# -----------------------------------------------------------

@app.route("/print_grid")
def print_grid():
    # 이 부분에서 A4 용지에 8개의 그리드를 출력하는 작업을 수행
    # 해당 작업을 위한 추가 코드 작성

    return "Grid 출력 페이지"

# -----------------------------------------------------------


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)