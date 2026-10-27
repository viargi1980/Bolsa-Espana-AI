import pandas as pd
from pathlib import Path
from datetime import datetime

# ==========================================
# CARGAR RANKING
# ==========================================

ranking = pd.read_csv(
    "ranking_hoy.csv"
)

ranking = ranking.sort_values(
    "probabilidad",
    ascending=False
)

ranking["probabilidad"] = (
    ranking["probabilidad"] * 100
).round(2).astype(str) + " %"

ranking = ranking.rename(
    columns={
        "ticker": "Ticker",
        "probabilidad": "Probabilidad de subir >3%"
    }
)

# ==========================================
# RANKING VISUAL
# ==========================================

ranking.insert(
    0,
    "Ranking",
    [
        "🥇🥇🥇" if i == 0 else
        "🥈🥈" if i == 1 else
        "🥉" if i == 2 else
        f"#{i+1}"
        for i in range(len(ranking))
    ]
)

# ==========================================
# DATOS CABECERA
# ==========================================

top1 = ranking.iloc[0]["Ticker"]
top2 = ranking.iloc[1]["Ticker"]
top3 = ranking.iloc[2]["Ticker"]

num_acciones = len(ranking)

fecha_actualizacion = datetime.now().strftime(
    "%d/%m/%Y %H:%M"
)

# ==========================================
# SEÑALES
# ==========================================

ranking["Señal"] = ranking[
    "Probabilidad de subir >3%"
].str.replace(" %", "").astype(float).apply(
    lambda x:
    "🟢 Alta Probabilidad" if x >= 70 else
    "🟡 Vigilar" if x >= 60 else
    "⚪ Esperar"
)

tabla = ranking.to_html(
    index=False,
    classes="tabla",
    escape=False
)

# ==========================================
# HTML
# ==========================================

html = f"""
<!DOCTYPE html>
<html lang="es">

<head>

<meta charset="UTF-8">

<title>Bolsa España AI</title>

<style>

* {{
    margin:0;
    padding:0;
    box-sizing:border-box;
}}

body {{
    background:linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color:white;
    font-family:Segoe UI, Arial, sans-serif;
}}

.header {{
    text-align:center;
    padding:60px 20px;
}}

.header h1 {{
    font-size:60px;
    margin-bottom:15px;
}}

.header p {{
    color:#cbd5e1;
    font-size:18px;
}}

.badge {{
    display:inline-block;
    margin-top:20px;
    padding:10px 20px;
    border-radius:30px;
    background:#22c55e;
    font-weight:bold;
}}

.stats {{
    display:flex;
    justify-content:center;
    gap:20px;
    flex-wrap:wrap;
    margin-bottom:40px;
}}

.stat {{
    background:#1e293b;
    padding:25px;
    width:250px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 10px 20px rgba(0,0,0,0.4);
}}

.stat h2 {{
    font-size:32px;
    color:#60a5fa;
}}

.card {{
    width:95%;
    max-width:1400px;
    margin:auto;
    margin-bottom:40px;
    background:#1e293b;
    padding:25px;
    border-radius:20px;
    box-shadow:0 10px 30px rgba(0,0,0,.4);
}}

.card h2 {{
    margin-bottom:20px;
}}

table {{
    width:100%;
    border-collapse:collapse;
}}

th {{
    background:#2563eb;
    color:white;
    padding:14px;
}}

td {{
    padding:12px;
    border-bottom:1px solid #334155;
}}

tr:nth-child(even) {{
    background:#182436;
}}

tr:hover {{
    background:#22314a;
}}

.footer {{
    text-align:center;
    color:#94a3b8;
    padding:40px;
}}

.logo {{
    font-size:80px;
}}

</style>

</head>

<body>

<div class="header">

<div class="logo">📈</div>

<h1>Bolsa España AI</h1>

<p>
Inteligencia Artificial para detectar oportunidades bursátiles
</p>

<div class="badge">
📅 Actualizado: {fecha_actualizacion}
</div>

</div>

<div class="stats">

<div class="stat">
<h2>166k+</h2>
<p>Registros históricos</p>
</div>

<div class="stat">
<h2>{num_acciones}</h2>
<p>Acciones españolas</p>
</div>

<div class="stat">
<h2>3%</h2>
<p>Objetivo de subida</p>
</div>

<div class="stat">
<h2>IA</h2>
<p>Predicción diaria</p>
</div>

</div>

<div class="card">

<h2>🔥 Top oportunidades de hoy</h2>

<p style="font-size:20px; line-height:2;">
🥇 {top1}<br>
🥈 {top2}<br>
🥉 {top3}
</p>

</div>

<div class="card">

<h2>🎯 Objetivo del modelo</h2>

<p>
La probabilidad indica la posibilidad estimada de que una acción
suba más de un 3% en la próxima sesión bursátil.
</p>

</div>

<div class="card">

<h2>🏆 Ranking de oportunidades</h2>

{tabla}

</div>

<div class="card">

<h2>📊 Interpretación</h2>

<ul style="line-height:2;padding-left:25px;">
<li>🟢 > 70% → Oportunidad muy fuerte</li>
<li>🟡 60% - 70% → Señal interesante</li>
<li>⚪ < 60% → Vigilancia</li>
</ul>

</div>

<div class="footer">

🤖 Generado automáticamente por Bolsa España AI<br>
📅 Última actualización: {fecha_actualizacion}

</div>

</body>

</html>
"""

# ==========================================
# CREAR CARPETA DOCS
# ==========================================

Path("docs").mkdir(
    exist_ok=True
)

# ==========================================
# GUARDAR HTML
# ==========================================

ruta_html = Path("docs/index.html")

with open(
    ruta_html,
    "w",
    encoding="utf-8"
) as f:
    f.write(html)

print()
print("✅ WEB GENERADA CORRECTAMENTE")
print()
print(f"📄 Archivo: {ruta_html.resolve()}")